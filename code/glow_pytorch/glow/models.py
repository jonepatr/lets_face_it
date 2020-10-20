import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from glow_pytorch.glow import modules, thops, utils


class ModalityEncoder(nn.Module):
    def __init__(self, input_size, params):
        super().__init__()

        if params["dropout"] > 0:
            self.dropout = nn.Dropout(params["dropout"])
        else:
            self.dropout = None

        if params["enc"] == "rnn":
            self.encoder = nn.GRU(
                input_size=input_size,
                hidden_size=params["hidden_dim"],
                batch_first=True,
            )
            self.dim = params["hidden_dim"] * 2
        elif params["enc"] == "lstm":
            self.encoder = nn.LSTM(
                input_size=input_size,
                hidden_size=params["hidden_dim"],
                batch_first=True,
            )
            self.dim = params["hidden_dim"] * 2
        elif params["enc"] == "mlp":
            self.encoder = nn.Sequential(
                nn.Linear(input_size * params["history"], params["hidden_dim"]),
                nn.LeakyReLU(),
            )
            self.dim = params["hidden_dim"]
        elif params["enc"] == "cnn":
            self.encoder = nn.Conv1d(
                input_size,
                params["hidden_dim"],
                params["kernel_size"],
                padding=params["kernel_size"] // 2,
            )
            self.dim = input_size - params["kernel_size"] + 1
        elif params["enc"] == "none":
            self.encoder = None
            self.dim = input_size * params["history"]
        else:
            raise NotImplementedError()

    def forward(self, x):
        if self.training and self.dropout is not None:
            mask = self.dropout(torch.ones(x.shape[:2]).type_as(x)).unsqueeze(-1)
            x = x * mask

        if isinstance(self.encoder, nn.GRU):
            # This is only relevant for multi-gpu
            # self.encoder.flatten_parameters()
            seq, h_state = self.encoder(x)
            output = torch.cat([seq[:, -1, :], h_state[0]], dim=1)
        elif isinstance(self.encoder, nn.LSTM):
            # This is only relevant for multi-gpu
            # self.encoder.flatten_parameters()
            seq, (h_state, _) = self.encoder(x)
            output = torch.cat([seq[:, -1, :], h_state[0]], dim=1)
        elif isinstance(self.encoder, nn.Sequential):
            output = self.encoder(x.reshape(x.shape[0], -1))
        elif isinstance(self.encoder, nn.Conv1d):
            output = (
                self.encoder(x.permute(0, 2, 1)).permute(0, 2, 1).reshape(x.size(0), -1)
            )
        elif self.encoder is None:
            output = x.reshape(x.shape[0], -1)
        else:
            raise NotImplementedError()
        return output


class FeatureEncoder(nn.Module):
    def __init__(self, conditioning_hparams, data_hparams):
        super().__init__()

        self.use_frame_nb = conditioning_hparams["use_frame_nb"]
        self.p1_speech_history = conditioning_hparams["p1_speech"]["history"]
        self.p2_speech_history = conditioning_hparams["p2_speech"]["history"]
        self.p2_face_history = conditioning_hparams["p2_face"]["history"]
        cond_dim = conditioning_hparams["cond_dim"]

        frame_nb_dim = 1 if self.use_frame_nb else 0

        face_dim = utils.face_dim(data_hparams)
        speech_dim = data_hparams["speech_dim"]

        # Define conditioning for P1 face
        self.p1_face_encoder = ModalityEncoder(
            face_dim, conditioning_hparams["p1_face"]
        )
        self.dim = self.p1_face_encoder.dim

        # Define conditioning for P2 face
        if self.p2_face_history:
            self.p2_face_encoder = ModalityEncoder(
                face_dim, conditioning_hparams["p2_face"]
            )
            self.dim += self.p2_face_encoder.dim

        # Define conditioning for P1 speech
        if self.p1_speech_history:
            self.p1_speech_encoder = ModalityEncoder(
                speech_dim, conditioning_hparams["p1_speech"]
            )
            self.dim += self.p1_speech_encoder.dim

        # Define conditioning for P2 speech
        if self.p2_speech_history:
            self.p2_speech_encoder = ModalityEncoder(
                speech_dim, conditioning_hparams["p2_speech"]
            )
            self.dim += self.p2_speech_encoder.dim

        if self.use_frame_nb:
            self.dim += 1

    def forward(self, condition):
        cond_input = self.p1_face_encoder(condition["prev_p1_face"])

        if self.p2_face_history:
            p2_face_encoded = self.p2_face_encoder(condition["p2_face"])
            cond_input = torch.cat([cond_input, p2_face_encoded], dim=1)

        if self.p1_speech_history:
            p1_speech_encoded = self.p1_speech_encoder(condition["p1_speech"])
            cond_input = torch.cat([cond_input, p1_speech_encoded], dim=1)

        if self.p2_speech_history:
            p2_speech_encoded = self.p2_speech_encoder(condition["p2_speech"])
            cond_input = torch.cat([cond_input, p2_speech_encoded], dim=1)

        if self.use_frame_nb:
            cond_input = torch.cat([cond_input, condition["frame_nb"]], dim=1)

        return cond_input


class f_seq(nn.Module):
    """
    Transformation to be used in a coupling layer
    """

    def __init__(
        self,
        input_size,
        output_size,
        hidden_size,
        cond_dim,
        feature_encoder_dim,
        rnn_type,
    ):
        """
        input_size:             (glow) channels
        output_size:            output channels (ToDo: should be equal to input_size?)
        hidden_size:            size of the hidden layer of the NN
        cond_dim:               final dim. of the conditioning info
        feature_encoder_dim:    intermediate conditioning info dim. size
        rnn_type:               GRU/LSTM
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.output_size = output_size
        self.rnn_type = rnn_type

        if rnn_type == "gru":
            self.rnn = nn.GRUCell(
                input_size=input_size + cond_dim, hidden_size=hidden_size,
            )
        elif rnn_type == "lstm":
            self.rnn = nn.LSTMCell(
                input_size=input_size + cond_dim, hidden_size=hidden_size,
            )

        self.cond_transform = nn.Sequential(
            nn.Linear(feature_encoder_dim, cond_dim), nn.LeakyReLU(),
        )

        self.final_linear = modules.LinearZeros(hidden_size, output_size)
        self.hidden = None
        self.cell = None

    def init_rnn_hidden(self):
        """
        Setting it to None is the same as zeros, 
        except now we don't need to know any sizes
        """
        self.hidden = None
        self.cell = None

    def forward(self, z, condition):
        if self.rnn_type == "gru":
            self.hidden = self.rnn(
                torch.cat((z, self.cond_transform(condition)), dim=1), self.hidden
            )
        elif self.rnn_type == "lstm":
            self.hidden, self.cell = self.rnn(
                torch.cat((z, self.cond_transform(condition)), dim=1),
                (self.hidden, self.cell),
            )
        return self.final_linear(self.hidden)


class FlowStep(nn.Module):
    FlowCoupling = ["additive", "affine"]
    FlowPermutation = {
        "reverse": lambda obj, z, logdet, rev: (obj.reverse(z, rev), logdet),
        "shuffle": lambda obj, z, logdet, rev: (obj.shuffle(z, rev), logdet),
        "invconv": lambda obj, z, logdet, rev: obj.invconv(z, logdet, rev),
    }

    def __init__(
        self,
        in_channels,
        hidden_channels,
        cond_dim,
        actnorm_scale=1.0,
        flow_permutation="shuffle",
        flow_coupling="additive",
        LU_decomposed=False,
        L=1,
        K=1,
        scale_eps=1e-6,
        scale_logging=False,
        feature_encoder_dim=0,
        glow_rnn_type=None,
    ):
        # check configures
        assert (
            flow_coupling in FlowStep.FlowCoupling
        ), "flow_coupling should be in `{}`".format(FlowStep.FlowCoupling)

        assert (
            flow_permutation in FlowStep.FlowPermutation
        ), "float_permutation should be in `{}`".format(FlowStep.FlowPermutation.keys())

        super().__init__()

        self.flow_permutation = flow_permutation
        self.flow_coupling = flow_coupling

        # Custom
        self.scale = None
        self.scale_logging = scale_logging
        self.scale_eps = scale_eps
        self.L = L  # Which multiscale layer this module is in
        self.K = K  # Which step of flow in self.L

        # 1. actnorm
        self.actnorm = modules.ActNorm2d(in_channels, actnorm_scale)

        # 2. permute
        if flow_permutation == "invconv":
            self.invconv = modules.InvertibleConv1x1(
                in_channels, LU_decomposed=LU_decomposed
            )
        elif flow_permutation == "shuffle":
            self.shuffle = modules.Permute2d(in_channels, shuffle=True)
        else:
            self.reverse = modules.Permute2d(in_channels, shuffle=False)

        # 3. coupling
        if flow_coupling == "additive":
            self.f = f_seq(
                in_channels // 2,
                in_channels - in_channels // 2,
                hidden_channels,
                cond_dim,
                feature_encoder_dim,
                glow_rnn_type,
            )
        elif flow_coupling == "affine":
            if in_channels % 2 == 0:
                self.f = f_seq(
                    in_channels // 2,
                    in_channels,
                    hidden_channels,
                    cond_dim,
                    feature_encoder_dim,
                    glow_rnn_type,
                )
            else:
                self.f = f_seq(
                    in_channels // 2,
                    in_channels + 1,
                    hidden_channels,
                    cond_dim,
                    feature_encoder_dim,
                    glow_rnn_type,
                )

    def forward(self, input_, audio_features, logdet=None, reverse=False):
        if not reverse:
            return self.normal_flow(input_, audio_features, logdet)
        else:
            return self.reverse_flow(input_, audio_features, logdet)

    def normal_flow(self, input_, condition, logdet):
        """
        Forward path
        """

        # 1. actnorm
        z, logdet = self.actnorm(input_, logdet=logdet, reverse=False)

        # 2. permute
        z, logdet = FlowStep.FlowPermutation[self.flow_permutation](
            self, z.float(), logdet, False
        )

        # 3. coupling
        z1, z2 = thops.split_feature(z, "split")

        # Require some conditioning
        assert condition is not None

        if self.flow_coupling == "additive":
            z2 = z2 + self.f(z1, condition)
        elif self.flow_coupling == "affine":
            h = self.f(z1, condition)
            shift, scale = thops.split_feature(h, "cross")
            scale = torch.sigmoid(scale + 2.0).clamp(self.scale_eps)
            if self.scale_logging:
                self.scale = scale
            z2 = z2 + shift
            z2 = z2 * scale
            logdet = thops.sum(torch.log(scale), dim=[1]) + logdet
        z = thops.cat_feature(z1, z2)
        return z, logdet
        # ToDo: why do we return audio_features ?

    def reverse_flow(self, input_, condition, logdet):
        """
        Backward path
        """

        # 1.coupling
        z1, z2 = thops.split_feature(input_, "split")

        # Require some conditioning
        assert condition is not None

        if self.flow_coupling == "additive":
            z2 = z2 - self.f(z1, condition)
        elif self.flow_coupling == "affine":
            h = self.f(z1, condition)
            shift, scale = thops.split_feature(h, "cross")
            scale = torch.sigmoid(scale + 2.0).clamp(self.scale_eps)
            z2 = z2 / scale
            z2 = z2 - shift

            logdet = -thops.sum(torch.log(scale), dim=[1]) + logdet
        z = thops.cat_feature(z1, z2)
        # 2. permute
        z, logdet = FlowStep.FlowPermutation[self.flow_permutation](
            self, z, logdet, True
        )
        # 3. actnorm
        z, logdet = self.actnorm(z, logdet=logdet, reverse=True)
        return z, logdet

    def init_rnn_hidden(self):
        self.f.init_rnn_hidden()


class FlowNet(nn.Module):
    """
    The whole flow as visualized below
    """

    def __init__(
        self,
        C,
        hidden_channels,
        cond_dim,
        K,
        L,
        actnorm_scale=1.0,
        flow_permutation="invconv",
        flow_coupling="additive",
        LU_decomposed=False,
        scale_eps=1e-6,
        scale_logging=False,
        feature_encoder_dim=0,
        glow_rnn_type=None,
    ):
        """
                             K                                      K
        --> [Squeeze] -> [FlowStep] -> [Split] -> [Squeeze] -> [FlowStep]
               ^                           v
               |          (L - 1)          |
               + --------------------------+
        """
        super().__init__()
        self.layers = nn.ModuleList()
        self.output_shapes = []
        self.K = K
        self.L = L

        for l in range(L):

            # 2. K FlowStep
            for k in range(K):
                self.layers.append(
                    FlowStep(
                        in_channels=C,
                        hidden_channels=hidden_channels,
                        cond_dim=cond_dim,
                        actnorm_scale=actnorm_scale,
                        flow_permutation=flow_permutation,
                        flow_coupling=flow_coupling,
                        LU_decomposed=LU_decomposed,
                        L=l,
                        K=k,
                        scale_eps=scale_eps,
                        scale_logging=scale_logging,
                        feature_encoder_dim=feature_encoder_dim,
                        glow_rnn_type=glow_rnn_type,
                    )
                )
                self.output_shapes.append([-1, C])

    def forward(self, input_, condition, logdet=0.0, reverse=False, eps_std=None):
        # audio_features = self.conditionNet(audio_features)  # Spectrogram

        if not reverse:
            return self.encode(input_, condition, logdet)
        else:
            return self.decode(input_, condition, eps_std)

    def encode(self, z, condition, logdet=0.0):
        """
        Forward path
        """

        for layer, shape in zip(self.layers, self.output_shapes):
            z, logdet = layer(z, condition, logdet, reverse=False)
        return z, logdet

    def decode(self, z, condition, eps_std=None):
        """
        Backward path
        """

        logdet = 0.0

        for layer in reversed(self.layers):
            z, logdet = layer(z, condition, logdet, reverse=True)
        return z, logdet

    def init_rnn_hidden(self):
        for layer in self.layers:
            if isinstance(layer, FlowStep):
                layer.init_rnn_hidden()


class Glow(nn.Module):
    def __init__(self, hparams, feature_encoder_dim=0):
        super().__init__()
        self.flow = FlowNet(
            C=utils.face_dim(hparams.Data),
            hidden_channels=hparams.Glow["hidden_channels"],
            cond_dim=hparams.Conditioning["cond_dim"],
            K=hparams.Glow["K"],
            L=hparams.Glow["L"],
            actnorm_scale=hparams.Glow["actnorm_scale"],
            flow_permutation=hparams.Glow["flow_permutation"],
            flow_coupling=hparams.Glow["flow_coupling"],
            LU_decomposed=hparams.Glow["LU_decomposed"],
            scale_eps=hparams.Glow["scale_eps"],
            scale_logging=hparams.Validation["scale_logging"],
            feature_encoder_dim=feature_encoder_dim,
            glow_rnn_type=hparams.Glow["rnn_type"],
        )

    def forward(
        self,
        x=None,
        condition=None,
        z=None,
        eps_std=None,
        reverse=False,
        output_shape=None,
    ):

        if not reverse:
            return self.normal_flow(x, condition)
        else:
            return self.reverse_flow(z, condition, eps_std, output_shape)

    def normal_flow(self, x, condition):
        logdet = torch.zeros_like(x[:, 0])
        return self.flow(x, condition, logdet=logdet, reverse=False)

    def reverse_flow(self, z, condition, eps_std, output_shape):
        with torch.no_grad():
            if z is None:
                z = modules.GaussianDiag.sample(output_shape, eps_std)
            x, logdet = self.flow(z, condition, eps_std=eps_std, reverse=True)
        return x, logdet

    def set_actnorm_init(self, inited=True):
        for name, m in self.named_modules():
            if m.__class__.__name__.find("ActNorm") >= 0:
                m.inited = inited

    def init_rnn_hidden(self):
        self.flow.init_rnn_hidden()
