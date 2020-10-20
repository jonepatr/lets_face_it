"""
Test the modules
"""
import torch
import numpy as np
from glow import modules, models


def test_actnorm():
    print("[Test]: actnorm")
    actnorm = modules.ActNorm2d(54)
    x = torch.Tensor(np.random.rand(2, 54))
    actnorm.initialize_parameters(x)
    y, det = actnorm(x, 0)
    x_, _ = actnorm(y, None, True)
    print("actnorm (forward,reverse) delta", float(torch.max(torch.abs(x_ - x))))
    print("  det", float(det))


def test_conv1x1():
    print("[Test]: invconv1x1")
    conv = modules.InvertibleConv1x1(96)
    x = torch.Tensor(np.random.rand(2, 96))
    y, det = conv(x, 0)
    x_, _ = conv(y, None, True)
    print("conv1x1 (forward,reverse) delta", float(torch.max(torch.abs(x_ - x))))
    print("  det", float(det))


def test_flow_step():
    print("[Test]: flow step")
    step = models.FlowStep(
        54,
        256,
        flow_permutation="invconv",
        flow_coupling="affine",
        cond_dim=32,
        feature_encoder_dim=64,
        glow_rnn_type="gru",
    )
    x = torch.Tensor(np.random.rand(2, 54))
    cond = torch.Tensor(np.random.rand(2, 64))

    y, det = step(x, cond, 0, False)
    x_, det0 = step(y, cond, det, True)
    print("flowstep (forward,reverse)delta", float(torch.max(torch.abs(x_ - x))))
    print("  det", det, det0)


def test_flow_net():
    print("[Test]: flow net")
    net = models.FlowNet(
        C=54,
        hidden_channels=256,
        cond_dim=64,
        K=3,
        L=1,
        feature_encoder_dim=32,
        glow_rnn_type="gru",
    )
    x = torch.Tensor(np.random.rand(4, 54))
    cond = torch.Tensor(np.random.rand(4, 32))
    y, det = net(x, cond)
    x_, det0 = net(y, cond, reverse=True)
    print("z", y.size())
    print("x_", x_.size())
    print(det, det0)


if __name__ == "__main__":
    test_actnorm()
    test_conv1x1()
    test_flow_step()
    test_flow_net()
