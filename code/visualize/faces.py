import os
import shutil
import tempfile
from contextlib import contextmanager
from pathlib import Path

import cv2
import numpy as np
import pyrender
import torch
import trimesh
from tqdm import tqdm

from FLAME_PyTorch.FLAME import FLAME
from misc.shared import BASE_DIR, CONFIG, flame_config


os.environ["PYOPENGL_PLATFORM"] = "egl"


class RenderContext(object):
    def __init__(self, z=0, x=0, y=0, width=1024, light_color=None, f=None):
        if light_color is None:
            light_color = np.array([1.0, 1.0, 1.0])

        if f is None:
            f = np.array([4754.97941935 / 2, 4754.97941935 / 2])

        self.mesh = None
        frustum = {"near": 0.01, "far": 100.0, "height": 1024, "width": width}
        camera_params = {
            "c": np.array([x, y]),
            "k": np.array([-0.19816071, 0.92822711, 0, 0, 0]),
            "f": f,
        }
        intensity = 1.5
        self.rgb_per_v = None

        self.scene = pyrender.Scene(
            ambient_light=[0.2, 0.2, 0.2], bg_color=[255, 255, 255]
        )
        camera = pyrender.IntrinsicsCamera(
            fx=camera_params["f"][0],
            fy=camera_params["f"][1],
            cx=camera_params["c"][0],
            cy=camera_params["c"][1],
            znear=frustum["near"],
            zfar=frustum["far"],
        )

        camera_pose = np.eye(4)
        camera_pose[:3, 3] = np.array([0, 0, 1.0 - z])
        self.scene.add(camera, pose=camera_pose)

        angle = np.pi / 6.0
        pos = [0, 0, 1]

        light = pyrender.PointLight(color=light_color, intensity=intensity)

        light_pose = np.eye(4)
        light_pose[:3, 3] = pos
        self.scene.add(light, pose=light_pose.copy())

        light_pose[:3, 3] = cv2.Rodrigues(np.array([angle, 0, 0]))[0].dot(pos)
        self.scene.add(light, pose=light_pose.copy())

        light_pose[:3, 3] = cv2.Rodrigues(np.array([-angle, 0, 0]))[0].dot(pos)
        self.scene.add(light, pose=light_pose.copy())

        light_pose[:3, 3] = cv2.Rodrigues(np.array([0, -angle, 0]))[0].dot(pos)
        self.scene.add(light, pose=light_pose.copy())

        light_pose[:3, 3] = cv2.Rodrigues(np.array([0, angle, 0]))[0].dot(pos)
        self.scene.add(light, pose=light_pose.copy())

        self.r = pyrender.OffscreenRenderer(
            viewport_width=frustum["width"], viewport_height=frustum["height"]
        )

    def render_meshes(self, meshes):
        scene_meshes = []
        for mesh in meshes:
            render_mesh = pyrender.Mesh.from_trimesh(mesh)
            scene_meshes.append(self.scene.add(render_mesh))  # , pose=np.eye(4)

        # flags = pyrender.RenderFlags.SKIP_CULL_FACES
        flags = None
        img, _ = self.r.render(self.scene, flags=flags)
        for scene_mesh in scene_meshes:
            self.scene.remove_node(scene_mesh)
        return img[..., ::-1]


@contextmanager
def create_temp_obj(texture_dir, skin_color):
    with tempfile.TemporaryDirectory() as tmpd:
        obj_file = Path(tmpd) / "file.obj"

        shutil.copy(
            texture_dir / "base_model.mtl", Path(tmpd) / "file.mtl",
        )
        shutil.copy(
            texture_dir / f"texture_{skin_color}.png", Path(tmpd) / "texture.png",
        )
        extra_info = (texture_dir / "base_model.partial_obj").read_text()

        def get_obj(vertices):
            with open(obj_file, "w") as tmpf:

                tmpf.write("mtllib file.mtl\n")
                for vertex in vertices:
                    tmpf.write("v " + " ".join(map(str, vertex.tolist())) + "\n")

                tmpf.write(extra_info)

            return obj_file

        yield get_obj


def render_double_face_video(
    file_name,
    vertices,
    vertices2,
    fps=50,
    skin_color_v1="white",
    skin_color_v2="white",
    width=2048,
):
    rc = RenderContext(
        x=width // 2,
        y=400,
        width=width,
        z=-1,
        f=np.array([4754.97941935, 4754.97941935]),
    )

    writer = cv2.VideoWriter(
        file_name, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, 1024)
    )

    cto1 = create_temp_obj(BASE_DIR / "code/visualize/texture", skin_color_v1)
    cto2 = create_temp_obj(BASE_DIR / "code/visualize/texture", skin_color_v2)

    with cto1 as obj_file1, cto2 as obj_file2:

        for vertice, vertice2 in tqdm(list(zip(vertices, vertices2)), leave=False):
            face_width = 0.1

            vertice[:, 0] -= face_width * 2
            vertice2[:, 0] += face_width * 2

            tri_mesh1 = trimesh.load_mesh(
                file_obj=open(obj_file1(vertice)), file_type="obj"
            )
            tri_mesh2 = trimesh.load_mesh(
                file_obj=open(obj_file2(vertice2)), file_type="obj"
            )

            color = rc.render_meshes([tri_mesh1, tri_mesh2])
            writer.write(color)

        writer.release()


def render_face_video(
    file_name,
    shape_params=None,
    pose_params=None,
    expression_params=None,
    neck_params=None,
    eye_params=None,
    fps=50,
    gender="generic",
    skin_color="white",
):
    vertices, _, faces = render_faces(
        shape_params,
        pose_params,
        expression_params,
        neck_params,
        eye_params,
        gender=gender,
    )

    writer = cv2.VideoWriter(
        file_name, cv2.VideoWriter_fourcc(*"mp4v"), fps, (1024, 1024)
    )
    rc = RenderContext(x=400, y=400, z=-1)
    with create_temp_obj(BASE_DIR / "code/visualize/texture", skin_color) as obj_file:

        for vertice in tqdm(vertices):
            tri_mesh = trimesh.load_mesh(
                file_obj=open(obj_file(vertice)), file_type="obj"
            )

            color = rc.render_meshes([tri_mesh])
            writer.write(color)

        writer.release()


def render_face_image(
    file_name,
    shape_params=None,
    pose_params=None,
    expression_params=None,
    neck_params=None,
    eye_params=None,
    gender="generic",
    skin_color="white",
):

    assert shape_params.shape[0] == 1
    vertices, *_ = render_faces(
        shape_params,
        pose_params,
        expression_params,
        neck_params,
        eye_params,
        gender=gender,
    )

    rc = RenderContext(x=400, y=400, z=-1)

    with create_temp_obj(BASE_DIR / "code/visualize/texture", skin_color) as obj_file:
        tri_mesh = trimesh.load_mesh(
            file_obj=open(obj_file(vertices[0])), file_type="obj"
        )

        color = rc.render_meshes([tri_mesh])
        cv2.imwrite(file_name, color)


def random_shape(self, seq_len):
    shape = torch.zeros((1, 300))
    shape[:, :100] = torch.rand(100)
    return shape.repeat(seq_len, 1)


def get_vert2(self, sequence, exp_dim=100, jaw_dim=3, neck_dim=3, gender="generic"):

    seq_len = sequence.shape[0]

    shape_params = random_shape(seq_len).type_as(sequence)
    expression_params = sequence[:, :exp_dim]

    zero_padding = torch.zeros((seq_len, 6 - jaw_dim)).type_as(sequence)
    jaw_params = sequence[:, exp_dim : exp_dim + jaw_dim]
    pose_params = torch.cat([zero_padding, jaw_params], dim=1)

    zero_padding = torch.zeros((seq_len, 3 - neck_dim)).type_as(sequence)
    neck_pose = torch.cat(
        [zero_padding, sequence[:, exp_dim + jaw_dim : exp_dim + jaw_dim + neck_dim]],
        dim=1,
    )

    flame = FLAME(
        flame_config(
            BASE_DIR / CONFIG["flame"][f"model_path_{gender}"],
            BASE_DIR / CONFIG["flame"]["static_landmark_embedding_path"],
            BASE_DIR / CONFIG["flame"]["dynamic_landmark_embedding_path"],
            seq_len,
            True,
            300,
            exp_dim,
            True,
        )
    )

    vertices, _ = flame(
        shape_params=shape_params,
        expression_params=expression_params,
        pose_params=pose_params,
        neck_pose=neck_pose,
    )

    return vertices.cpu().numpy()


def get_vert(sequence, gender="generic"):
    seq_len = sequence.size(0)

    flame = FLAME(
        flame_config(
            BASE_DIR / CONFIG["flame"][f"model_path_{gender}"],
            BASE_DIR / CONFIG["flame"]["static_landmark_embedding_path"],
            BASE_DIR / CONFIG["flame"]["dynamic_landmark_embedding_path"],
            seq_len,
            True,
            300,
            100,
            True,
        )
    )

    shape = torch.zeros((1, 300)).type_as(sequence)
    shape[:, :100] = torch.randn(100)

    shape_params = shape.repeat(seq_len, 1)
    pose_params = torch.cat([torch.zeros((seq_len, 3)), sequence[:, 100:103]], dim=1)

    vertices, _ = flame(
        shape_params=shape_params,
        expression_params=sequence[:, :100],
        pose_params=pose_params,
        neck_pose=sequence[:, 103:106],
    )

    return vertices.cpu().numpy()


def render_face(gender, clip, lipsync, right_side, shape):
    seq_len = clip["expression"].shape[0]
    eye_params = lipsync["pose"][:, 6:]
    eye_params[:, 0] -= 0.05
    eye_params[:, 1] += 0.1
    eye_params[:, 5] -= 0.05

    neck_params = clip["neck"] + lipsync["pose"][:, 3:6]

    jaw_params = clip["jaw"] + lipsync["pose"][:, :3]
    pose_params = torch.cat([torch.zeros(seq_len, 3), jaw_params], dim=1)

    pose_params[:, 1] += -0.11 if right_side else 0.11

    eye_params[:, :3] -= neck_params
    eye_params[:, 3:] -= neck_params

    expression_params = (
        clip["expression"] + lipsync["expression"][:, : clip["expression"].size(1)]
    )

    flame = FLAME(
        flame_config(
            BASE_DIR / CONFIG["flame"][f"model_path_{gender}"],
            BASE_DIR / CONFIG["flame"]["static_landmark_embedding_path"],
            BASE_DIR / CONFIG["flame"]["dynamic_landmark_embedding_path"],
            seq_len,
            True,
            shape.size(1),
            clip["expression"].size(1),
            True,
        )
    )

    vertices, _ = flame(
        shape_params=shape,
        expression_params=expression_params,
        pose_params=pose_params,
        neck_pose=neck_params,
        eye_pose=eye_params,
    )

    return vertices.cpu().numpy()


def render_faces(
    shape_params=None,
    pose_params=None,
    expression_params=None,
    neck_params=None,
    eye_params=None,
    gender="generic",
    cuda=True,
):

    shape = torch.from_numpy(shape_params).float()
    pose = torch.from_numpy(pose_params).float()
    expression = torch.from_numpy(expression_params).float()
    neck = torch.from_numpy(neck_params).float()
    eye = torch.from_numpy(eye_params).float()

    config = flame_config(
        BASE_DIR / CONFIG["flame"][f"model_path_{gender}"],
        BASE_DIR / CONFIG["flame"]["static_landmark_embedding_path"],
        BASE_DIR / CONFIG["flame"]["dynamic_landmark_embedding_path"],
        shape_params.shape[0],
        True,
        shape.shape[1],
        expression.shape[1],
        True,
    )

    flamelayer = FLAME(config)

    if cuda:
        flamelayer = flamelayer.cuda()
        shape = shape.cuda()
        pose = pose.cuda()
        expression = expression.cuda()
        neck = neck.cuda()
        eye = eye.cuda()

    vertices, landmarks = flamelayer(
        shape_params=shape,
        expression_params=expression,
        pose_params=pose,
        neck_pose=neck,
        eye_pose=eye,
    )

    return (
        vertices.detach().cpu().numpy(),
        landmarks.detach().cpu().numpy(),
        flamelayer.faces,
    )
