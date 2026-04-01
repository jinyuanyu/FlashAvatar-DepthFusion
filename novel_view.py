import os
import sys
import math
import random
import argparse
import datetime

import cv2
import numpy as np
import torch
from tqdm import tqdm

from scene import GaussianModel, Scene_mica
from scene.cameras import MiniCam
from src.deform_model import Deform_Model
from gaussian_renderer import render
from arguments import ModelParams, PipelineParams, OptimizationParams
from utils.graphics_utils import getWorld2View2


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _rot_x(rad):
    c, s = math.cos(rad), math.sin(rad)
    return np.array(
        [[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]],
        dtype=np.float32,
    )


def _rot_y(rad):
    c, s = math.cos(rad), math.sin(rad)
    return np.array(
        [[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]],
        dtype=np.float32,
    )


def _rot_z(rad):
    c, s = math.cos(rad), math.sin(rad)
    return np.array(
        [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )


def build_novel_cam(src_cam, frame_idx, args):
    # Convert back to conventional world->camera matrix (3x3 R, 3x1 t).
    w2c = src_cam.world_view_transform.transpose(0, 1).detach().cpu().numpy().astype(np.float32)
    r_wc = w2c[:3, :3]
    t_wc = w2c[:3, 3]

    # Camera center in world coordinates.
    c_w = -r_wc.T @ t_wc
    target = np.array([args.target_x, args.target_y, args.target_z], dtype=np.float32)

    orbit_deg = 0.0
    if args.orbit_range_deg > 0:
        orbit_deg = args.orbit_range_deg * math.sin(2.0 * math.pi * frame_idx / max(1, args.orbit_period))

    yaw = math.radians(args.yaw_deg + orbit_deg)
    pitch = math.radians(args.pitch_deg)
    roll = math.radians(args.roll_deg)
    r_delta = _rot_z(roll) @ _rot_y(yaw) @ _rot_x(pitch)

    # Rotate both camera center and orientation in world frame.
    c_w_new = r_delta @ (c_w - target) + target
    r_cw = r_wc.T
    r_cw_new = r_delta @ r_cw
    r_wc_new = r_cw_new.T
    t_wc_new = -r_wc_new @ c_w_new

    # Camera class expects "R" stored transposed due to historical CUDA convention.
    r_param = r_wc_new.T
    world_view_transform = torch.tensor(
        getWorld2View2(r_param, t_wc_new),
        dtype=torch.float32,
        device="cuda",
    ).transpose(0, 1)
    full_proj_transform = (
        world_view_transform.unsqueeze(0).bmm(src_cam.projection_matrix.unsqueeze(0)).squeeze(0)
    )

    return MiniCam(
        src_cam.image_width,
        src_cam.image_height,
        src_cam.FoVy,
        src_cam.FoVx,
        src_cam.znear,
        src_cam.zfar,
        world_view_transform,
        full_proj_transform,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Novel-view rendering from fitted FlashAvatar model")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)

    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument("--idname", type=str, default="id1_25", help="Identity folder name.")
    parser.add_argument("--logname", type=str, default="log", help="Log folder under dataset/<idname>.")
    parser.add_argument("--image_res", type=int, default=512, help="Output image resolution.")
    parser.add_argument("--checkpoint", type=str, default=None, help="Model checkpoint path.")

    parser.add_argument("--yaw_deg", type=float, default=20.0, help="Base yaw offset in degrees.")
    parser.add_argument("--pitch_deg", type=float, default=0.0, help="Pitch offset in degrees.")
    parser.add_argument("--roll_deg", type=float, default=0.0, help="Roll offset in degrees.")
    parser.add_argument("--orbit_range_deg", type=float, default=0.0, help="Extra sinusoidal yaw range.")
    parser.add_argument("--orbit_period", type=int, default=120, help="Orbit period in frames.")

    parser.add_argument("--target_x", type=float, default=0.0, help="Orbit target X in world coordinates.")
    parser.add_argument("--target_y", type=float, default=0.0, help="Orbit target Y in world coordinates.")
    parser.add_argument("--target_z", type=float, default=0.0, help="Orbit target Z in world coordinates.")

    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output video path. Default: dataset/<id>/log/test_novel_*.avi",
    )
    parser.add_argument(
        "--with_gt",
        action="store_true",
        help="Write side-by-side (GT | novel render).",
    )

    args = parser.parse_args(sys.argv[1:])
    args.device = "cuda"
    lpt = lp.extract(args)
    opt = op.extract(args)
    ppt = pp.extract(args)

    set_random_seed(args.seed)

    deform_model = Deform_Model(args.device).to(args.device)
    deform_model.training_setup()
    deform_model.eval()

    data_dir = os.path.join("dataset", args.idname)
    mica_datadir = os.path.join("metrical-tracker/output", args.idname)
    logdir = os.path.join(data_dir, args.logname)
    scene = Scene_mica(data_dir, mica_datadir, train_type=1, white_background=lpt.white_background, device=args.device)

    gaussians = GaussianModel(lpt.sh_degree)
    gaussians.training_setup(opt)
    if args.checkpoint:
        model_params, gauss_params, _ = torch.load(args.checkpoint)
        deform_model.restore(model_params)
        gaussians.restore(gauss_params, opt)
    else:
        raise ValueError("--checkpoint is required for novel-view rendering.")

    bg_color = [1, 1, 1] if lpt.white_background else [0, 1, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device=args.device)

    if args.output is None:
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output = os.path.join(logdir, f"test_novel_{ts}.avi")
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    width = args.image_res * 2 if args.with_gt else args.image_res
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    out = cv2.VideoWriter(args.output, fourcc, 25, (width, args.image_res), True)

    viewpoint = scene.getCameras().copy()
    codedict = {"shape": scene.shape_param.to(args.device)}
    deform_model.example_init(codedict)

    for i, src_cam in enumerate(tqdm(viewpoint)):
        codedict["expr"] = src_cam.exp_param
        codedict["eyes_pose"] = src_cam.eyes_pose
        codedict["eyelids"] = src_cam.eyelids
        codedict["jaw_pose"] = src_cam.jaw_pose

        verts_final, rot_delta, scale_coef = deform_model.decode(codedict)
        gaussians.update_xyz_rot_scale(verts_final[0], rot_delta[0], scale_coef[0])

        novel_cam = build_novel_cam(src_cam, i, args)
        render_pkg = render(novel_cam, gaussians, ppt, background)
        novel = render_pkg["render"].clamp(0, 1)
        novel_np = (novel * 255.0).permute(1, 2, 0).detach().cpu().numpy().astype(np.uint8)

        if args.with_gt:
            gt_np = (src_cam.original_image * 255.0).permute(1, 2, 0).detach().cpu().numpy().astype(np.uint8)
            canvas = np.zeros((args.image_res, args.image_res * 2, 3), dtype=np.uint8)
            canvas[:, :args.image_res, :] = gt_np
            canvas[:, args.image_res:, :] = novel_np
        else:
            canvas = novel_np

        # RGB -> BGR for OpenCV writer.
        out.write(canvas[:, :, [2, 1, 0]])

    out.release()
    print(f"[NovelView] Saved: {args.output}")
