import os, sys 
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import cv2

try:
    import lpips
except ImportError:
    lpips = None

from scene import GaussianModel, Scene_mica
from src.deform_model import Deform_Model
from gaussian_renderer import render
from arguments import ModelParams, PipelineParams, OptimizationParams
from utils.loss_utils import huber_loss, scale_invariant_depth_loss
from utils.general_utils import normalize_for_percep


def set_random_seed(seed):
    r"""Set random seeds for everything.

    Args:
        seed (int): Random seed.
        by_rank (bool):
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def sidl_depth_loss_for_camera(points_world, viewpoint_cam, min_depth_samples=256):
    if viewpoint_cam.mono_depth is None or viewpoint_cam.depth_valid_mask is None:
        return None

    ones = torch.ones((points_world.shape[0], 1), dtype=points_world.dtype, device=points_world.device)
    points_h = torch.cat([points_world, ones], dim=1)

    view_points = torch.matmul(points_h, viewpoint_cam.world_view_transform)
    clip_points = torch.matmul(points_h, viewpoint_cam.full_proj_transform)

    clip_w = clip_points[:, 3:4]
    valid_w = torch.abs(clip_w[:, 0]) > 1e-8
    safe_w = torch.where(clip_w >= 0.0, torch.clamp(clip_w, min=1e-8), torch.clamp(clip_w, max=-1e-8))
    ndc = clip_points[:, :3] / safe_w
    grid = torch.stack((ndc[:, 0], -ndc[:, 1]), dim=-1)

    in_bounds = (
        (grid[:, 0] >= -1.0)
        & (grid[:, 0] <= 1.0)
        & (grid[:, 1] >= -1.0)
        & (grid[:, 1] <= 1.0)
    )

    grid_sample = grid.view(1, -1, 1, 2)
    mono_depth = viewpoint_cam.mono_depth.unsqueeze(0)
    depth_valid_mask = viewpoint_cam.depth_valid_mask.unsqueeze(0)

    sampled_mono_depth = F.grid_sample(
        mono_depth, grid_sample, mode="bilinear", padding_mode="zeros", align_corners=True
    ).view(-1)
    sampled_valid_mask = (
        F.grid_sample(depth_valid_mask, grid_sample, mode="bilinear", padding_mode="zeros", align_corners=True)
        .view(-1)
        > 0.5
    )

    render_depth = view_points[:, 2]
    valid = (
        valid_w
        & in_bounds
        & sampled_valid_mask
        & torch.isfinite(sampled_mono_depth)
        & torch.isfinite(render_depth)
        & (sampled_mono_depth > 0.0)
        & (render_depth > 0.0)
    )

    if valid.sum().item() < int(min_depth_samples):
        return None

    return scale_invariant_depth_loss(render_depth[valid], sampled_mono_depth[valid])

if __name__ == "__main__":
    # Set up command line argument parser
    parser = argparse.ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--seed', type=int, default=0, help='Random seed.')
    parser.add_argument('--idname', type=str, default='id1_25', help='id name')
    parser.add_argument('--image_res', type=int, default=512, help='image resolution')
    parser.add_argument("--start_checkpoint", type=str, default = None)
    parser.add_argument("--save_every", type=int, default=5000, help="Checkpoint interval.")
    parser.add_argument("--use_depth_supervision", action="store_true", help="Enable SIDL monocular-depth supervision.")
    parser.add_argument("--depth_loss_weight", type=float, default=0.05, help="Weight of the SIDL depth supervision loss.")
    parser.add_argument("--depth_start_iter", type=int, default=0, help="Iteration to start applying SIDL depth loss.")
    parser.add_argument("--depth_erode_kernel", type=int, default=3, help="Erode kernel size for depth valid mask.")
    parser.add_argument("--min_depth_samples", type=int, default=256, help="Minimum valid depth samples per frame for SIDL.")
    args = parser.parse_args(sys.argv[1:])
    args.device = "cuda"
    lpt = lp.extract(args)
    opt = op.extract(args)
    ppt = pp.extract(args)

    batch_size = 1
    set_random_seed(args.seed)

    mid_num = 15000
    percep_module = None
    if lpips is not None and opt.iterations > mid_num:
        percep_module = lpips.LPIPS(net='vgg').to(args.device)

    ## deform model
    DeformModel = Deform_Model(args.device).to(args.device)
    DeformModel.training_setup()

    ## dataloader
    data_dir = os.path.join('dataset', args.idname)
    mica_datadir = os.path.join('metrical-tracker/output', args.idname)
    log_dir = os.path.join(data_dir, 'log')
    train_dir = os.path.join(log_dir, 'train')
    model_dir = os.path.join(log_dir, 'ckpt')
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    scene = Scene_mica(
        data_dir,
        mica_datadir,
        train_type=0,
        white_background=lpt.white_background,
        device=args.device,
        load_depth=args.use_depth_supervision,
        depth_erode_kernel=args.depth_erode_kernel,
    )
    
    first_iter = 0
    gaussians = GaussianModel(lpt.sh_degree)
    gaussians.training_setup(opt)
    if args.start_checkpoint:
        (model_params, gauss_params, first_iter) = torch.load(args.start_checkpoint)
        DeformModel.restore(model_params)
        gaussians.restore(gauss_params, opt)

    bg_color = [1, 1, 1] if lpt.white_background else [0, 1, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device=args.device)
    
    codedict = {}
    codedict['shape'] = scene.shape_param.to(args.device)
    DeformModel.example_init(codedict)

    viewpoint_stack = None
    first_iter += 1
    for iteration in range(first_iter, opt.iterations + 1):
        # Every 500 its we increase the levels of SH up to a maximum degree
        if iteration % 500 == 0:
            gaussians.oneupSHdegree()

        # random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getCameras().copy()
            random.shuffle(viewpoint_stack)
            if len(viewpoint_stack)>2000:
                viewpoint_stack = viewpoint_stack[:2000]
        viewpoint_cam = viewpoint_stack.pop(random.randint(0, len(viewpoint_stack)-1)) 
        frame_id = viewpoint_cam.uid

        # deform gaussians
        codedict['expr'] = viewpoint_cam.exp_param
        codedict['eyes_pose'] = viewpoint_cam.eyes_pose
        codedict['eyelids'] = viewpoint_cam.eyelids
        codedict['jaw_pose'] = viewpoint_cam.jaw_pose 
        verts_final, rot_delta, scale_coef = DeformModel.decode(codedict)
        
        if iteration == 1:
            gaussians.create_from_verts(verts_final[0])
            gaussians.training_setup(opt)
        gaussians.update_xyz_rot_scale(verts_final[0], rot_delta[0], scale_coef[0])

        # Render
        render_pkg = render(viewpoint_cam, gaussians, ppt, background)
        image = render_pkg["render"]

        # Loss
        gt_image = viewpoint_cam.original_image
        mouth_mask = viewpoint_cam.mouth_mask
        
        loss_huber = huber_loss(image, gt_image, 0.1) + 40*huber_loss(image*mouth_mask, gt_image*mouth_mask, 0.1)
        
        loss_G = 0.
        head_mask = viewpoint_cam.head_mask
        image_percep = normalize_for_percep(image*head_mask)
        gt_image_percep = normalize_for_percep(gt_image*head_mask)
        if iteration>mid_num and percep_module is not None:
            loss_G = torch.mean(percep_module.forward(image_percep, gt_image_percep))*0.05

        loss_depth = torch.zeros((), dtype=torch.float32, device=args.device)
        if args.use_depth_supervision and iteration >= args.depth_start_iter:
            sidl_loss = sidl_depth_loss_for_camera(
                gaussians.get_xyz,
                viewpoint_cam,
                min_depth_samples=args.min_depth_samples,
            )
            if sidl_loss is not None:
                loss_depth = sidl_loss

        loss = loss_huber*1 + loss_G*1 + args.depth_loss_weight*loss_depth

        loss.backward()

        with torch.no_grad():
            # Optimizer step
            gaussians.optimizer.step()
            DeformModel.optimizer.step()
            gaussians.optimizer.zero_grad(set_to_none = True)
            DeformModel.optimizer.zero_grad(set_to_none = True)
            
            # print loss
            if iteration % 500 == 0:
                if iteration<=mid_num:
                    if args.use_depth_supervision:
                        print("step: %d, huber: %.5f, depth: %.5f" %(iteration, loss_huber.item(), loss_depth.item()))
                    else:
                        print("step: %d, huber: %.5f" %(iteration, loss_huber.item()))
                else:
                    if args.use_depth_supervision:
                        print("step: %d, huber: %.5f, percep: %.5f, depth: %.5f" %(iteration, loss_huber.item(), loss_G.item(), loss_depth.item()))
                    else:
                        print("step: %d, huber: %.5f, percep: %.5f" %(iteration, loss_huber.item(), loss_G.item()))
            
            # visualize results
            if iteration % 500 == 0 or iteration==1:
                gt_image_np = (gt_image*255.).permute(1,2,0).detach().cpu().numpy().astype(np.uint8)
                image = image.clamp(0, 1)
                image_np = (image*255.).permute(1,2,0).detach().cpu().numpy().astype(np.uint8)
                image_h, image_w = gt_image_np.shape[:2]
                save_image = np.zeros((image_h, image_w*2, 3), dtype=np.uint8)
                save_image[:, :image_w, :] = gt_image_np
                save_image[:, image_w:, :] = image_np
                cv2.imwrite(os.path.join(train_dir, f"{iteration}.jpg"), save_image[:,:,[2,1,0]])
            
            # save checkpoint
            if iteration % args.save_every == 0:
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((DeformModel.capture(), gaussians.capture(), iteration), model_dir + "/chkpnt" + str(iteration) + ".pth")

    final_ckpt_path = os.path.join(model_dir, f"chkpnt{opt.iterations}.pth")
    if not os.path.exists(final_ckpt_path):
        torch.save((DeformModel.capture(), gaussians.capture(), opt.iterations), final_ckpt_path)

           
