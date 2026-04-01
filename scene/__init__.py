import os, sys
import random
import json
from PIL import Image
import torch
import math
import numpy as np
import cv2
from tqdm import tqdm

from scene.gaussian_model import GaussianModel
from scene.cameras import Camera
from arguments import ModelParams
from utils.general_utils import PILtoTensor
from utils.graphics_utils import focal2fov


class Scene_mica:
    def __init__(self, datadir, mica_datadir, train_type, white_background, device, load_depth=False, depth_erode_kernel=3):
        ## train_type: 0 for train, 1 for test, 2 for eval
        frame_delta = 1 # default mica-tracking starts from the second frame
        images_folder = os.path.join(datadir, "imgs")
        parsing_folder = os.path.join(datadir, "parsing")
        alpha_folder = os.path.join(datadir, "alpha")
        depth_folder = os.path.join(datadir, "depth")
        
        self.bg_image = torch.zeros((3, 512, 512))
        if white_background:
            self.bg_image[:, :, :] = 1
        else:
            self.bg_image[1, :, :] = 1

        mica_ckpt_dir = os.path.join(mica_datadir, 'checkpoint')
        self.N_frames = len(os.listdir(mica_ckpt_dir))
        self.cameras = []
        if self.N_frames <= 500:
            # Keep short custom clips trainable by reserving only a small holdout split.
            test_num = min(50, max(1, self.N_frames // 5))
            eval_num = min(50, test_num)
        else:
            test_num = 500
            eval_num = 50
        max_train_num = 10000
        train_num = min(max_train_num, max(1, self.N_frames - test_num))
        ckpt_path = os.path.join(mica_ckpt_dir, '00000.frame')
        payload = torch.load(ckpt_path)
        flame_params = payload['flame']
        self.shape_param = torch.as_tensor(flame_params['shape'])
        orig_w, orig_h = payload['img_size']
        K = payload['opencv']['K'][0]
        fl_x = K[0, 0]
        fl_y = K[1, 1]
        FovY = focal2fov(fl_y, orig_h)
        FovX = focal2fov(fl_x, orig_w)
        if train_type == 0:
            range_down = 0
            range_up = min(self.N_frames, train_num)
        if train_type == 1:
            range_down = max(0, self.N_frames - test_num)
            range_up = self.N_frames
        if train_type == 2:
            range_down = max(0, self.N_frames - eval_num)
            range_up = self.N_frames

        for frame_id in tqdm(range(range_down, range_up)):
            image_name_mica = str(frame_id).zfill(5) # obey mica tracking
            image_name_ori = str(frame_id+frame_delta).zfill(5)
            ckpt_path = os.path.join(mica_ckpt_dir, image_name_mica+'.frame')
            payload = torch.load(ckpt_path)
            
            flame_params = payload['flame']
            exp_param = torch.as_tensor(flame_params['exp'])
            eyes_pose = torch.as_tensor(flame_params['eyes'])
            eyelids = torch.as_tensor(flame_params['eyelids'])
            jaw_pose = torch.as_tensor(flame_params['jaw'])

            oepncv = payload['opencv']
            w2cR = oepncv['R'][0]
            w2cT = oepncv['t'][0]
            R = np.transpose(w2cR) # R is stored transposed due to 'glm' in CUDA code
            T = w2cT

            image_path = os.path.join(images_folder, image_name_ori+'.jpg')
            image = Image.open(image_path)
            resized_image_rgb = PILtoTensor(image)
            gt_image = resized_image_rgb[:3, ...]
            
            # alpha
            alpha_path = os.path.join(alpha_folder, image_name_ori+'.jpg')
            alpha = Image.open(alpha_path)
            alpha = PILtoTensor(alpha)

            # # if add head mask
            head_mask_path = os.path.join(parsing_folder, image_name_ori+'_neckhead.png')
            head_mask = Image.open(head_mask_path)
            head_mask = PILtoTensor(head_mask)
            gt_image = gt_image * alpha + self.bg_image * (1 - alpha)
            gt_image = gt_image * head_mask + self.bg_image * (1 - head_mask)

            # mouth mask
            mouth_mask_path = os.path.join(parsing_folder, image_name_ori+'_mouth.png')
            mouth_mask = Image.open(mouth_mask_path)
            mouth_mask = PILtoTensor(mouth_mask)

            mono_depth = None
            depth_valid_mask = None
            if load_depth:
                depth_path = os.path.join(depth_folder, image_name_ori+'.npy')
                if not os.path.exists(depth_path):
                    raise FileNotFoundError(f"Missing depth map for frame {image_name_ori}: {depth_path}")

                depth_np = np.load(depth_path).astype(np.float32)
                if depth_np.ndim == 3:
                    depth_np = np.squeeze(depth_np)
                if depth_np.ndim != 2:
                    raise ValueError(f"Depth map must be 2D, got shape {depth_np.shape} at {depth_path}")

                target_h, target_w = gt_image.shape[1], gt_image.shape[2]
                if depth_np.shape != (target_h, target_w):
                    depth_np = cv2.resize(depth_np, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
                mono_depth = torch.from_numpy(depth_np).unsqueeze(0)

                head_np = (head_mask.squeeze(0).numpy() > 0.5).astype(np.uint8) * 255
                mouth_np = (mouth_mask.squeeze(0).numpy() > 0.5).astype(np.uint8) * 255
                kernel_size = max(1, int(depth_erode_kernel))
                if kernel_size > 1:
                    kernel = np.ones((kernel_size, kernel_size), np.uint8)
                    head_valid = cv2.erode(head_np, kernel, iterations=1)
                else:
                    head_valid = head_np
                valid_np = ((head_valid > 127) & (mouth_np <= 127)).astype(np.float32)
                depth_valid_mask = torch.from_numpy(valid_np).unsqueeze(0)

            camera_indiv = Camera(colmap_id=frame_id, R=R, T=T, 
                                FoVx=FovX, FoVy=FovY, 
                                image=gt_image, head_mask=head_mask, mouth_mask=mouth_mask,
                                exp_param=exp_param, eyes_pose=eyes_pose, eyelids=eyelids, jaw_pose=jaw_pose,
                                mono_depth=mono_depth, depth_valid_mask=depth_valid_mask,
                                image_name=image_name_mica, uid=frame_id, data_device=device)
            self.cameras.append(camera_indiv)
    
    def getCameras(self):
        return self.cameras





    
