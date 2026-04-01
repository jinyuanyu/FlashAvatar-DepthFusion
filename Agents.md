# Agents.md

## Purpose

This repository implements **FlashAvatar** from the paper `FlashAvatar: High-fidelity Head Avatar with Efficient Gaussian Embedding`.

Its goal is:

- reconstruct a personalized animatable head avatar from a **monocular video**
- represent the avatar as a **fixed-size Gaussian field** attached to a FLAME face mesh
- drive Gaussian motion with **tracked FLAME expression / eye / jaw / eyelid parameters**
- render fast with **3D Gaussian Splatting**

The codebase is **not** a full raw-video pipeline. It assumes the video has already been preprocessed into:

- extracted frames
- alpha masks
- parsing masks
- per-frame FLAME + camera tracking results under `metrical-tracker/output/<id>/checkpoint/*.frame`

## Functional Scope

What this repo does:

- train the Gaussian appearance + offset MLP from tracked monocular data
- render same-view test videos
- restore checkpoints and render novel views if camera transforms are changed

What this repo does not do:

- run face tracking from raw video
- generate alpha / semantic parsing from raw video
- perform audio-driven animation
- train a generalizable multi-identity model

## Main Entry Points

- [`train.py`](/root/AvatarStack/modules/FlashAvatar-code/train.py): optimize one identity
- [`test.py`](/root/AvatarStack/modules/FlashAvatar-code/test.py): render same-view test split
- [`novel_view.py`](/root/AvatarStack/modules/FlashAvatar-code/novel_view.py): workspace-added utility for novel-view rendering
- [`scene/__init__.py`](/root/AvatarStack/modules/FlashAvatar-code/scene/__init__.py): load preprocessed samples into `Camera` objects
- [`src/deform_model.py`](/root/AvatarStack/modules/FlashAvatar-code/src/deform_model.py): FLAME-driven Gaussian offset model
- [`scene/gaussian_model.py`](/root/AvatarStack/modules/FlashAvatar-code/scene/gaussian_model.py): Gaussian state and learnable attributes
- [`gaussian_renderer/__init__.py`](/root/AvatarStack/modules/FlashAvatar-code/gaussian_renderer/__init__.py): splatting renderer
- [`flame/flame_mica.py`](/root/AvatarStack/modules/FlashAvatar-code/flame/flame_mica.py): FLAME geometry backend

## Input / Output Contract

### Filesystem Input

Per identity `dataset/<idname>`:

- `imgs/*.jpg`: RGB frames, expected to be aligned to tracker output with a `+1` offset
- `alpha/*.jpg`: foreground alpha mask, same spatial size as images
- `parsing/*_neckhead.png`: binary head-region mask
- `parsing/*_mouth.png`: binary mouth-region mask

Per identity `metrical-tracker/output/<idname>/checkpoint/*.frame`:

- one PyTorch payload per frame
- required keys:
  - `flame.exp`: `(1, 100)`
  - `flame.shape`: `(1, 300)`
  - `flame.eyes`: `(1, 12)`
  - `flame.eyelids`: `(1, 2)`
  - `flame.jaw`: `(1, 6)`
  - `opencv.R`: `(1, 3, 3)`
  - `opencv.t`: `(1, 3)`
  - `opencv.K`: `(1, 3, 3)`
  - `img_size`: `(2,)`, typically `[512, 512]`

Checkpoint input:

- `dataset/<idname>/log/ckpt/chkpnt*.pth`
- saved tuple:
  - deform model state
  - Gaussian model state
  - training iteration

### Filesystem Output

Training:

- `dataset/<idname>/log/train/*.jpg`: periodic side-by-side previews
- `dataset/<idname>/log/ckpt/chkpnt*.pth`: model checkpoints

Testing:

- `dataset/<idname>/log/test.avi`: side-by-side GT + render

Novel view:

- custom `.avi` or `.mp4` written to the requested output path

## Core Representation

FlashAvatar here is implemented as:

- a **fixed-count Gaussian set** with learnable:
  - color SH coefficients
  - opacity
  - base scaling
  - base rotation
- a **FLAME-conditioned offset MLP** that predicts per-Gaussian:
  - position residual `Delta mu`
  - rotation residual `Delta r`
  - scale coefficient `Delta s`

Important design choice:

- unlike vanilla 3D-GS, this repo does **not** rely on runtime densification for the intended pipeline
- Gaussian count is determined by UV sampling once, then stays fixed

## End-to-End Data Flow

### 1. Offline Preconditions

The paper assumes tracked inputs:

- monocular video frames `I`
- camera intrinsics / poses `K, P`
- tracked FLAME meshes / expression codes

In this codebase, those are already encoded into:

- `dataset/<id>/imgs`, `alpha`, `parsing`
- `metrical-tracker/output/<id>/checkpoint/*.frame`

There is no raw-video preprocessing stage in this repo.

### 2. Sample Loading (`Scene_mica`)

[`scene/__init__.py`](/root/AvatarStack/modules/FlashAvatar-code/scene/__init__.py) converts filesystem samples into a list of `Camera` objects.

Per frame:

- read image: `gt_image` shape `(3, H, W)`, usually `(3, 512, 512)`
- read alpha: shape `(1, H, W)`
- read head mask: shape `(1, H, W)`
- read mouth mask: shape `(1, H, W)`
- composite background:
  - first by alpha
  - then by head mask
- read FLAME parameters from `.frame`
- read OpenCV camera `R, t, K`
- convert focal length to `FoVx/FoVy`
- build `Camera`

State transition:

- filesystem sample -> `Camera`
- `Camera.original_image`: `(3, 512, 512)`
- `Camera.head_mask`: `(1, 512, 512)`
- `Camera.mouth_mask`: `(1, 512, 512)`
- `Camera.exp_param`: `(1, 100)`
- `Camera.eyes_pose`: `(1, 12)`
- `Camera.eyelids`: `(1, 2)`
- `Camera.jaw_pose`: `(1, 6)`

Identity-level state:

- `Scene_mica.shape_param`: `(1, 300)` from the first checkpoint frame

### 3. Canonical UV Gaussian Initialization (`Deform_Model.example_init`)

[`src/deform_model.py`](/root/AvatarStack/modules/FlashAvatar-code/src/deform_model.py) performs the paper's "surface-embedded Gaussian initialization".

Static geometry:

- FLAME template vertices from `generic_model.pkl`: `(5023, 3)`
- FLAME mesh topology for UV embedding from `FlameMesh.obj`
- UV rasterizer resolution: `128 x 128`

Initialization steps:

1. Generate neutral-shape FLAME geometry:
   - `geometry_shape`: `(B, 5023, 3)`
2. Convert vertices to per-face representation:
   - `face_vertices_shape`: `(B, F, 3, 3)`
3. Rasterize mesh into UV space once:
   - `rast_out`: `(B, 4, 128, 128)`
   - first 3 channels are 3D positions in UV raster
   - last channel is visibility mask
4. Flatten visible UV pixels:
   - `uv_vertices_shape`: `(1, V_uv, 3)`
5. Positional encode canonical UV-sampled 3D positions:
   - embed dim = `3 * (1 + 2 * 8) = 51`
   - `uv_vertices_shape_embeded`: `(1, V_uv, 51)`
6. Apply head-region mask from `FLAME_masks.pkl`

Observed runtime sizes in this workspace:

- `V_uv` before head filtering: `14876`
- head-region Gaussian count after filtering: `13453`

This matches the paper's fixed UV-resolution strategy.

### 4. Expression-Conditioned Offset Decoding (`Deform_Model.decode`)

Condition input is the paper's expression code `psi`, implemented as:

- expression: `(1, 100)`
- jaw: `(1, 6)`
- eyes: `(1, 12)`
- eyelids: `(1, 2)`
- concatenated condition: `(1, 120)`

MLP input:

- canonical positional encoding: `51`
- condition: `120`
- total per-Gaussian input dim: `171`

MLP output per Gaussian: `10`

- first 3: position residual
- next 4: rotation delta
- last 3: scale coefficient

Decoded tensors:

- `verts_final`: `(1, 13453, 3)`
- `rot_delta`: `(1, 13453, 4)`
- `scale_coef`: `(1, 13453, 3)`

State transition:

- tracked FLAME parameters -> deformed mesh vertices
- canonical UV sample positions -> expression-specific Gaussian centers

### 5. Gaussian State Construction (`GaussianModel`)

On first training iteration:

- `create_from_verts(verts_final[0])`

Initial learnable tensors:

- `xyz`: `(13453, 3)`
- `features_dc`: `(13453, 1, 3)`
- `features_rest`: `(13453, 15, 3)` for SH degree 3
- `scaling_base`: `(13453, 3)`
- `rotation_base`: `(13453, 4)`
- `opacity`: `(13453, 1)`

Per frame / per iteration update:

- `xyz = verts_final`
- `rotation = quatProduct(rotation_base, rot_delta)`
- `scaling = scaling_base * scale_coef`

Only these are optimized in the intended pipeline:

- Gaussian appearance / opacity / base rotation / base scaling
- offset MLP weights

Gaussian positions are not free parameters after init; they are driven by FLAME + offset.

### 6. Rendering (`render`)

[`gaussian_renderer/__init__.py`](/root/AvatarStack/modules/FlashAvatar-code/gaussian_renderer/__init__.py) performs Gaussian splatting.

Inputs:

- camera matrices from `Camera`
- Gaussian centers / scales / rotations / opacity / SH
- background color tensor on CUDA

Output:

- `render`: `(3, H, W)`
- `viewspace_points`: `(N_gs, 3)` screen-space proxy for gradients
- `visibility_filter`: `(N_gs,)`
- `radii`: `(N_gs,)`

The renderer assumes:

- CUDA is available
- rasterizer extension is compiled
- all tensors are on GPU

### 7. Loss and Optimization

Training loss in [`train.py`](/root/AvatarStack/modules/FlashAvatar-code/train.py):

- base Huber reconstruction loss
- extra mouth-weighted Huber term with weight `40`
- LPIPS-VGG added only after `15000` iterations with weight `0.05`

Concretely:

- `image`: `(3, 512, 512)`
- `gt_image`: `(3, 512, 512)`
- `mouth_mask`: `(1, 512, 512)`
- `head_mask`: `(1, 512, 512)`

Optimization schedule:

- total iterations: `150000`
- deform MLP LR: `1e-4`
- Gaussian learning rates inherited from 3D-GS-style defaults in [`arguments/__init__.py`](/root/AvatarStack/modules/FlashAvatar-code/arguments/__init__.py)

## Split Logic

`Scene_mica(train_type=...)` uses:

- `0`: train split
- `1`: test split
- `2`: eval split

Original repo logic assumed at least 500 frames for testing. In this workspace, [`scene/__init__.py`](/root/AvatarStack/modules/FlashAvatar-code/scene/__init__.py) has been patched so short sequences do not produce negative ranges.

Current behavior:

- if `N_frames <= 500`, test uses all frames and eval uses up to 50
- otherwise, test uses the last 500 frames

## Important Constraints

- **Preprocessed input is mandatory**: raw `mp4` alone is insufficient
- **Resolution is effectively fixed to 512** in several places:
  - background tensor is created as `(3, 512, 512)`
  - training/test visualization assumes `args.image_res=512`
- **Frame alignment uses `frame_delta = 1`**:
  - tracker frame `00000.frame` pairs with image `00001.jpg`
- **Batch size is effectively 1**
- **CUDA is hardcoded** in many modules
- **The code expects valid `00000.frame`** to derive shape and camera intrinsics
- **Jaw / eye rotations are 6D rotations**, not axis-angle, at model input time
- **No robustness layer** exists for missing masks or missing frame files

## Boundary Conditions and Failure Modes

### Filesystem / Data Issues

- missing `checkpoint/00000.frame` will fail scene initialization immediately
- missing `alpha`, `parsing`, or `imgs` files for any indexed frame will crash image loading
- mismatched counts between frames and checkpoints are not validated up front
- if image filenames do not obey the `+1` offset convention, training/testing silently use the wrong supervision

### Alpha Quality Requirements

An alpha set is suitable for FlashAvatar only if it satisfies all of the following:

- temporal stability: sampled beginning / middle / ending frames show no obvious contour flicker, jitter, or sudden size changes
- key-region completeness: hair, ears, neck, chin, and other thin or weak-contrast regions are not consistently cut away
- clean background separation: no large residual background blobs, noise, or fragments remain near the outer silhouette
- strict alignment: alpha stays frame-wise identical to the source image in size, indexing, and pixel position, with no shift, scale, or crop mismatch
- training usability: once consumed by [`scene/__init__.py`](/root/AvatarStack/modules/FlashAvatar-code/scene/__init__.py) and the train/test pipeline, it should not cause obviously blurry edges, contour drift, or background leakage into the learned avatar

### Sequence Length

- upstream repo assumes long sequences; short clips needed the local patch in [`scene/__init__.py`](/root/AvatarStack/modules/FlashAvatar-code/scene/__init__.py)

### Geometry / Tracking Quality

- the method depends heavily on accurate FLAME tracking
- bad global pose or poor expression tracking causes visible misalignment
- non-rigid hair motion is not explicitly modeled beyond learned Gaussian offsets

### Runtime / Environment

- requires a working CUDA PyTorch environment
- requires `pytorch3d`, `simple-knn`, and `diff-gaussian-rasterization`
- CPU execution is not supported in practice

## Practical Reading of the Paper vs This Repo

Paper idea:

- UV-space uniform Gaussian initialization
- mesh-attached motion
- expression-conditioned dynamic offset

Repo implementation:

- UV resolution fixed to `128`
- sampled visible UV points = `14876`
- head-region filtered Gaussians = `13453`
- condition code = `expr + jaw + eyes + eyelids = 120 dims`
- offset MLP input = `171 dims`
- output = `10 dims`

This repo faithfully implements the core method, but omits upstream preprocessing and exposes only the single-identity optimization/rendering stage.

## Agent Guidance

If you need to modify this repo, start from the task type:

- data loading / split bugs:
  - [`scene/__init__.py`](/root/AvatarStack/modules/FlashAvatar-code/scene/__init__.py)
- camera / projection / novel view:
  - [`scene/cameras.py`](/root/AvatarStack/modules/FlashAvatar-code/scene/cameras.py)
  - [`utils/graphics_utils.py`](/root/AvatarStack/modules/FlashAvatar-code/utils/graphics_utils.py)
  - [`novel_view.py`](/root/AvatarStack/modules/FlashAvatar-code/novel_view.py)
- FLAME conditioning / shape logic:
  - [`src/deform_model.py`](/root/AvatarStack/modules/FlashAvatar-code/src/deform_model.py)
  - [`flame/flame_mica.py`](/root/AvatarStack/modules/FlashAvatar-code/flame/flame_mica.py)
- Gaussian attributes / checkpoints:
  - [`scene/gaussian_model.py`](/root/AvatarStack/modules/FlashAvatar-code/scene/gaussian_model.py)
- rendering correctness / performance:
  - [`gaussian_renderer/__init__.py`](/root/AvatarStack/modules/FlashAvatar-code/gaussian_renderer/__init__.py)

Before changing model behavior, verify three things first:

1. whether the issue is actually data-format / frame-alignment related
2. whether the fix should happen in FLAME-space or Gaussian-space
3. whether the code path is training-only, test-only, or shared

## Workspace-Specific Notes

This workspace is not a pristine upstream checkout.

Observed local changes / additions:

- [`scene/__init__.py`](/root/AvatarStack/modules/FlashAvatar-code/scene/__init__.py) has a short-sequence patch
- [`novel_view.py`](/root/AvatarStack/modules/FlashAvatar-code/novel_view.py) exists as a local rendering utility
- the PDF paper is stored locally for reference

If reproducing paper numbers exactly, treat local utilities and local patches separately from the original repository.
