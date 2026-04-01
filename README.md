# FlashAvatar-DepthFusion

Monocular-depth-regularized FlashAvatar with scale-invariant depth supervision for head avatar reconstruction.

This repository is a practical derivative of the original FlashAvatar codebase, extended with:

- per-frame monocular depth loading
- SIDL-style depth supervision via Pearson correlation loss
- conservative valid-mask design for jaw / hair boundary safety
- mouth-interior depth exclusion for non-rigid speech motion
- preprocessing documentation for `mp4 -> imgs / alpha / parsing / depth / checkpoint`

The goal is to improve geometric stability without introducing an explicit intermediate 3D mesh reconstruction stage.

## Highlights

- `train.py` adds optional monocular-depth supervision through `--use_depth_supervision`.
- `Scene_mica` can read `dataset/<id>/depth/*.npy` and build a valid depth mask online.
- The depth loss is scale-invariant:

```text
L_depth = 1 - corr(render_depth[valid], mono_depth[valid])
```

- The valid region follows a conservative rule:

```text
valid_depth_mask = erode(head_mask, 2~3 px) - mouth_mask
```

This avoids the two most common monocular-depth failure modes:

- edge depth bleeding near jawlines and hair boundaries
- incorrect depth inside open-mouth regions

## Architecture

The current training design keeps FlashAvatar's FLAME-conditioned Gaussian pipeline, then injects monocular depth as an extra regularization branch rather than as a standalone geometry source.

![Depth Fusion Architecture](figures/depth_fusion_architecture/depth_fusion_architecture_preview.png)

Key idea:

- RGB reconstruction remains the primary optimization target.
- FLAME and metrical-tracker still provide the motion / camera backbone.
- Monocular depth only regularizes geometry on conservative, high-confidence facial regions.

## Qualitative Results

Example comparison cards generated from the current workspace:

| Identity | Result |
| --- | --- |
| Obama | ![Obama Comparison](_tmp_compare/Obama_contact.png) |
| Mead2 | ![Mead2 Comparison](_tmp_compare/Mead2_contact.png) |
| Luoxiang | ![Luoxiang Comparison](_tmp_compare/luoxiang_contact.png) |

## What Is Implemented

Depth-related code already wired into this repository:

- [train.py](train.py)
  adds:
  - `--use_depth_supervision`
  - `--depth_loss_weight`
  - `--depth_start_iter`
  - `--depth_erode_kernel`
  - `--min_depth_samples`
- [utils/loss_utils.py](utils/loss_utils.py)
  adds `scale_invariant_depth_loss(...)`
- [scene/__init__.py](scene/__init__.py)
  loads `depth/*.npy` and builds `depth_valid_mask`
- [scene/cameras.py](scene/cameras.py)
  carries `mono_depth` and `depth_valid_mask` per frame

Important note:

- the current renderer path still computes depth supervision by projecting Gaussian centers into image space
- it does not yet expose a full rasterized dense depth map as a default return of `render(...)`

## Data Layout

Each identity uses the following layout:

```text
dataset/
  <id>/
    imgs/
    alpha/
    depth/          # float32 .npy, optional but required when depth supervision is enabled
    parsing/

metrical-tracker/
  output/
    <id>/
      checkpoint/
```

Required frame alignment:

```text
00000.frame  <->  00001.jpg
00001.frame  <->  00002.jpg
...
```

Depth convention:

- `dataset/<id>/depth/00001.npy`
- one `float32` depth map per frame
- same resolution and numbering as `imgs/*.jpg`
- recommended source: VideoDepth Anything

## Training

Baseline training:

```bash
python train.py --idname <id_name>
```

Training with monocular depth supervision:

```bash
python train.py \
  --idname <id_name> \
  --use_depth_supervision \
  --depth_loss_weight 0.05 \
  --depth_start_iter 0 \
  --depth_erode_kernel 3 \
  --min_depth_samples 256
```

Test rendering:

```bash
python test.py --idname <id_name> --checkpoint dataset/<id_name>/log/ckpt/chkpnt.pth
```

Novel-view rendering:

```bash
python novel_view.py --idname <id_name> --checkpoint dataset/<id_name>/log/ckpt/chkpnt.pth
```

## Preprocessing Assets

This repo also includes preprocessing-side documents and helper scripts:

- [preprocess_flashavatar_mp4.py](preprocess_flashavatar_mp4.py)
- [pseudocode.md](pseudocode.md)
- [data_schema.json](data_schema.json)
- [Agents.zh-CN.md](Agents.zh-CN.md)
- [test_flashavatar_schema.py](test_flashavatar_schema.py)

Visualization / figure scripts:

- [generate_depth_fusion_architecture.py](generate_depth_fusion_architecture.py)
- [draw_depth_fusion_flowchart.py](draw_depth_fusion_flowchart.py)
- [make_view_stability_comparison.py](make_view_stability_comparison.py)

## Setup Notes

Create the environment:

```bash
conda env create --file environment.yml
conda activate FlashAvatar
```

Install PyTorch3D:

```bash
conda install -c fvcore -c iopath -c conda-forge fvcore iopath
conda install -c bottler nvidiacub
conda install pytorch3d -c pytorch3d
```

Submodules / native extensions:

- `submodules/simple-knn`
- `submodules/diff-gaussian-rasterization`

You may need to rebuild the CUDA extensions in your own environment.

## External Assets Not Included

For legal and size reasons, this repository does not bundle every runtime asset.

You still need to provide or download:

- FLAME geometry model files required by the `flame/` module
- FLAME mask assets if your environment depends on them
- monocular depth maps if you enable depth supervision
- metrical-tracker outputs under `metrical-tracker/output/<id>/checkpoint`

## Repository Positioning

This is an engineering-focused research fork for experimenting with:

- FlashAvatar + monocular depth fusion
- conservative depth supervision masking
- improved preprocessing contracts for custom videos

It should be treated as a practical extension of FlashAvatar, not as the official upstream repository.

## Acknowledgements

This repository builds on:

- FlashAvatar
- 3D Gaussian Splatting
- metrical-tracker / MICA
- BiSeNet face parsing
- MatAnyone
- VideoDepth Anything

## Citation

If you use this codebase, please cite the original FlashAvatar work and acknowledge this depth-fusion derivative accordingly.

```bibtex
@inproceedings{xiang2024flashavatar,
  author    = {Jun Xiang and Xuan Gao and Yudong Guo and Juyong Zhang},
  title     = {FlashAvatar: High-fidelity Head Avatar with Efficient Gaussian Embedding},
  booktitle = {CVPR},
  year      = {2024}
}
```
