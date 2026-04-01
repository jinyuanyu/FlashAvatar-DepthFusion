# Overview

## What This Fork Changes

`FlashAvatar-DepthFusion` keeps the original FlashAvatar training pipeline centered around:

- FLAME-conditioned deformation
- 3D Gaussian rendering
- RGB reconstruction
- metrical-tracker camera / motion estimation

The main extension is a monocular-depth supervision branch that regularizes geometry without introducing an explicit intermediate mesh-fitting stage.

## Core Additions

### Monocular depth input

Each frame can optionally load:

- `dataset/<id>/depth/{frame}.npy`

Expected properties:

- `float32`
- same spatial resolution as `imgs`
- same numbering as `imgs`
- recommended source: VideoDepth Anything

### Scale-invariant depth loss

This fork adds a scale-invariant depth objective implemented as:

```text
L_depth = 1 - corr(render_depth[valid], mono_depth[valid])
```

where `corr` is Pearson correlation.

This formulation is more appropriate than direct absolute-depth regression when the input depth is only reliable up to relative ordering and affine ambiguity.

### Conservative valid-depth mask

Depth is only trusted in a high-confidence facial core.

Recommended rule:

```text
valid_depth_mask = erode(head_mask, 2~3 px) - mouth_mask
```

This avoids:

- jawline and hair-boundary depth bleeding
- open-mouth depth conflicts
- unstable supervision at silhouette edges

## Code Paths

The main modified paths are:

- [`train.py`](../train.py)
- [`scene/__init__.py`](../scene/__init__.py)
- [`scene/cameras.py`](../scene/cameras.py)
- [`utils/loss_utils.py`](../utils/loss_utils.py)

## Current Scope

This repository should be read as:

- a practical research fork
- an engineering prototype for depth-regularized head avatars
- a public-facing version of the local workspace improvements

It should not be read as:

- the official upstream FlashAvatar repository
- a benchmark-finalized release
- a complete one-click production pipeline
