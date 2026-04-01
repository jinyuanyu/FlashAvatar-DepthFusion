# Preprocessing

## Goal

Prepare a monocular video sequence into a FlashAvatar-compatible training package, with optional depth maps for depth supervision.

## Required Outputs

Per identity:

```text
dataset/<id>/
  imgs/
  alpha/
  parsing/
  depth/          # optional, needed for depth supervision

metrical-tracker/output/<id>/
  checkpoint/
```

## File Semantics

### `imgs/*.jpg`

- extracted RGB frames
- 5-digit numbering recommended

### `alpha/*.jpg`

- foreground alpha / matte
- same frame numbering and resolution as `imgs`

### `parsing/*_neckhead.png`

- binary head/neck valid region
- used to constrain supervision and background compositing

### `parsing/*_mouth.png`

- binary mouth region
- used for stronger mouth RGB supervision
- also used to exclude mouth interior from depth supervision

### `depth/*.npy`

- optional per-frame monocular depth
- `float32`
- same resolution and numbering as `imgs`
- recommended source: VideoDepth Anything

### `checkpoint/*.frame`

- FLAME parameters and camera parameters produced by metrical-tracker

Required alignment:

```text
00000.frame <-> 00001.jpg
00001.frame <-> 00002.jpg
...
```

## Preprocessing Stages

1. Extract frames from input video.
2. Run first-frame face parsing.
3. Create person and mouth seeds.
4. Run temporal matte propagation.
5. Build `alpha` and `parsing` outputs.
6. Optionally run monocular depth estimation and save `depth/*.npy`.
7. Run metrical-tracker to produce `checkpoint/*.frame`.
8. Validate frame counts, resolution alignment, and naming consistency.

## Validation Checklist

- all modalities share the same frame numbering
- all modalities share the same resolution
- `00000.frame` exists
- `count(checkpoint) + 1 == count(imgs)`
- `depth/*.npy` are finite and not degenerate if depth supervision is enabled
- mouth and head masks are visually aligned with RGB
- tracker pose and expression appear synchronized with the input
