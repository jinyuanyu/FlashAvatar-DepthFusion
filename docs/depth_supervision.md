# Depth Supervision

## Motivation

Monocular head-video reconstruction has a familiar weakness: RGB alone often leaves geometry underconstrained around profile views, cheeks, jawlines, and depth ordering changes during expression.

A monocular depth prior can help, but only if it is used conservatively.

## Design Choice

This fork does not convert monocular depth into an explicit mesh target.

Instead, it uses depth as a soft geometric regularizer on top of the existing FlashAvatar pipeline:

- FLAME still provides structured facial motion priors
- 3D Gaussians still represent the rendered avatar
- RGB remains the dominant reconstruction signal
- monocular depth only nudges geometry toward more stable relative depth structure

## Loss Definition

The current loss is:

```text
L_depth = 1 - corr(render_depth[valid], mono_depth[valid])
```

Properties:

- scale-invariant
- shift-invariant after centering
- robust to the relative-depth nature of monocular depth estimators

## Why Pearson Correlation

Direct L1/L2 supervision on monocular depth is often brittle because:

- monocular depth is not metrically calibrated
- per-frame scale can drift
- global depth offsets may be arbitrary

Pearson correlation fits this setting better because it rewards consistent relative ordering and shape trends instead of exact metric equality.

## Edge Depth Bleeding

Single-image or video monocular depth models often produce smooth depth ramps near silhouette boundaries:

- jawline vs background
- hair strands vs background
- ears vs background

In reality, these regions often contain sharp depth discontinuities.

If that smoothed depth is used naively, geometry can be pulled outward and produce:

- cone-like side profiles
- widened silhouettes
- unstable contour geometry

## Mouth Interior Conflicts

Open-mouth frames are another failure zone.

Monocular depth models often flatten:

- lips
- teeth
- oral cavity
- inner-mouth shadows

This conflicts with:

- FLAME jaw motion
- RGB evidence
- expression-dependent non-rigid deformation

So the mouth interior should not be depth-supervised.

## Valid Mask Strategy

Recommended valid region:

```text
valid_depth_mask = erode(head_mask, 2~3 px) - mouth_mask
```

Interpretation:

- start with the stable head/neck supervision region
- shrink it inward by a few pixels
- remove mouth supervision entirely

This preserves depth supervision on:

- forehead
- cheeks
- nose bridge
- stable lower-face surface

while dropping it from:

- silhouette boundaries
- hair edges
- jaw-background transitions
- mouth interior

## Current Implementation Boundary

The current code path supervises depth by projecting Gaussian centers into the image plane and sampling monocular depth there.

That means:

- the implementation is lightweight
- the renderer API stays mostly unchanged
- the loss is easy to integrate into existing training

But it also means:

- it is not yet a dense rendered-depth supervision path
- a future dense depth rasterization branch would still be valuable
