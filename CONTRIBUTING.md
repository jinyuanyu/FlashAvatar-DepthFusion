# Contributing

Thanks for helping improve `FlashAvatar-DepthFusion`.

This repository is a research fork, so the most valuable contributions are usually:

- bug fixes in the depth-supervision path
- preprocessing improvements and validation tooling
- documentation clarifications
- reproducibility and setup fixes
- cleaner public examples that do not require proprietary assets

## Before Opening A PR

- keep changes focused and easy to review
- describe the motivation and expected behavior change
- note whether the change affects training, preprocessing, or documentation
- mention any required external assets that reviewers would need

## Repository Hygiene

- do not commit private datasets
- do not commit pretrained weights unless redistribution is clearly allowed
- do not commit FLAME assets or other restricted files
- avoid committing generated videos, checkpoints, or large intermediate caches

The repository intentionally excludes items such as:

- `dataset/`
- `pretrained_models/`
- `metrical-tracker/`
- large media files

## Code Style

- prefer small, localized changes
- keep research assumptions explicit in code comments and docs
- update README or `docs/` when behavior changes
- keep file naming and frame numbering conventions consistent with the preprocessing schema

## Good PR Examples

- adding a safer depth-mask rule with documentation
- improving error messages for missing `depth/*.npy`
- adding validation for frame-count mismatches
- making preprocessing outputs easier to inspect or resume

## Issues And Discussions

When reporting a bug, please include:

- the command you ran
- the identity layout you used
- whether depth supervision was enabled
- the failing stack trace or incorrect behavior
- whether the issue is reproducible on a clean run

## Scope

This repository is positioned as a public engineering fork rather than the official FlashAvatar implementation. Contributions that make the fork easier to understand, reproduce, and evaluate are especially welcome.
