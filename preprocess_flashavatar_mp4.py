import argparse
import json
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parent
MODULES_ROOT = REPO_ROOT.parent
BISENET_ROOT = MODULES_ROOT / "BiSeNet-face-parsing"
MATANYONE_ROOT = MODULES_ROOT / "MatAnyone"
METRICAL_TRACKER_ROOT = MODULES_ROOT / "metrical-tracker"


@dataclass
class PreprocessResult:
    identity_id: str
    frame_count: int
    dataset_dir: Path
    status: str
    tracker_checkpoint_dir: Optional[Path]


def _ensure_sys_path(path: Path) -> None:
    text = str(path)
    if text not in sys.path:
        sys.path.insert(0, text)


def _sorted_paths(folder: Path, pattern: str) -> list[Path]:
    return sorted(folder.glob(pattern))


def _run_command(command: list[str], cwd: Optional[Path] = None) -> None:
    subprocess.run(command, check=True, cwd=None if cwd is None else str(cwd))


def _extract_video_frames(input_mp4_path: Path, raw_frame_dir: Path) -> list[Path]:
    raw_frame_dir.mkdir(parents=True, exist_ok=True)
    existing_frames = _sorted_paths(raw_frame_dir, "*.jpg")
    if existing_frames:
        return existing_frames

    _run_command(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(input_mp4_path),
            "-start_number",
            "0",
            str(raw_frame_dir / "%05d.jpg"),
        ]
    )
    frame_paths = _sorted_paths(raw_frame_dir, "*.jpg")
    if not frame_paths:
        raise RuntimeError(f"no frames decoded from {input_mp4_path}")
    return frame_paths


def _load_bisenet_model(model_name: str, weight_path: Path):
    _ensure_sys_path(BISENET_ROOT)
    import torch
    from inference import load_model

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(model_name, 19, str(weight_path), device)
    return model, device


def _infer_bisenet_mask(
    image_path: Path,
    output_path: Path,
    model_name: str,
    weight_path: Path,
) -> np.ndarray:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        return np.array(Image.open(output_path).convert("L"), dtype=np.uint8)

    _ensure_sys_path(BISENET_ROOT)
    import torch
    from inference import prepare_image

    model, device = _load_bisenet_model(model_name, weight_path)
    image = Image.open(image_path).convert("RGB")
    image_batch = prepare_image(image).to(device)

    with torch.no_grad():
        output = model(image_batch)[0]

    predicted_mask = output.squeeze(0).detach().cpu().numpy().argmax(0).astype(np.uint8)
    restored_mask = Image.fromarray(predicted_mask, mode="L").resize(
        image.size, resample=Image.NEAREST
    )
    restored_mask.save(output_path)
    return np.array(restored_mask, dtype=np.uint8)


def _save_binary_mask(mask: np.ndarray, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(mask.astype(np.uint8), mode="L").save(output_path)


def _generate_seed_masks(
    parsing_raw: np.ndarray,
    seed_dir: Path,
) -> tuple[Path, Path]:
    person_seed_path = seed_dir / "00000_person_seed.png"
    mouth_seed_path = seed_dir / "00000_mouth_seed.png"

    person_seed = np.where(parsing_raw != 0, 255, 0).astype(np.uint8)
    mouth_seed = np.where(np.isin(parsing_raw, [11, 12, 13]), 255, 0).astype(np.uint8)

    if not person_seed.any():
        raise RuntimeError("person seed is empty")
    if not mouth_seed.any():
        raise RuntimeError("mouth seed is empty")

    _save_binary_mask(person_seed, person_seed_path)
    _save_binary_mask(mouth_seed, mouth_seed_path)
    return person_seed_path, mouth_seed_path


def _run_matanyone(
    input_source_path: Path,
    mask_path: Path,
    output_path: Path,
    ckpt_path: Path,
) -> None:
    expected_dir = output_path / input_source_path.name / "pha"
    if expected_dir.exists() and any(expected_dir.glob("*.png")):
        return

    _ensure_sys_path(MATANYONE_ROOT)
    from inference_matanyone import main as matanyone_main

    try:
        matanyone_main(
            input_path=str(input_source_path),
            mask_path=str(mask_path),
            output_path=str(output_path),
            ckpt_path=str(ckpt_path),
            save_image=True,
            max_size=-1,
        )
    except Exception:
        if expected_dir.exists() and any(expected_dir.glob("*.png")):
            return
        raise


def _copy_images(src_paths: Iterable[Path], dst_dir: Path) -> None:
    dst_dir.mkdir(parents=True, exist_ok=True)
    for src_path in src_paths:
        dst_path = dst_dir / src_path.name
        if dst_path.exists():
            continue
        shutil.copy2(src_path, dst_path)


def _save_alpha_and_parsing(
    frame_count: int,
    pha_dir: Path,
    alpha_dir: Path,
    parsing_dir: Path,
    suffix: str,
    threshold: int,
    save_alpha_jpg: bool,
) -> None:
    alpha_dir.mkdir(parents=True, exist_ok=True)
    parsing_dir.mkdir(parents=True, exist_ok=True)

    for frame_index in range(frame_count):
        src_path = pha_dir / f"{frame_index:04d}.png"
        if not src_path.exists():
            raise RuntimeError(f"missing MatAnyone matte: {src_path}")

        matte = np.array(Image.open(src_path).convert("L"), dtype=np.uint8)
        frame_name = f"{frame_index:05d}"

        if save_alpha_jpg:
            Image.fromarray(matte, mode="L").save(alpha_dir / f"{frame_name}.jpg", quality=95)

        binary = np.where(matte >= threshold, 255, 0).astype(np.uint8)
        Image.fromarray(binary, mode="L").save(parsing_dir / f"{frame_name}_{suffix}.png")


def _write_tracker_config(config_path: Path, identity_id: str) -> None:
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(
        "\n".join(
            [
                f"actor: ./input/{identity_id}",
                "save_folder: ./output/",
                "begin_frames: 1",
                "optimize_shape: true",
                "optimize_jaw: true",
                "",
            ]
        ),
        encoding="utf-8",
    )


def _run_tracker_if_ready(
    identity_id: str,
    input_mp4_path: Path,
    mica_identity_path: Optional[Path],
) -> Optional[Path]:
    if mica_identity_path is None or not mica_identity_path.exists():
        return None

    tracker_actor_dir = METRICAL_TRACKER_ROOT / "input" / identity_id
    tracker_actor_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(input_mp4_path, tracker_actor_dir / "video.mp4")
    shutil.copy2(mica_identity_path, tracker_actor_dir / "identity.npy")

    config_path = METRICAL_TRACKER_ROOT / "configs" / "actors" / f"{identity_id}.yml"
    _write_tracker_config(config_path, identity_id)

    checkpoint_dir = METRICAL_TRACKER_ROOT / "output" / identity_id / "checkpoint"
    if not checkpoint_dir.exists() or not any(checkpoint_dir.glob("*.frame")):
        _run_command(
            ["python", "tracker.py", "--cfg", f"./configs/actors/{identity_id}.yml"],
            cwd=METRICAL_TRACKER_ROOT,
        )

    return checkpoint_dir


def _validate_dataset(dataset_dir: Path, frame_count: int) -> None:
    imgs = _sorted_paths(dataset_dir / "imgs", "*.jpg")
    alpha = _sorted_paths(dataset_dir / "alpha", "*.jpg")
    neck = _sorted_paths(dataset_dir / "parsing", "*_neckhead.png")
    mouth = _sorted_paths(dataset_dir / "parsing", "*_mouth.png")

    counts = [len(imgs), len(alpha), len(neck), len(mouth)]
    if counts != [frame_count, frame_count, frame_count, frame_count]:
        raise RuntimeError(f"dataset count mismatch for {dataset_dir}: {counts}")

    for frame_index in (0, frame_count // 2, frame_count - 1):
        frame_name = f"{frame_index:05d}"
        image_path = dataset_dir / "imgs" / f"{frame_name}.jpg"
        alpha_path = dataset_dir / "alpha" / f"{frame_name}.jpg"
        neck_path = dataset_dir / "parsing" / f"{frame_name}_neckhead.png"
        mouth_path = dataset_dir / "parsing" / f"{frame_name}_mouth.png"

        with Image.open(image_path) as image:
            image_size = image.size
        for path in [alpha_path, neck_path, mouth_path]:
            with Image.open(path) as image:
                if image.size != image_size:
                    raise RuntimeError(f"resolution mismatch: {path}")


def preprocess_mp4_for_flashavatar(
    input_mp4_path: Path,
    identity_id: str,
    work_root: Path,
    dataset_root: Path,
    bisenet_weight_path: Path,
    matanyone_ckpt_path: Path,
    mica_identity_path: Optional[Path] = None,
    bisenet_model_name: str = "resnet18",
    mouth_threshold: int = 127,
    neckhead_threshold: int = 127,
) -> PreprocessResult:
    actor_work_root = work_root / identity_id
    raw_frame_dir = actor_work_root / "frames_raw"
    seed_dir = actor_work_root / "seed_masks"
    matting_root = actor_work_root / "matanyone"

    dataset_dir = dataset_root / identity_id
    imgs_dir = dataset_dir / "imgs"
    alpha_dir = dataset_dir / "alpha"
    parsing_dir = dataset_dir / "parsing"

    frame_paths = _extract_video_frames(input_mp4_path, raw_frame_dir)
    frame_count = len(frame_paths)

    parsing_raw = _infer_bisenet_mask(
        image_path=raw_frame_dir / "00000.jpg",
        output_path=seed_dir / "00000_parsing_raw.png",
        model_name=bisenet_model_name,
        weight_path=bisenet_weight_path,
    )
    person_seed_path, mouth_seed_path = _generate_seed_masks(parsing_raw, seed_dir)

    _run_matanyone(
        input_source_path=raw_frame_dir,
        mask_path=person_seed_path,
        output_path=matting_root / "person",
        ckpt_path=matanyone_ckpt_path,
    )
    _run_matanyone(
        input_source_path=raw_frame_dir,
        mask_path=mouth_seed_path,
        output_path=matting_root / "mouth",
        ckpt_path=matanyone_ckpt_path,
    )

    _copy_images(frame_paths, imgs_dir)
    _save_alpha_and_parsing(
        frame_count=frame_count,
        pha_dir=matting_root / "person" / raw_frame_dir.name / "pha",
        alpha_dir=alpha_dir,
        parsing_dir=parsing_dir,
        suffix="neckhead",
        threshold=neckhead_threshold,
        save_alpha_jpg=True,
    )
    _save_alpha_and_parsing(
        frame_count=frame_count,
        pha_dir=matting_root / "mouth" / raw_frame_dir.name / "pha",
        alpha_dir=alpha_dir,
        parsing_dir=parsing_dir,
        suffix="mouth",
        threshold=mouth_threshold,
        save_alpha_jpg=False,
    )

    tracker_checkpoint_dir = _run_tracker_if_ready(
        identity_id=identity_id,
        input_mp4_path=input_mp4_path,
        mica_identity_path=mica_identity_path,
    )

    _validate_dataset(dataset_dir, frame_count)
    status = "flashavatar_ready" if tracker_checkpoint_dir is not None else "mask_ready_tracker_pending"
    return PreprocessResult(
        identity_id=identity_id,
        frame_count=frame_count,
        dataset_dir=dataset_dir,
        status=status,
        tracker_checkpoint_dir=tracker_checkpoint_dir,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess an mp4 into FlashAvatar dataset layout.")
    parser.add_argument("--input-mp4", type=Path, required=True)
    parser.add_argument("--identity-id", type=str, required=True)
    parser.add_argument("--work-root", type=Path, default=REPO_ROOT / "preprocess_work")
    parser.add_argument("--dataset-root", type=Path, default=REPO_ROOT / "dataset")
    parser.add_argument(
        "--bisenet-weight",
        type=Path,
        default=BISENET_ROOT / "weights" / "resnet18.pt",
    )
    parser.add_argument(
        "--matanyone-ckpt",
        type=Path,
        default=MATANYONE_ROOT / "pretrained_models" / "matanyone.pth",
    )
    parser.add_argument("--mica-identity", type=Path, default=None)
    parser.add_argument("--mouth-threshold", type=int, default=127)
    parser.add_argument("--neckhead-threshold", type=int, default=127)
    args = parser.parse_args()

    result = preprocess_mp4_for_flashavatar(
        input_mp4_path=args.input_mp4,
        identity_id=args.identity_id,
        work_root=args.work_root,
        dataset_root=args.dataset_root,
        bisenet_weight_path=args.bisenet_weight,
        matanyone_ckpt_path=args.matanyone_ckpt,
        mica_identity_path=args.mica_identity,
        mouth_threshold=args.mouth_threshold,
        neckhead_threshold=args.neckhead_threshold,
    )

    print(
        json.dumps(
            {
                "identity_id": result.identity_id,
                "frame_count": result.frame_count,
                "dataset_dir": str(result.dataset_dir),
                "tracker_checkpoint_dir": (
                    None if result.tracker_checkpoint_dir is None else str(result.tracker_checkpoint_dir)
                ),
                "status": result.status,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
