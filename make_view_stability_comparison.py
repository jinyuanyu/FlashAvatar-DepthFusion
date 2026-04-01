import argparse
from pathlib import Path

import cv2
import numpy as np


def _parse_frame_indices(raw: str, total_frames: int):
    if raw:
        return [max(0, min(total_frames - 1, int(item))) for item in raw.split(",") if item.strip()]

    anchors = (0.15, 0.40, 0.65, 0.90)
    return [max(0, min(total_frames - 1, round((total_frames - 1) * t))) for t in anchors]


def _read_frame(video_path: Path, index: int):
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")
    capture.set(cv2.CAP_PROP_POS_FRAMES, index)
    ok, frame = capture.read()
    capture.release()
    if not ok:
        raise RuntimeError(f"Failed to read frame {index} from {video_path}")
    return frame


def _frame_count(video_path: Path):
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")
    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    capture.release()
    if total <= 0:
        raise RuntimeError(f"Video has no frames: {video_path}")
    return total


def _crop_render(frame: np.ndarray, mode: str):
    height, width = frame.shape[:2]
    if mode == "none":
        return frame
    if mode == "right":
        return frame[:, width // 2 :, :]
    if mode == "left":
        return frame[:, : width // 2, :]
    if width >= height * 2:
        return frame[:, width // 2 :, :]
    return frame


def _put_label(image, text, x, y, scale=0.7, color=(255, 255, 255), thickness=2):
    cv2.putText(
        image,
        text,
        (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        thickness,
        cv2.LINE_AA,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Create a comparison figure for train-view stability vs novel-view drift/artifacts."
    )
    parser.add_argument("--train-video", required=True, help="Training-view render video, e.g. dataset/<id>/log/test.mp4")
    parser.add_argument("--novel-video", required=True, help="Novel-view render video, e.g. dataset/<id>/log/test_novel_*.mp4")
    parser.add_argument("--output", required=True, help="Output PNG path.")
    parser.add_argument("--title", default="View Stability Comparison", help="Figure title.")
    parser.add_argument("--subtitle", default="Left: train-view render (stable)   Right: novel-view render (drift / artifacts)", help="Figure subtitle.")
    parser.add_argument("--footer", default="Novel-view render typically exposes silhouette drift, jaw/ear tearing, and geometry instability beyond the training camera manifold.", help="Footer note.")
    parser.add_argument("--identity-label", default="", help="Optional label such as Obama or Mead2.")
    parser.add_argument("--frames", default="", help="Comma-separated frame indices. Default: 15%, 40%, 65%, 90% of the clip.")
    parser.add_argument("--tile-size", type=int, default=256, help="Tile size for each crop.")
    parser.add_argument("--gap", type=int, default=20, help="Gap between rows.")
    parser.add_argument("--train-crop", choices=["auto", "right", "left", "none"], default="auto", help="Crop mode for the train video.")
    parser.add_argument("--novel-crop", choices=["auto", "right", "left", "none"], default="auto", help="Crop mode for the novel video.")
    args = parser.parse_args()

    train_video = Path(args.train_video)
    novel_video = Path(args.novel_video)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    total_frames = min(_frame_count(train_video), _frame_count(novel_video))
    frame_indices = _parse_frame_indices(args.frames, total_frames)

    rows = []
    for frame_index in frame_indices:
        train_frame = _read_frame(train_video, frame_index)
        novel_frame = _read_frame(novel_video, frame_index)

        train_crop = _crop_render(train_frame, args.train_crop)
        novel_crop = _crop_render(novel_frame, args.novel_crop)

        train_tile = cv2.resize(train_crop, (args.tile_size, args.tile_size), interpolation=cv2.INTER_AREA)
        novel_tile = cv2.resize(novel_crop, (args.tile_size, args.tile_size), interpolation=cv2.INTER_AREA)
        pair = cv2.hconcat([train_tile, novel_tile])
        _put_label(pair, f"frame {frame_index:04d}", 10, 24)
        rows.append(pair)

    header_h = 92
    footer_h = 56
    body_h = len(rows) * args.tile_size + max(0, len(rows) - 1) * args.gap
    canvas_h = header_h + body_h + footer_h
    canvas_w = args.tile_size * 2
    canvas = np.full((canvas_h, canvas_w, 3), 245, dtype=np.uint8)

    title = args.title
    if args.identity_label:
        title = f"{args.identity_label}: {title}"

    _put_label(canvas, title, 16, 34, scale=0.82, color=(25, 25, 25), thickness=2)
    _put_label(canvas, args.subtitle, 16, 66, scale=0.54, color=(80, 80, 80), thickness=1)

    y = header_h
    for row in rows:
        canvas[y : y + args.tile_size, :, :] = row
        y += args.tile_size + args.gap

    _put_label(canvas, args.footer, 12, canvas_h - 18, scale=0.46, color=(55, 55, 55), thickness=1)

    if not cv2.imwrite(str(output_path), canvas):
        raise RuntimeError(f"Failed to save figure: {output_path}")

    print(f"[ViewStability] Saved figure: {output_path}")


if __name__ == "__main__":
    main()
