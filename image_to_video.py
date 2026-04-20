#!/usr/bin/env python3
"""Generate an animated video from a still image using the Ken Burns effect."""

import argparse
import sys
import numpy as np

try:
    import cv2
except ImportError:
    print("Install opencv-python: pip install opencv-python numpy")
    sys.exit(1)


def ken_burns(image: np.ndarray, duration: float, fps: int,
              zoom_start: float, zoom_end: float,
              pan_x: float, pan_y: float) -> list[np.ndarray]:
    """Return frames with a slow zoom/pan (Ken Burns) effect."""
    h, w = image.shape[:2]
    total_frames = int(duration * fps)
    frames = []

    for i in range(total_frames):
        t = i / max(total_frames - 1, 1)

        # Ease in/out (smoothstep)
        t_ease = t * t * (3.0 - 2.0 * t)

        zoom = zoom_start + (zoom_end - zoom_start) * t_ease
        crop_w = int(w / zoom)
        crop_h = int(h / zoom)

        # Center offset + gentle pan
        cx = w / 2 + pan_x * w * t_ease
        cy = h / 2 + pan_y * h * t_ease

        x1 = int(max(cx - crop_w / 2, 0))
        y1 = int(max(cy - crop_h / 2, 0))
        x2 = min(x1 + crop_w, w)
        y2 = min(y1 + crop_h, h)

        # Keep crop in bounds
        if x2 - x1 < crop_w:
            x1 = max(x2 - crop_w, 0)
        if y2 - y1 < crop_h:
            y1 = max(y2 - crop_h, 0)

        cropped = image[y1:y2, x1:x2]
        frame = cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LANCZOS4)
        frames.append(frame)

    return frames


def add_vignette(frame: np.ndarray, strength: float = 0.5) -> np.ndarray:
    """Apply a subtle dark vignette to the edges of a frame."""
    h, w = frame.shape[:2]
    sigma = min(h, w) * 0.6
    cx, cy = w / 2, h / 2
    Y, X = np.ogrid[:h, :w]
    dist = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)
    mask = np.clip(1.0 - strength * (dist / sigma) ** 2, 0, 1)
    mask = mask[:, :, np.newaxis].astype(np.float32)
    return np.clip(frame.astype(np.float32) * mask, 0, 255).astype(np.uint8)


def write_video(frames: list[np.ndarray], output: str, fps: int) -> None:
    h, w = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output, fourcc, fps, (w, h))
    for frame in frames:
        writer.write(frame)
    writer.release()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create an animated video from a still image."
    )
    parser.add_argument("image", help="Input image path (jpg, png, …)")
    parser.add_argument("-o", "--output", default="output.mp4",
                        help="Output video path (default: output.mp4)")
    parser.add_argument("--duration", type=float, default=8.0,
                        help="Video length in seconds (default: 8)")
    parser.add_argument("--fps", type=int, default=30,
                        help="Frames per second (default: 30)")
    parser.add_argument("--zoom-start", type=float, default=1.0,
                        help="Initial zoom level (default: 1.0 = no zoom)")
    parser.add_argument("--zoom-end", type=float, default=1.35,
                        help="Final zoom level (default: 1.35)")
    parser.add_argument("--pan-x", type=float, default=0.04,
                        help="Horizontal pan fraction of width (default: 0.04)")
    parser.add_argument("--pan-y", type=float, default=0.02,
                        help="Vertical pan fraction of height (default: 0.02)")
    parser.add_argument("--vignette", type=float, default=0.45,
                        help="Vignette strength 0-1 (default: 0.45, 0 = off)")
    args = parser.parse_args()

    image = cv2.imread(args.image)
    if image is None:
        print(f"Error: cannot read image '{args.image}'")
        sys.exit(1)

    print(f"Image loaded: {image.shape[1]}x{image.shape[0]}")
    print(f"Generating {args.duration}s @ {args.fps}fps …")

    frames = ken_burns(
        image, args.duration, args.fps,
        args.zoom_start, args.zoom_end,
        args.pan_x, args.pan_y,
    )

    if args.vignette > 0:
        frames = [add_vignette(f, args.vignette) for f in frames]

    write_video(frames, args.output, args.fps)
    print(f"Video saved → {args.output}")


if __name__ == "__main__":
    main()
