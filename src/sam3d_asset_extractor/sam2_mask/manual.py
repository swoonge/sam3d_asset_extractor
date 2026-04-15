"""SAM2 interactive point-based mask UI.

Keybindings:
  * LMB                : add a positive point
  * Shift + LMB        : add a negative point
  * z                  : undo last point
  * r                  : reset all points / mask
  * s                  : save current mask as PNG
  * q / ESC            : quit
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import torch

from sam3d_asset_extractor.logging_setup import configure_logging, get_logger
from sam3d_asset_extractor.sam2_mask._common import (
    autocast_context,
    default_checkpoint_path,
    default_model_cfg,
    prepare_sam2_import,
    resolve_device,
    resolve_model_cfg,
)

logger = get_logger("sam2_mask.manual")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SAM2 point-based mask UI")
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--model-cfg", type=str, default=default_model_cfg())
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--window-width", type=int, default=1280)
    parser.add_argument("--window-height", type=int, default=720)
    parser.add_argument("--log-level", type=str, default="INFO")
    return parser.parse_args()


def _build_help_panel(
    lines: list[str],
    font_scale: float = 0.5,
    thickness: int = 1,
    padding: int = 10,
    line_gap: int = 6,
    bg: tuple[int, int, int] = (245, 245, 245),
    text_color: tuple[int, int, int] = (20, 20, 20),
    border: tuple[int, int, int] = (200, 200, 200),
) -> np.ndarray:
    font = cv2.FONT_HERSHEY_SIMPLEX
    if not lines:
        panel = np.full((60, 200, 3), bg, dtype=np.uint8)
        cv2.rectangle(panel, (0, 0), (199, 59), border, 1)
        return panel
    metrics = []
    max_w = 1
    total_h = 0
    for line in lines:
        (w, h), b = cv2.getTextSize(line, font, font_scale, thickness)
        metrics.append((w, h, b))
        max_w = max(max_w, w)
        total_h += h + b
    if len(lines) > 1:
        total_h += line_gap * (len(lines) - 1)
    height = padding * 2 + max(total_h, 1)
    width = padding * 2 + max_w
    panel = np.full((height, width, 3), bg, dtype=np.uint8)
    cv2.rectangle(panel, (0, 0), (width - 1, height - 1), border, 1)
    y = padding
    for line, (w, h, b) in zip(lines, metrics):
        y += h
        cv2.putText(panel, line, (padding, y), font, font_scale, text_color, thickness, cv2.LINE_AA)
        y += b + line_gap
    return panel


def _stack_with_panel_right(image: np.ndarray, panel: np.ndarray) -> np.ndarray:
    img_h, img_w = image.shape[:2]
    panel_h, panel_w = panel.shape[:2]
    height = max(img_h, panel_h)
    canvas = np.full((height, img_w + panel_w, 3), (245, 245, 245), dtype=np.uint8)
    canvas[:img_h, :img_w] = image
    canvas[:panel_h, img_w:img_w + panel_w] = panel
    return canvas


def main() -> int:
    args = parse_args()
    configure_logging(level=args.log_level)

    device = resolve_device(args.device)
    checkpoint_path = args.checkpoint or default_checkpoint_path()
    image_path = args.image.resolve()
    output_dir = args.output_dir.resolve()

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Missing SAM2 checkpoint: {checkpoint_path}")
    if not image_path.exists():
        raise FileNotFoundError(f"Missing image: {image_path}")
    output_dir.mkdir(parents=True, exist_ok=True)

    _, sam2_pkg_root = prepare_sam2_import()
    model_cfg = resolve_model_cfg(args.model_cfg, sam2_pkg_root)

    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    logger.info("building SAM2 predictor (device=%s)", device)
    model = build_sam2(model_cfg, str(checkpoint_path), device=device)
    predictor = SAM2ImagePredictor(model)

    img_bgr = cv2.imread(str(image_path))
    if img_bgr is None:
        raise FileNotFoundError(image_path)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    with torch.inference_mode(), autocast_context(device):
        predictor.set_image(img_rgb)

    points: list[tuple[int, int]] = []
    labels: list[int] = []
    last_mask: np.ndarray | None = None
    last_score: float | None = None
    status_msg = ""
    status_time = 0.0

    def run_predict() -> None:
        nonlocal last_mask, last_score
        if not points:
            last_mask = None
            last_score = None
            return
        coords = np.array(points, dtype=np.float32)
        lbls = np.array(labels, dtype=np.int32)
        with torch.inference_mode(), autocast_context(device):
            masks, scores, _ = predictor.predict(
                point_coords=coords, point_labels=lbls, multimask_output=True,
            )
        best = int(np.argmax(scores))
        last_mask = (masks[best] > 0).astype(np.uint8) * 255
        last_score = float(scores[best])

    def draw_help_panel() -> np.ndarray:
        score = "-" if last_score is None else f"{last_score:.4f}"
        lines: list[str] = []
        if status_msg and (time.time() - status_time) < 5:
            lines.append(status_msg)
        lines.extend(
            [
                f"score: {score}  points: {len(points)}",
                "LMB=positive",
                "Shift+LMB=negative",
                "z=undo",
                "r=reset",
                "s=save",
                "q/ESC=quit",
            ]
        )
        return _build_help_panel(lines)

    def draw_ui() -> np.ndarray:
        vis = img_bgr.copy()
        if last_mask is not None:
            green = np.zeros_like(vis)
            green[:, :, 1] = 255
            alpha = (last_mask > 0).astype(np.float32) * 0.35
            vis = (vis * (1 - alpha[..., None]) + green * alpha[..., None]).astype(np.uint8)
        for (x, y), lab in zip(points, labels):
            color = (0, 255, 0) if lab == 1 else (0, 0, 255)
            cv2.circle(vis, (int(x), int(y)), 5, color, -1)
        return _stack_with_panel_right(vis, draw_help_panel())

    def on_mouse(event: int, x: int, y: int, flags: int, param) -> None:
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append((x, y))
            labels.append(0 if (flags & cv2.EVENT_FLAG_SHIFTKEY) else 1)
            run_predict()

    cv2.namedWindow("sam2_point_ui", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("sam2_point_ui", args.window_width, args.window_height)
    cv2.setMouseCallback("sam2_point_ui", on_mouse)
    run_predict()

    save_index = 0
    while True:
        cv2.imshow("sam2_point_ui", draw_ui())
        k = cv2.waitKey(10) & 0xFF
        if k == ord("z"):
            if points:
                points.pop()
                labels.pop()
                run_predict()
        elif k == ord("r"):
            points.clear()
            labels.clear()
            last_mask = None
            last_score = None
        elif k == ord("s"):
            if last_mask is None:
                status_msg = "no mask to save"
                status_time = time.time()
            else:
                out_mask = output_dir / f"{image_path.stem}_{save_index:03d}.png"
                cv2.imwrite(str(out_mask), last_mask)
                status_msg = f"saved: {out_mask.name}"
                status_time = time.time()
                save_index += 1
        elif k in (ord("q"), 27):
            break

    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
