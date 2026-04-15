"""SAM2 automatic mask generation.

Runs inside the ``sam2`` conda env (as a subprocess spawned by the CLI
orchestrator, or directly via ``python -m``). Produces one PNG per generated
mask under ``--output-dir`` with naming ``<image_stem>_<idx>.png``.

Filtering pipeline (applied in order when ``--apply-extra-filtering`` is on):
  1. per-mask keep largest connected component
  2. dedupe via IoU
  3. drop masks touching top/bottom image borders
  4. if depth available: drop nested masks that sit on the same depth surface
"""

from __future__ import annotations

import argparse
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

logger = get_logger("sam2_mask.auto")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SAM2 automatic mask generation")
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--depth-image", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--model-cfg", type=str, default=default_model_cfg())
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--points-per-side", type=int, default=32)
    parser.add_argument("--pred-iou-thresh", type=float, default=0.8)
    parser.add_argument("--stability-score-thresh", type=float, default=0.9)
    parser.add_argument("--min-mask-region-area", type=int, default=500)
    parser.add_argument(
        "--depth-same-surface-thresh", type=float, default=None,
        help="Depth delta threshold for nested-surface filtering (auto-inferred if omitted).",
    )
    parser.add_argument("--dedupe-iou-thresh", type=float, default=0.85)
    parser.add_argument("--nested-containment-thresh", type=float, default=0.98)
    parser.add_argument("--depth-connected-ratio-thresh", type=float, default=0.9)
    parser.add_argument(
        "--keep-largest-component",
        dest="keep_largest_component",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--filter-border",
        dest="filter_border",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--apply-extra-filtering",
        dest="apply_extra_filtering",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--clean-output",
        dest="clean_output",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--log-level", type=str, default="INFO")
    return parser.parse_args()


def clear_output_files(output_dir: Path) -> None:
    if not output_dir.exists():
        return
    for child in output_dir.iterdir():
        if child.is_file():
            child.unlink()


def infer_depth_surface_threshold(depth_map: np.ndarray) -> tuple[float, float]:
    """Pick a reasonable same-surface depth threshold from the depth distribution."""
    valid_depth = depth_map[np.isfinite(depth_map) & (depth_map > 0)]
    if valid_depth.size == 0:
        raise ValueError("No valid depth values found.")
    depth_median = float(np.median(valid_depth))
    thresh = 20.0 if depth_median > 10.0 else 0.02
    return depth_median, thresh


def _mask_pixel_values(mask: np.ndarray, depth_map: np.ndarray) -> np.ndarray:
    vals = depth_map[mask]
    return vals[np.isfinite(vals) & (vals > 0)]


def compute_mask_info(mask: np.ndarray, depth_map: np.ndarray | None) -> dict | None:
    ys, xs = np.where(mask)
    if ys.size == 0:
        return None
    info = {
        "mask": mask,
        "area": int(mask.sum()),
        "y_min": int(ys.min()),
        "y_max": int(ys.max()),
        "x_min": int(xs.min()),
        "x_max": int(xs.max()),
    }
    if depth_map is not None:
        vals = _mask_pixel_values(mask, depth_map)
        if vals.size == 0:
            return None
        info.update(
            depth_median=float(np.median(vals)),
            depth_mean=float(np.mean(vals)),
            depth_std=float(np.std(vals)),
        )
    return info


def is_border_mask(info: dict, height: int) -> bool:
    return info["y_min"] == 0 or info["y_max"] == height - 1


def keep_largest_component(mask: np.ndarray) -> np.ndarray:
    mask_uint8 = mask.astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_uint8, connectivity=8)
    if num_labels <= 1:
        return mask
    areas = stats[1:, cv2.CC_STAT_AREA]
    largest_label = 1 + int(np.argmax(areas))
    return labels == largest_label


def mask_iou(mask1: np.ndarray, mask2: np.ndarray) -> float:
    inter = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    if union == 0:
        return 0.0
    return float(inter / union)


def remove_duplicate_masks(mask_infos: list[dict], iou_thresh: float) -> list[dict]:
    keep = [True] * len(mask_infos)
    for i in range(len(mask_infos)):
        if not keep[i]:
            continue
        for j in range(i + 1, len(mask_infos)):
            if not keep[j]:
                continue
            iou = mask_iou(mask_infos[i]["mask"], mask_infos[j]["mask"])
            if iou <= iou_thresh:
                continue
            if mask_infos[i]["area"] >= mask_infos[j]["area"]:
                keep[j] = False
            else:
                keep[i] = False
                break
    return [m for k, m in zip(keep, mask_infos) if k]


def bbox_contains(small: dict, large: dict) -> bool:
    return (
        small["x_min"] >= large["x_min"]
        and small["x_max"] <= large["x_max"]
        and small["y_min"] >= large["y_min"]
        and small["y_max"] <= large["y_max"]
    )


def mask_containment_ratio(small_mask: np.ndarray, large_mask: np.ndarray) -> float:
    inter = np.logical_and(small_mask, large_mask).sum()
    area_small = small_mask.sum()
    if area_small == 0:
        return 0.0
    return float(inter / area_small)


def depth_surface_connected(
    small_mask: np.ndarray,
    large_mask: np.ndarray,
    depth_map: np.ndarray,
    depth_thresh: float,
    connected_ratio_thresh: float,
) -> bool:
    ys, xs = np.where(small_mask)
    total = len(xs)
    if total == 0:
        return False
    connected = 0
    for y, x in zip(ys, xs):
        d = depth_map[y, x]
        if not np.isfinite(d) or d <= 0:
            continue
        found = False
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                ny, nx = y + dy, x + dx
                if ny < 0 or ny >= depth_map.shape[0]:
                    continue
                if nx < 0 or nx >= depth_map.shape[1]:
                    continue
                if not large_mask[ny, nx]:
                    continue
                d2 = depth_map[ny, nx]
                if np.isfinite(d2) and abs(d - d2) < depth_thresh:
                    found = True
                    break
            if found:
                break
        if found:
            connected += 1
    return (connected / total) > connected_ratio_thresh


def remove_nested_same_depth_masks(
    mask_infos: list[dict],
    depth_map: np.ndarray,
    depth_thresh: float,
    containment_thresh: float,
    connected_ratio_thresh: float,
) -> list[dict]:
    keep = [True] * len(mask_infos)
    order = np.argsort([m["area"] for m in mask_infos])
    for oi in order:
        if not keep[oi]:
            continue
        small = mask_infos[oi]
        for oj in range(len(mask_infos)):
            if oi == oj or not keep[oj]:
                continue
            large = mask_infos[oj]
            if large["area"] <= small["area"] * 1.2:
                continue
            if not bbox_contains(small, large):
                continue
            if mask_containment_ratio(small["mask"], large["mask"]) < containment_thresh:
                continue
            if depth_surface_connected(
                small["mask"], large["mask"], depth_map, depth_thresh, connected_ratio_thresh
            ):
                logger.debug(
                    "dropping nested mask (area=%d inside area=%d)",
                    small["area"], large["area"],
                )
                keep[oi] = False
                break
    return [m for k, m in zip(keep, mask_infos) if k]


def main() -> int:
    args = parse_args()
    configure_logging(level=args.log_level)

    device = resolve_device(args.device)
    checkpoint_path = args.checkpoint or default_checkpoint_path()
    image_path = args.image.resolve()
    depth_path = args.depth_image.resolve() if args.depth_image is not None else None
    output_dir = args.output_dir.resolve()

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Missing SAM2 checkpoint: {checkpoint_path}")
    if not image_path.exists():
        raise FileNotFoundError(f"Missing image: {image_path}")
    if depth_path is not None and not depth_path.exists():
        raise FileNotFoundError(f"Missing depth image: {depth_path}")

    output_dir.mkdir(parents=True, exist_ok=True)
    if args.clean_output:
        clear_output_files(output_dir)

    _, sam2_pkg_root = prepare_sam2_import()
    model_cfg = resolve_model_cfg(args.model_cfg, sam2_pkg_root)

    from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
    from sam2.build_sam import build_sam2

    img_bgr = cv2.imread(str(image_path))
    if img_bgr is None:
        raise FileNotFoundError(image_path)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    image_h = img_rgb.shape[0]

    depth_map = None
    depth_thresh = None
    if depth_path is not None:
        depth_raw = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
        if depth_raw is None:
            raise FileNotFoundError(depth_path)
        depth_map = depth_raw.astype(np.float32)
        if args.depth_same_surface_thresh is None:
            median, depth_thresh = infer_depth_surface_threshold(depth_map)
            logger.info("depth median=%.4f, same-surface thresh=%.4f", median, depth_thresh)
        else:
            depth_thresh = float(args.depth_same_surface_thresh)

    logger.info("building SAM2 (device=%s, cfg=%s)", device, model_cfg)
    model = build_sam2(model_cfg, str(checkpoint_path), device=device)
    generator = SAM2AutomaticMaskGenerator(
        model,
        points_per_side=args.points_per_side,
        pred_iou_thresh=args.pred_iou_thresh,
        stability_score_thresh=args.stability_score_thresh,
        min_mask_region_area=args.min_mask_region_area,
    )

    logger.info("generating masks...")
    with torch.inference_mode(), autocast_context(device):
        raw = generator.generate(img_rgb)
    logger.info("raw masks: %d", len(raw))

    if args.apply_extra_filtering:
        mask_infos = []
        for item in raw:
            mask = np.asarray(item["segmentation"], dtype=bool)
            if args.keep_largest_component:
                mask = keep_largest_component(mask)
            info = compute_mask_info(mask, depth_map)
            if info is None:
                continue
            mask_infos.append(info)
        logger.info("after preprocess: %d", len(mask_infos))
        mask_infos = remove_duplicate_masks(mask_infos, iou_thresh=args.dedupe_iou_thresh)
        logger.info("after dedupe: %d", len(mask_infos))
        if args.filter_border:
            mask_infos = [m for m in mask_infos if not is_border_mask(m, image_h)]
            logger.info("after border filter: %d", len(mask_infos))
        if depth_map is not None:
            mask_infos = remove_nested_same_depth_masks(
                mask_infos,
                depth_map=depth_map,
                depth_thresh=float(depth_thresh),
                containment_thresh=args.nested_containment_thresh,
                connected_ratio_thresh=args.depth_connected_ratio_thresh,
            )
            logger.info("after nested-depth filter: %d", len(mask_infos))
        final_masks = [m["mask"] for m in mask_infos]
    else:
        final_masks = [np.asarray(m["segmentation"], dtype=bool) for m in raw]

    # Overlay visualization (filename is outside the mask glob pattern).
    vis = img_bgr.copy()
    rng = np.random.default_rng(args.seed)
    for mask in final_masks:
        color = rng.integers(0, 255, size=3, dtype=np.uint8)
        overlay = vis[mask].astype(np.float32) * 0.5 + color.astype(np.float32) * 0.5
        vis[mask] = overlay.astype(np.uint8)
    vis_path = output_dir / f"vis_{image_path.stem}.png"
    cv2.imwrite(str(vis_path), vis)
    logger.info("saved overlay: %s", vis_path)

    for i, mask in enumerate(final_masks):
        mask_u8 = np.asarray(mask, dtype=np.uint8) * 255
        mask_path = output_dir / f"{image_path.stem}_{i:03d}.png"
        cv2.imwrite(str(mask_path), mask_u8)
        logger.info("saved mask: %s", mask_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
