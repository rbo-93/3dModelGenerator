#!/usr/bin/env python3
"""Split a 3-view or 4-view anime turnaround sheet into calibrated 4-view PNGs.

Detects whitespace column gaps, preserves a single shared vertical scale across all
views, mirrors a single side profile to synthesize the opposite 90/270-deg view,
and outputs:
  - RGB turnaround sheets on neutral white (`inputs/<char>_4view/`) for BiRefNet
  - Exterior-alpha RGBA fixtures (`eval_fixtures/<char>_4view/`) for evaluation
"""

from __future__ import annotations

import argparse
import json
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageOps


def extract_exterior_fg_mask(rgb_or_rgba: np.ndarray, color_tol: float = 22.0) -> np.ndarray:
    """Flood-fill background from image borders to extract a clean foreground boolean mask."""
    arr = np.asarray(rgb_or_rgba)
    if arr.ndim == 3 and arr.shape[2] == 4 and np.any(arr[:, :, 3] < 250):
        return arr[:, :, 3] > 128

    h, w = arr.shape[:2]
    rgb = arr[:, :, :3].astype(np.float32)
    corners = np.concatenate(
        [
            rgb[:8, :8].reshape(-1, 3),
            rgb[:8, -8:].reshape(-1, 3),
            rgb[-8:, :8].reshape(-1, 3),
            rgb[-8:, -8:].reshape(-1, 3),
        ],
        axis=0,
    )
    bg_col = np.median(corners, axis=0)
    is_bg_cand = np.linalg.norm(rgb - bg_col[None, None, :], axis=-1) <= color_tol

    bg = np.zeros((h, w), dtype=bool)
    q: deque[tuple[int, int]] = deque()
    for x in range(w):
        for y in (0, h - 1):
            if is_bg_cand[y, x] and not bg[y, x]:
                bg[y, x] = True
                q.append((y, x))
    for y in range(1, h - 1):
        for x in (0, w - 1):
            if is_bg_cand[y, x] and not bg[y, x]:
                bg[y, x] = True
                q.append((y, x))

    while q:
        cy, cx = q.popleft()
        for ny, nx in ((cy - 1, cx), (cy + 1, cx), (cy, cx - 1), (cy, cx + 1)):
            if 0 <= ny < h and 0 <= nx < w and not bg[ny, nx] and is_bg_cand[ny, nx]:
                bg[ny, nx] = True
                q.append((ny, nx))
    return ~bg


def _find_column_splits(fg_mask: np.ndarray, expected_views: int | None = None) -> list[tuple[int, int]]:
    """Find contiguous horizontal column intervals corresponding to individual character views."""
    h, w = fg_mask.shape
    col_fg = fg_mask.sum(axis=0) > max(4, int(0.005 * h))
    active = np.where(col_fg)[0]
    if len(active) == 0:
        raise ValueError("No foreground pixels detected in turnaround sheet")

    # Merge tiny horizontal gaps (<= 8 px) within a single view (e.g. disconnected wing tips)
    runs: list[tuple[int, int]] = []
    start = int(active[0])
    prev = int(active[0])
    for x in active[1:]:
        xi = int(x)
        if xi - prev > 10:
            runs.append((start, prev + 1))
            start = xi
        prev = xi
    runs.append((start, prev + 1))

    # Filter tiny dust speckles
    runs = [r for r in runs if (r[1] - r[0]) >= max(24, int(0.04 * w))]
    if expected_views is not None and len(runs) != expected_views:
        # Try bridging larger internal gaps if over-split
        while len(runs) > expected_views:
            gaps = [(runs[i + 1][0] - runs[i][1], i) for i in range(len(runs) - 1)]
            _, min_idx = min(gaps)
            merged = (runs[min_idx][0], runs[min_idx + 1][1])
            runs = runs[:min_idx] + [merged] + runs[min_idx + 2 :]
    return runs


def _estimate_spine_x(fg: np.ndarray) -> float:
    """Estimate vertical spine X inside a cropped view from crown+ankle bands (robust to trailing hair/wings)."""
    ys, xs = np.where(fg)
    if len(ys) == 0:
        return fg.shape[1] * 0.5
    y0, y1 = int(ys.min()), int(ys.max())
    span = max(10, y1 - y0)
    h = fg.shape[0]
    band = np.zeros_like(fg)
    band[max(0, y0 + int(0.06 * span)) : min(h, y0 + int(0.18 * span)), :] = True
    band[max(0, y1 - int(0.12 * span)) : min(h, y1 + 1), :] = True
    band_xs = np.where(fg & band)[1]
    if len(band_xs) >= 16:
        return float(np.median(band_xs))
    return float(0.5 * (xs.min() + xs.max()))


def _detect_side_facing_role(side_fg: np.ndarray) -> str:
    """Return 'right' (90 deg, nose pointing left) or 'left' (270 deg, nose pointing right)."""
    ys, xs = np.where(side_fg)
    if len(ys) == 0:
        return "right"
    y0, y1 = int(ys.min()), int(ys.max())
    span = max(10, y1 - y0)
    # Compare upper-back hair/wings mass vs lower-foot toe protrusion direction:
    # In anime side profiles, toes point toward the nose while long hair trails opposite.
    foot_band = side_fg[max(0, y1 - int(0.06 * span)) : y1 + 1, :]
    ankle_band = side_fg[max(0, y1 - int(0.14 * span)) : max(1, y1 - int(0.06 * span)), :]
    f_xs = np.where(foot_band)[1]
    a_xs = np.where(ankle_band)[1]
    if len(f_xs) > 0 and len(a_xs) > 0:
        ankle_cx = float(np.median(a_xs))
        left_reach = ankle_cx - float(f_xs.min())
        right_reach = float(f_xs.max()) - ankle_cx
        # Nose points toward the longer forward toe reach
        return "right" if left_reach >= right_reach else "left"
    return "right"


def split_turnaround_sheet(
    sheet_path: Path,
    *,
    out_inputs_dir: Path,
    out_fixture_dir: Path | None = None,
    char_id: str = "character",
    target_w: int = 1024,
    target_h: int = 1536,
    fill_frac: float = 0.88,
    side_role: str = "auto",
) -> dict[str, Any]:
    """Split a turnaround sheet into shared-scale 4 orthogonal views."""
    im = Image.open(sheet_path).convert("RGB")
    arr_rgb = np.asarray(im, dtype=np.uint8)
    fg_all = extract_exterior_fg_mask(arr_rgb, color_tol=24.0)
    runs = _find_column_splits(fg_all)
    if len(runs) not in (3, 4):
        raise ValueError(f"Expected 3 or 4 character columns in {sheet_path}, detected {len(runs)}: {runs}")

    # Pad each column split slightly into surrounding whitespace
    h_src, w_src = arr_rgb.shape[:2]
    col_crops: list[tuple[np.ndarray, np.ndarray]] = []
    global_y0, global_y1 = h_src, 0
    for idx, (rx0, rx1) in enumerate(runs):
        left_bound = 0 if idx == 0 else (runs[idx - 1][1] + rx0) // 2
        right_bound = w_src if idx == len(runs) - 1 else (rx1 + runs[idx + 1][0]) // 2
        crop_rgb = arr_rgb[:, left_bound:right_bound]
        crop_fg = fg_all[:, left_bound:right_bound]
        ys = np.where(crop_fg)[0]
        if len(ys):
            global_y0 = min(global_y0, int(ys.min()))
            global_y1 = max(global_y1, int(ys.max()) + 1)
        col_crops.append((crop_rgb, crop_fg))

    if global_y1 <= global_y0:
        raise ValueError("Failed to compute valid foreground vertical bounds")

    shared_h = global_y1 - global_y0
    max_w = max(int(np.where(fg)[1].max() - np.where(fg)[1].min() + 1) for _, fg in col_crops if np.any(fg))
    scale = min((target_h * fill_frac) / float(shared_h), (target_w * 0.92) / float(max_w))

    def _place_on_canvas(crop_rgb: np.ndarray, crop_fg: np.ndarray, *, is_side: bool) -> tuple[Image.Image, Image.Image]:
        ys, xs = np.where(crop_fg)
        x0, x1 = int(xs.min()), int(xs.max()) + 1
        sub_rgb = crop_rgb[global_y0:global_y1, x0:x1]
        sub_fg = crop_fg[global_y0:global_y1, x0:x1]

        # Clean exterior background to pure white in RGB, transparent in RGBA
        rgb_clean = np.where(sub_fg[..., None], sub_rgb, np.uint8(255))
        rgba_clean = np.dstack([rgb_clean, np.where(sub_fg, np.uint8(255), np.uint8(0))])

        ch, cw = sub_rgb.shape[:2]
        nw = max(1, int(round(cw * scale)))
        nh = max(1, int(round(ch * scale)))

        rgb_res = Image.fromarray(rgb_clean, mode="RGB").resize((nw, nh), Image.Resampling.LANCZOS)
        rgba_res = Image.fromarray(rgba_clean, mode="RGBA").resize((nw, nh), Image.Resampling.LANCZOS)

        spine_local_x = _estimate_spine_x(sub_fg) if is_side else (0.5 * cw)
        spine_scaled_x = int(round(spine_local_x * scale))
        ox = int(round(target_w * 0.5 - spine_scaled_x))
        ox = max(0, min(target_w - nw, ox))
        oy = (target_h - nh) // 2

        canvas_rgb = Image.new("RGB", (target_w, target_h), (255, 255, 255))
        canvas_rgb.paste(rgb_res, (ox, oy))

        canvas_rgba = Image.new("RGBA", (target_w, target_h), (0, 0, 0, 0))
        canvas_rgba.paste(rgba_res, (ox, oy))
        return canvas_rgb, canvas_rgba

    views_rgb: dict[str, Image.Image] = {}
    views_rgba: dict[str, Image.Image] = {}

    if len(col_crops) == 3:
        # Layout: [Front, Side, Back]
        views_rgb["view_00_front.png"], views_rgba["view_00_front.png"] = _place_on_canvas(*col_crops[0], is_side=False)
        views_rgb["view_02_back.png"], views_rgba["view_02_back.png"] = _place_on_canvas(*col_crops[2], is_side=False)

        eff_side = _detect_side_facing_role(col_crops[1][1]) if side_role == "auto" else side_role
        side_rgb, side_rgba = _place_on_canvas(*col_crops[1], is_side=True)
        if eff_side == "right":
            views_rgb["view_01_right.png"] = side_rgb
            views_rgba["view_01_right.png"] = side_rgba
            views_rgb["view_03_left.png"] = ImageOps.mirror(side_rgb)
            views_rgba["view_03_left.png"] = ImageOps.mirror(side_rgba)
        else:
            views_rgb["view_03_left.png"] = side_rgb
            views_rgba["view_03_left.png"] = side_rgba
            views_rgb["view_01_right.png"] = ImageOps.mirror(side_rgb)
            views_rgba["view_01_right.png"] = ImageOps.mirror(side_rgba)
    else:
        # Layout: [Front, Right, Back, Left]
        for fname, crop_pair, is_s in (
            ("view_00_front.png", col_crops[0], False),
            ("view_01_right.png", col_crops[1], True),
            ("view_02_back.png", col_crops[2], False),
            ("view_03_left.png", col_crops[3], True),
        ):
            views_rgb[fname], views_rgba[fname] = _place_on_canvas(*crop_pair, is_side=is_s)
        eff_side = "4view_explicit"

    ordered = ("view_00_front.png", "view_01_right.png", "view_02_back.png", "view_03_left.png")

    def _write_view_set(dst_dir: Path, imgs: dict[str, Image.Image], *, mode: str) -> None:
        dst_dir.mkdir(parents=True, exist_ok=True)
        for fn in ordered:
            imgs[fn].save(dst_dir / fn)
        grid = Image.new("RGB", (target_w * 4, target_h), (255, 255, 255))
        for i, fn in enumerate(ordered):
            tile = imgs[fn]
            if tile.mode == "RGBA":
                bg = Image.new("RGBA", (target_w, target_h), (255, 255, 255, 255))
                tile = Image.alpha_composite(bg, tile).convert("RGB")
            grid.paste(tile, (i * target_w, 0))
        grid.save(dst_dir / "turnaround_grid.png")

    _write_view_set(out_inputs_dir, views_rgb, mode="RGB")
    if out_fixture_dir is not None:
        _write_view_set(out_fixture_dir, views_rgba, mode="RGBA")

    meta = {
        "fixture_id": f"{char_id}_4view",
        "source_sheet": str(sheet_path.resolve()),
        "detected_columns": len(col_crops),
        "column_bounds_x": runs,
        "detected_side_role": eff_side,
        "shared_vertical_span_src": [global_y0, global_y1],
        "scale_to_target": round(float(scale), 6),
        "views": list(ordered),
        "resolution": [target_w, target_h],
    }
    (out_inputs_dir / "fixture_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    if out_fixture_dir is not None:
        (out_fixture_dir / "fixture_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta


def main() -> int:
    ap = argparse.ArgumentParser(description="Split a 3-view or 4-view turnaround sheet into 4 canonical views.")
    ap.add_argument("--sheet", type=Path, required=True)
    ap.add_argument("--char-id", default="valkyrie")
    ap.add_argument("--out-inputs-dir", type=Path, required=True)
    ap.add_argument("--out-fixture-dir", type=Path, default=None)
    ap.add_argument("--side-role", choices=("auto", "right", "left"), default="auto")
    ap.add_argument("--width", type=int, default=1024)
    ap.add_argument("--height", type=int, default=1536)
    args = ap.parse_args()

    meta = split_turnaround_sheet(
        args.sheet,
        out_inputs_dir=args.out_inputs_dir,
        out_fixture_dir=args.out_fixture_dir,
        char_id=args.char_id,
        target_w=args.width,
        target_h=args.height,
        side_role=args.side_role,
    )
    print(json.dumps(meta, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
