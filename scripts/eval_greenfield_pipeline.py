#!/usr/bin/env python3
"""Greenfield Stage-by-Stage Evaluator & Gate Verifier.

Runs all local visual & quantitative verification gates on the final Stage 6 GLB:
  1. 4-view orthographic turnaround renders (`0, 90, 180, 270 deg`) + Silhouette IoU & Delta-E76
  2. `walk_00`, `walk_15`, `run_15` 3/4 Stage B keyframe preview (`clean_cel` + `exr_4b`)
  3. Stage B strict hygiene gate (`prepare_ai_character.py --strict`)
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

REPO_ROOT = Path(__file__).resolve().parents[1]
STAGE_B_ROOT = REPO_ROOT.parent / "2d_test_animate"
STAGE_B_SRC = STAGE_B_ROOT / "src"
DEFAULT_PREP_SCRIPT = STAGE_B_ROOT / "scripts" / "prepare_ai_character.py"
DEFAULT_WALK_FBX = STAGE_B_ROOT / "assets" / "inputs" / "3d_model" / "genshin-impact-lauma" / "lauma_walk.fbx"
DEFAULT_RUN_FBX = STAGE_B_ROOT / "assets" / "inputs" / "3d_model" / "genshin-impact-lauma" / "lauma_run.fbx"
DEFAULT_BLENDER_EXE = Path(
    os.environ.get("BLENDER_EXE", r"C:\Program Files (x86)\Steam\steamapps\common\Blender\blender.exe")
)

VIEW_AZIMUTHS: tuple[tuple[str, float], ...] = (
    ("view_00_front.png", 0.0),
    ("view_01_right.png", 90.0),
    ("view_02_back.png", 180.0),
    ("view_03_left.png", 270.0),
)

_BLENDER_ORTHO_SCRIPT = r"""
import json, math, sys
from pathlib import Path
import bpy
from mathutils import Vector

argv = sys.argv[sys.argv.index("--") + 1 :]
model_path = Path(argv[0]).resolve()
out_dir = Path(argv[1]).resolve()
stage_b_src = argv[2]
width = int(argv[3])
height = int(argv[4])

sys.path.insert(0, stage_b_src)
from sprite_pipeline.infrastructure import blender_ingest, blender_renderer

out_dir.mkdir(parents=True, exist_ok=True)
blender_ingest.import_model_clean(model_path, reset_scene=True)
disc = blender_ingest.discover_renderable_objects()
arm_obj = disc["primary_armature"]
meshes = disc["meshes"]
if not meshes:
    raise RuntimeError(f"No renderable meshes in {model_path}")

blender_ingest.sanitize_corrupted_custom_normals(meshes)
blender_ingest.analyze_and_clean_coplanar(meshes)
blender_ingest.auto_orient_scale_and_ground(arm_obj, meshes, target_height=1.75)

ir_list = blender_renderer.extract_material_ir()
blender_renderer.sanitize_vrm_unlit_ir(ir_list)
blender_renderer.apply_render_profile(ir_list, "deadcells_dualpass", char_height=1.75, emit_aovs=False)

deps = bpy.context.evaluated_depsgraph_get()
all_pts = []
part_stats = []
for obj in meshes:
    ev = obj.evaluated_get(deps)
    mw, me = ev.matrix_world, ev.data
    all_pts.extend([mw @ v.co for v in me.vertices])
    uv_area = 0.0
    if me.uv_layers and len(me.polygons) > 0:
        uv_d = me.uv_layers.active.data
        for poly in me.polygons:
            ls = poly.loop_indices
            if len(ls) >= 3:
                u0, v0 = float(uv_d[ls[0]].uv.x), float(uv_d[ls[0]].uv.y)
                for k in range(1, len(ls) - 1):
                    u1, v1 = float(uv_d[ls[k]].uv.x), float(uv_d[ls[k]].uv.y)
                    u2, v2 = float(uv_d[ls[k + 1]].uv.x), float(uv_d[ls[k + 1]].uv.y)
                    uv_area += 0.5 * abs((u1 - u0) * (v2 - v0) - (u2 - u0) * (v1 - v0))
    part_stats.append({"name": obj.name, "verts": len(me.vertices), "faces": len(me.polygons), "uv_area": round(float(uv_area), 6)})

xs, ys, zs = [p.x for p in all_pts], [p.y for p in all_pts], [p.z for p in all_pts]
z_min, z_max = float(min(zs)), float(max(zs))
x_span, y_span, z_span = float(max(xs) - min(xs)), float(max(ys) - min(ys)), max(0.1, z_max - z_min)
center_x, center_y, center_z = 0.5 * float(min(xs) + max(xs)), 0.5 * float(min(ys) + max(ys)), 0.5 * (z_min + z_max)
ortho_scale = max(z_span * 1.12, max(x_span, y_span) * 1.12 / max(float(width) / float(height), 1e-4))

scene = bpy.context.scene
scene.render.resolution_x = width
scene.render.resolution_y = height
scene.render.resolution_percentage = 100
scene.render.film_transparent = True
scene.render.image_settings.file_format = "PNG"
scene.render.image_settings.color_mode = "RGBA"

cam_data = bpy.data.cameras.new("EvalOrthoCam")
cam_data.type = "ORTHO"
cam_data.sensor_fit = "VERTICAL"
cam_data.ortho_scale = ortho_scale
cam_data.clip_start, cam_data.clip_end = 0.05, 50.0
cam_obj = bpy.data.objects.new("EvalOrthoCam", cam_data)
scene.collection.objects.link(cam_obj)
scene.camera = cam_obj

target = Vector((center_x, center_y, center_z))
for fname, az_deg in (("view_00_front.png", 0.0), ("view_01_right.png", 90.0), ("view_02_back.png", 180.0), ("view_03_left.png", 270.0)):
    rad = math.radians(az_deg)
    cam_pos = Vector((center_x + 4.0 * math.sin(rad), center_y - 4.0 * math.cos(rad), center_z))
    cam_obj.location = cam_pos
    cam_obj.rotation_euler = (target - cam_pos).normalized().to_track_quat("-Z", "Y").to_euler()
    scene.render.filepath = str(out_dir / fname)
    bpy.ops.render.render(write_still=True)

(out_dir / "render_meta.json").write_text(json.dumps({
    "model_path": str(model_path),
    "resolution": [width, height],
    "ortho_scale": round(ortho_scale, 5),
    "center": [round(center_x, 5), round(center_y, 5), round(center_z, 5)],
    "parts": part_stats,
}, indent=2), encoding="utf-8")
"""

_BLENDER_KEYFRAME_SCRIPT = r'''
import json, sys
from pathlib import Path
import bpy, numpy as np
from mathutils import Vector

spec = json.loads(Path(sys.argv[sys.argv.index("--") + 1]).read_text(encoding="utf-8"))
sys.path.insert(0, spec["stage_b_src"])

from sprite_pipeline.config import Rig3DConfig
from sprite_pipeline.domain import material_ir
from sprite_pipeline.infrastructure import blender_ingest, blender_rig
from sprite_pipeline.infrastructure.blender_renderer import (
    _collect_swept_vertices,
    _configure_multilayer_exr_output,
    _configure_swept_ortho_camera,
    apply_render_profile,
    extract_material_ir,
    sanitize_vrm_unlit_ir,
    setup_head_proxy_and_shadow_protection,
    setup_studio_environment,
)

eff_cfg = Rig3DConfig.for_preset("512x768", render_profile="clean_cel", outline_mode="exr_4b", use_freestyle=False)
profile_name = material_ir.normalize_profile_name(eff_cfg.render_profile)
n_frames = 16
yaw_deg = float(eff_cfg.cam_theta_deg - 90.0)
pitch_deg = float(eff_cfg.cam_pitch_deg)

sampled_clips = {}
for cyc, fk in (("walk", "walk_fbx"), ("run", "run_fbx")):
    fp = Path(spec[fk]) if spec.get(fk) else None
    if fp and fp.is_file():
        sampled_clips[cyc] = blender_rig.sample_animation_clip(fp, n_samples=n_frames)

ingest = blender_ingest.ingest_universal(Path(spec["model_path"]), target_height=1.75)
arm_obj, mesh_objs = ingest["primary_armature"], ingest["meshes"]
tgt_bones, tgt_map, rig_meta = blender_rig.map_rig_universal(arm_obj, mesh_objs)
arm_obj.data.pose_position = "REST"
bpy.context.view_layer.update()

ir_list = extract_material_ir()
sanitize_vrm_unlit_ir(ir_list)
pts_bb = [mobj.matrix_world @ Vector(c) for mobj in mesh_objs for c in mobj.bound_box]
cmin = Vector((min(p.x for p in pts_bb), min(p.y for p in pts_bb), min(p.z for p in pts_bb)))
cmax = Vector((max(p.x for p in pts_bb), max(p.y for p in pts_bb), max(p.z for p in pts_bb)))
ch_h = max(0.01, cmax.z - cmin.z)
ccenter = (cmin + cmax) * 0.5

apply_render_profile(ir_list, profile_name, ch_h, emit_aovs=eff_cfg.use_multilayer_exr)
cam_obj, key_light = setup_studio_environment(cmin, cmax, ch_h, ccenter, profile_name)
setup_head_proxy_and_shadow_protection(
    key_light_obj=key_light,
    enable_proxy_normals=eff_cfg.enable_head_proxy_normals,
    enable_shadow_protection=eff_cfg.enable_face_shadow_protection,
    mix_factor=eff_cfg.head_proxy_blend,
)

arm_obj.data.pose_position = "POSE"
bpy.context.view_layer.update()

scene = bpy.context.scene
scene.render.resolution_x, scene.render.resolution_y = eff_cfg.render_w, eff_cfg.render_h
scene.render.resolution_percentage = 100
scene.render.film_transparent = True
scene.render.use_freestyle = False
_configure_multilayer_exr_output(scene, eff_cfg)

hips_bn = rig_meta.get("pelvis_root_carrier") or tgt_map.canonical_to_source.get("hips")
actions_by_cycle, swept_chunks, hips_meds = {}, [], []
for cyc_name, is_run_flag in (("walk", False), ("run", True)):
    rt = blender_rig.retarget_and_bake_cycle(
        arm_obj, mesh_objs, tgt_bones, tgt_map, rig_meta,
        sampled_clips[cyc_name], is_run=is_run_flag, clip_name=cyc_name.capitalize(), n_cycle_samples=n_frames,
    )
    actions_by_cycle[cyc_name] = rt["action"]
    pts, hmed, _ = _collect_swept_vertices(mesh_objs, arm_obj, hips_bn, list(range(1, n_frames + 1)))
    swept_chunks.append(pts)
    hips_meds.append(hmed)

cam_info = _configure_swept_ortho_camera(
    cam_obj, yaw_deg=yaw_deg, pitch_deg=pitch_deg,
    swept_pts=np.vstack(swept_chunks), hips_median=np.mean(np.array(hips_meds), axis=0),
    margin_frac=eff_cfg.frame_margin_frac, res_x=eff_cfg.render_w, res_y=eff_cfg.render_h,
)

exr_dir = Path(spec["exr_dir"])
exr_dir.mkdir(parents=True, exist_ok=True)
rendered = []
for cyc_name, f_idx in (("walk", 0), ("walk", 15), ("run", 15)):
    arm_obj.animation_data.action = actions_by_cycle[cyc_name]
    scene.frame_set((f_idx % n_frames) + 1)
    bpy.context.view_layer.update()
    out_p = exr_dir / f"{cyc_name}_{f_idx:02d}.exr"
    scene.render.filepath = str(out_p)
    bpy.ops.render.render(write_still=True)
    rendered.append({"cycle": cyc_name, "frame": f_idx, "raw_path": str(out_p)})

Path(spec["blender_manifest"]).write_text(json.dumps({"rendered": rendered, "cam_info": cam_info}, indent=2), encoding="utf-8")
'''


def _save_turnaround_grid(view_paths: list[Path], out_grid: Path) -> None:
    imgs = [Image.open(p).convert("RGBA") for p in view_paths if p.is_file()]
    if not imgs:
        return
    w, h = imgs[0].size
    grid = Image.new("RGB", (w * len(imgs), h), (255, 255, 255))
    for i, im in enumerate(imgs):
        bg = Image.new("RGBA", (w, h), (255, 255, 255, 255))
        grid.paste(Image.alpha_composite(bg, im).convert("RGB"), (i * w, 0))
    grid.save(out_grid)


def compute_view_pair_metrics(ref_path: Path, cand_path: Path) -> dict[str, float]:
    ref_rgba = np.asarray(Image.open(ref_path).convert("RGBA"))
    cand_rgba = np.asarray(Image.open(cand_path).convert("RGBA"))
    if ref_rgba.shape[:2] != cand_rgba.shape[:2]:
        cand_rgba = np.asarray(Image.fromarray(cand_rgba, mode="RGBA").resize((ref_rgba.shape[1], ref_rgba.shape[0]), Image.Resampling.LANCZOS))
    ref_fg = ref_rgba[:, :, 3] > 128
    cand_fg = cand_rgba[:, :, 3] > 128
    raw_iou = float(np.sum(ref_fg & cand_fg) / max(1, int(np.sum(ref_fg | cand_fg))))

    ry, rx = np.where(ref_fg)
    cy, cx = np.where(cand_fg)
    aligned_iou, delta_e_med, delta_e_mean = 0.0, 999.0, 999.0
    if len(ry) and len(cy):
        rw, rh = int(rx.max() - rx.min() + 1), int(ry.max() - ry.min() + 1)
        cand_crop = cand_rgba[int(cy.min()) : int(cy.max()) + 1, int(cx.min()) : int(cx.max()) + 1]
        cand_fit = np.asarray(Image.fromarray(cand_crop, mode="RGBA").resize((rw, rh), Image.Resampling.BILINEAR))
        aligned_canvas = np.zeros_like(ref_rgba)
        aligned_canvas[int(ry.min()) : int(ry.min()) + rh, int(rx.min()) : int(rx.min()) + rw] = cand_fit
        aligned_fg = aligned_canvas[:, :, 3] > 128
        aligned_iou = float(np.sum(ref_fg & aligned_fg) / max(1, int(np.sum(ref_fg | aligned_fg))))
        shared = ref_fg & aligned_fg
        if np.any(shared):
            ref_lab = cv2.cvtColor(ref_rgba[:, :, :3], cv2.COLOR_RGB2LAB).astype(np.float32)
            cand_lab = cv2.cvtColor(aligned_canvas[:, :, :3], cv2.COLOR_RGB2LAB).astype(np.float32)
            d_lab = ref_lab[shared] - cand_lab[shared]
            de76 = np.sqrt((d_lab[:, 0] * (100.0 / 255.0)) ** 2 + d_lab[:, 1] ** 2 + d_lab[:, 2] ** 2)
            delta_e_med = float(np.median(de76))
            delta_e_mean = float(np.mean(de76))

    return {
        "raw_silhouette_iou": round(raw_iou, 4),
        "bbox_aligned_silhouette_iou": round(aligned_iou, 4),
        "delta_e76_median": round(delta_e_med, 3),
        "delta_e76_mean": round(delta_e_mean, 3),
    }


def render_model_ortho_views(model_path: Path, out_dir: Path, *, blender_exe: Path = DEFAULT_BLENDER_EXE, width: int = 1024, height: int = 1536) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="gf_eval_ortho_") as td:
        sc_p = Path(td) / "render_ortho.py"
        sc_p.write_text(_BLENDER_ORTHO_SCRIPT, encoding="utf-8")
        subprocess.run(
            [str(blender_exe), "-b", "--factory-startup", "--python-exit-code", "1", "-P", str(sc_p), "--", str(model_path), str(out_dir), str(STAGE_B_SRC), str(width), str(height)],
            check=True,
        )
    _save_turnaround_grid([out_dir / fn for fn, _ in VIEW_AZIMUTHS], out_dir / "turnaround_grid.png")
    return json.loads((out_dir / "render_meta.json").read_text(encoding="utf-8"))


def render_keyframe_preview(model_path: Path, out_dir: Path, *, out_grid: Path | None = None, ortho_views_dir: Path | None = None, blender_exe: Path = DEFAULT_BLENDER_EXE) -> dict[str, Any]:
    if str(STAGE_B_SRC) not in sys.path:
        sys.path.insert(0, str(STAGE_B_SRC))
    from sprite_pipeline.config import SpriteSheetConfig
    from sprite_pipeline.domain.postprocess import downsample_and_ink_3d_frame
    from sprite_pipeline.infrastructure.exr_reader import load_multilayer_exr_frame

    out_dir.mkdir(parents=True, exist_ok=True)
    exr_dir = out_dir / "exr"
    with tempfile.TemporaryDirectory() as td:
        td_p = Path(td)
        spec_p, sc_p, man_p = td_p / "spec.json", td_p / "kf.py", td_p / "manifest.json"
        spec_p.write_text(json.dumps({
            "stage_b_src": str(STAGE_B_SRC),
            "model_path": str(model_path.resolve()),
            "walk_fbx": str(DEFAULT_WALK_FBX),
            "run_fbx": str(DEFAULT_RUN_FBX),
            "exr_dir": str(exr_dir),
            "blender_manifest": str(man_p),
        }, indent=2), encoding="utf-8")
        sc_p.write_text(_BLENDER_KEYFRAME_SCRIPT, encoding="utf-8")
        subprocess.run([str(blender_exe), "-b", "--factory-startup", "--python-exit-code", "1", "-P", str(sc_p), "--", str(spec_p)], check=True)
        man = json.loads(man_p.read_text(encoding="utf-8"))

    ss_cfg = SpriteSheetConfig.for_preset("512x768")
    kf_pngs: list[tuple[str, Path]] = []
    for item in man["rendered"]:
        exr_data = load_multilayer_exr_frame(Path(item["raw_path"]))
        inked, _nrm = downsample_and_ink_3d_frame(
            exr_data["albedo_u8"],
            exr_data["normal_u8"],
            item["cycle"],
            cfg=ss_cfg,
            use_3d_ink=True,
            mat_id_512=exr_data["mat_id"],
            depth_512=exr_data["depth"],
            vec_512_f32=exr_data["normal_f32"],
            alb_f32_srgb=exr_data["albedo_srgb_f32"],
        )
        png_p = out_dir / f"{item['cycle']}_{item['frame']:02d}_3q.png"
        Image.fromarray(inked, mode="RGBA").save(png_p)
        kf_pngs.append((f"{item['cycle']}_{item['frame']:02d}", png_p))

    if out_grid is not None:
        tiles = []
        if ortho_views_dir is not None:
            tiles.append([(lbl, ortho_views_dir / fn) for fn, lbl in (("view_00_front.png", "Ortho Front"), ("view_01_right.png", "Ortho Right"), ("view_02_back.png", "Ortho Back"), ("view_03_left.png", "Ortho Left"))])
        tiles.append(kf_pngs)
        tw, th, hh, pad = 360, 540, 34, 10
        max_cols = max(len(r) for r in tiles)
        sheet = Image.new("RGBA", (max_cols * tw + (max_cols + 1) * pad, len(tiles) * (th + hh) + (len(tiles) + 1) * pad), (28, 30, 34, 255))
        for ri, row in enumerate(tiles):
            for ci, (lbl, p) in enumerate(row):
                tile = Image.new("RGBA", (tw, th + hh), (212, 215, 220, 255))
                draw = ImageDraw.Draw(tile)
                draw.rectangle([0, 0, tw - 1, hh - 1], fill=(38, 42, 48, 255))
                draw.text((10, 7), lbl, fill=(242, 244, 248, 255))
                if p.is_file():
                    im = Image.open(p).convert("RGBA")
                    sc = min((tw - 8) / im.width, (th - 8) / im.height)
                    rz = im.resize((max(1, int(im.width * sc)), max(1, int(im.height * sc))), Image.Resampling.LANCZOS)
                    tile.alpha_composite(rz, ((tw - rz.width) // 2, hh + (th - rz.height) // 2))
                sheet.alpha_composite(tile, (pad + ci * (tw + pad), pad + ri * (th + hh + pad)))
        out_grid.parent.mkdir(parents=True, exist_ok=True)
        sheet.save(out_grid)
    return {"rendered": kf_pngs}


def run_stage_b_strict_prep(glb_path: Path, out_prep_glb: Path, out_report_json: Path, *, blender_exe: Path = DEFAULT_BLENDER_EXE) -> dict[str, Any]:
    out_prep_glb.parent.mkdir(parents=True, exist_ok=True)
    out_report_json.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [str(blender_exe), "-b", "--factory-startup", "--python-exit-code", "1", "-P", str(DEFAULT_PREP_SCRIPT), "--", "--model", str(glb_path), "--out_model", str(out_prep_glb), "--report", str(out_report_json), "--strict"],
        check=True,
    )
    return json.loads(out_report_json.read_text(encoding="utf-8"))


def evaluate_stage_a_glb(
    glb_path: Path,
    fixture_dir: Path,
    out_dir: Path,
    *,
    char_id: str = "character",
    blender_exe: Path = DEFAULT_BLENDER_EXE,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    render_dir = out_dir / "eval" / "rendered_views"
    render_meta = render_model_ortho_views(glb_path, render_dir, blender_exe=blender_exe)

    view_metrics = {}
    for fname, _ in VIEW_AZIMUTHS:
        ref_v, cand_v = fixture_dir / fname, render_dir / fname
        if ref_v.is_file() and cand_v.is_file():
            view_metrics[fname] = compute_view_pair_metrics(ref_v, cand_v)

    parts = render_meta.get("parts", [])
    kf_dir = out_dir / "keyframe_preview"
    gate_grid = out_dir / f"{char_id}_gate3_keyframes_3q.png"
    render_keyframe_preview(glb_path, kf_dir, out_grid=gate_grid, ortho_views_dir=render_dir, blender_exe=blender_exe)

    prep_glb = out_dir / f"{char_id}_stage_b_prep.glb"
    prep_rep_p = out_dir / f"{char_id}_stage_b_report.json"
    stage_b_rep = run_stage_b_strict_prep(glb_path, prep_glb, prep_rep_p, blender_exe=blender_exe)

    eval_report = {
        "glb_path": str(glb_path),
        "fixture_dir": str(fixture_dir),
        "view_metrics": view_metrics,
        "total_faces": sum(int(p["faces"]) for p in parts),
        "parts": parts,
        "gate_grid": str(gate_grid),
        "stage_b_prep_glb": str(prep_glb),
        "stage_b_all_gates_passed": bool(stage_b_rep["qa_gates"]["all_gates_passed"]),
        "stage_b_failed_checks": stage_b_rep["qa_gates"]["failed_checks"],
        "stage_b_metrics": stage_b_rep["metrics"],
    }
    (out_dir / "eval" / "eval_report.json").write_text(json.dumps(eval_report, indent=2), encoding="utf-8")
    return eval_report


def main() -> int:
    ap = argparse.ArgumentParser(description="Evaluate Greenfield Stage A GLB against turnaround + Stage B strict gates")
    ap.add_argument("--glb", type=Path, required=True)
    ap.add_argument("--fixture-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--char-id", default="valkyrie")
    ap.add_argument("--blender-exe", type=Path, default=DEFAULT_BLENDER_EXE)
    args = ap.parse_args()

    rep = evaluate_stage_a_glb(args.glb, args.fixture_dir, args.out_dir, char_id=args.char_id, blender_exe=args.blender_exe)
    print(json.dumps(rep, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
