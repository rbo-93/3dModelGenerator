#!/usr/bin/env python3
"""Stage 6B: Direct 4-View Orthographic Flat Cel Albedo Bake & Pull-Push Crevice Fill.

Projects the 4 de-lit turnaround RGBA views (0, 90, 180, 270 deg) directly onto
the Stage 6 UV layout (2048x2048 body+skirt shared atlas, 1024x1024 high-res face,
1024x1024 hair, 1024x1024 rigid head prop) using exact Cycles Float32 world-position
visibility EXRs and multi-scale pull-push inpainting for occluded crevices.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image
from scipy.spatial import cKDTree

DEFAULT_BLENDER_EXE = Path(
    os.environ.get("BLENDER_EXE", r"C:\Program Files (x86)\Steam\steamapps\common\Blender\blender.exe")
)
TARGET_HEIGHT_M = 1.75


def _delight_rgba_soft(rgba: np.ndarray, lift_strength: float = 0.22) -> np.ndarray:
    """Gently lift cast midtone shadows in CIELAB L-channel without washing out ink or dark armor."""
    fg = rgba[..., 3] >= 128
    if np.sum(fg) < 64:
        return rgba
    lab = cv2.cvtColor(rgba[..., :3], cv2.COLOR_RGB2LAB).astype(np.float32)
    L = lab[..., 0]
    p78 = float(np.percentile(L[fg], 78))
    lift_mask = fg & (L >= 42.0) & (L < p78)
    L[lift_mask] = np.clip(L[lift_mask] + lift_strength * np.maximum(0.0, p78 - L[lift_mask]), 0.0, 252.0)
    lab[..., 0] = L
    out = rgba.copy()
    out[..., :3] = cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2RGB)
    return out


def _read_exr_rgba(exr_path: Path) -> np.ndarray:
    import OpenEXR

    with OpenEXR.File(str(exr_path)) as f:
        ch = f.channels()
        if "RGBA" in ch:
            return ch["RGBA"].pixels.astype(np.float32)
        return np.stack([ch[k].pixels.astype(np.float32) for k in ("R", "G", "B", "A")], axis=-1)


def _rasterize_uv_attributes(
    uvs: np.ndarray,
    positions: np.ndarray,
    normals: np.ndarray,
    faces_tri: np.ndarray,
    res: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pos_map = np.zeros((res, res, 3), dtype=np.float32)
    norm_map = np.zeros((res, res, 3), dtype=np.float32)
    valid_map = np.zeros((res, res), dtype=bool)
    hard_valid = np.zeros((res, res), dtype=bool)

    base_bary = (
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, 0.0, 1.0),
        (0.5, 0.5, 0.0),
        (0.5, 0.0, 0.5),
        (0.0, 0.5, 0.5),
        (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0),
    )

    for tri in faces_tri:
        i0, i1, i2 = int(tri[0]), int(tri[1]), int(tri[2])
        u0, v0 = uvs[i0]
        u1, v1 = uvs[i1]
        u2, v2 = uvs[i2]
        px0, py0 = u0 * (res - 1), (1.0 - v0) * (res - 1)
        px1, py1 = u1 * (res - 1), (1.0 - v1) * (res - 1)
        px2, py2 = u2 * (res - 1), (1.0 - v2) * (res - 1)

        any_hard = False
        denom = (py1 - py2) * (px0 - px2) + (px2 - px1) * (py0 - py2)
        if abs(denom) >= 1e-8:
            xmin = max(0, int(math.floor(min(px0, px1, px2))) - 1)
            xmax = min(res - 1, int(math.ceil(max(px0, px1, px2))) + 1)
            ymin = max(0, int(math.floor(min(py0, py1, py2))) - 1)
            ymax = min(res - 1, int(math.ceil(max(py0, py1, py2))) + 1)
            if xmax >= xmin and ymax >= ymin:
                xs = np.arange(xmin, xmax + 1, dtype=np.float32)[None, :]
                ys = np.arange(ymin, ymax + 1, dtype=np.float32)[:, None]
                w0 = ((py1 - py2) * (xs - px2) + (px2 - px1) * (ys - py2)) / denom
                w1 = ((py2 - py0) * (xs - px2) + (px0 - px2) * (ys - py2)) / denom
                w2 = 1.0 - w0 - w1
                inside = (w0 >= -0.04) & (w1 >= -0.04) & (w2 >= -0.04)
                if np.any(inside):
                    any_hard = True
                    p_sub = w0[inside, None] * positions[i0] + w1[inside, None] * positions[i1] + w2[inside, None] * positions[i2]
                    n_sub = w0[inside, None] * normals[i0] + w1[inside, None] * normals[i1] + w2[inside, None] * normals[i2]
                    n_sub /= np.maximum(np.linalg.norm(n_sub, axis=1, keepdims=True), 1e-6)
                    pos_map[ymin : ymax + 1, xmin : xmax + 1][inside] = p_sub
                    norm_map[ymin : ymax + 1, xmin : xmax + 1][inside] = n_sub
                    valid_map[ymin : ymax + 1, xmin : xmax + 1][inside] = True
                    hard_valid[ymin : ymax + 1, xmin : xmax + 1][inside] = True

        # Guarantee 100% coverage of bilinear 2x2 footprints along every vertex/edge/centroid (especially sub-pixel slivers)
        bary_samples = list(base_bary)
        max_edge_px = max(math.hypot(px1 - px0, py1 - py0), math.hypot(px2 - px1, py2 - py1), math.hypot(px0 - px2, py0 - py2))
        if (not any_hard or max_edge_px > 2.0) and max_edge_px > 2.0:
            n_edge = min(16, int(math.ceil(max_edge_px)))
            for s in range(1, n_edge):
                t = float(s) / float(n_edge)
                bary_samples.append((1.0 - t, t, 0.0))
                bary_samples.append((0.0, 1.0 - t, t))
                bary_samples.append((t, 0.0, 1.0 - t))

        p0, p1, p2 = positions[i0], positions[i1], positions[i2]
        n0, n1, n2 = normals[i0], normals[i1], normals[i2]
        for b0, b1, b2 in bary_samples:
            px = b0 * px0 + b1 * px1 + b2 * px2
            py = b0 * py0 + b1 * py1 + b2 * py2
            fx0 = max(0, min(res - 1, int(math.floor(px))))
            fy0 = max(0, min(res - 1, int(math.floor(py))))
            fx1 = min(res - 1, fx0 + 1)
            fy1 = min(res - 1, fy0 + 1)
            if hard_valid[fy0, fx0] and hard_valid[fy0, fx1] and hard_valid[fy1, fx0] and hard_valid[fy1, fx1]:
                continue
            p_s = b0 * p0 + b1 * p1 + b2 * p2
            n_s = b0 * n0 + b1 * n1 + b2 * n2
            nl = float(np.linalg.norm(n_s))
            if nl > 1e-6:
                n_s = n_s / nl
            for sy in (fy0, fy1):
                for sx in (fx0, fx1):
                    if not hard_valid[sy, sx]:
                        pos_map[sy, sx] = p_s
                        norm_map[sy, sx] = n_s
                        valid_map[sy, sx] = True

    return pos_map, norm_map, valid_map


def _pull_push_inpaint(
    rgb_f32: np.ndarray,
    covered_mask: np.ndarray,
    island_mask: np.ndarray,
    *,
    fallback_rgb: np.ndarray,
) -> np.ndarray:
    """Multi-scale pull-push pyramid fill for occluded crevices + 12px UV gutter dilation."""
    h, w = covered_mask.shape
    rgb = rgb_f32.copy()
    k3 = np.ones((3, 3), np.uint8)
    uncovered = island_mask & (~covered_mask)
    if np.any(uncovered):
        if np.any(covered_mask):
            pyr_c = [np.where(covered_mask[..., None], rgb, 0.0)]
            pyr_w = [covered_mask.astype(np.float32)]
            max_lvl = max(1, int(math.log2(min(h, w))) - 2)
            for _ in range(max_lvl):
                c_prev, w_prev = pyr_c[-1], pyr_w[-1]
                if min(c_prev.shape[:2]) <= 4:
                    break
                pyr_c.append(cv2.resize(c_prev, (c_prev.shape[1] // 2, c_prev.shape[0] // 2), interpolation=cv2.INTER_AREA))
                pyr_w.append(cv2.resize(w_prev, (w_prev.shape[1] // 2, w_prev.shape[0] // 2), interpolation=cv2.INTER_AREA))

            filled = np.zeros_like(pyr_c[-1])
            w_top = pyr_w[-1]
            nz_top = w_top > 1e-6
            filled[nz_top] = pyr_c[-1][nz_top] / w_top[nz_top, None]
            if not np.all(nz_top):
                filled[~nz_top] = fallback_rgb
            for lvl in range(len(pyr_c) - 2, -1, -1):
                up = cv2.resize(filled, (pyr_c[lvl].shape[1], pyr_c[lvl].shape[0]), interpolation=cv2.INTER_LINEAR)
                w_cur = pyr_w[lvl]
                nz = w_cur > 1e-4
                blend = up.copy()
                blend[nz] = (pyr_c[lvl][nz] + up[nz] * 0.05) / (w_cur[nz, None] + 0.05)
                filled = blend
            rgb[uncovered] = filled[uncovered]
        else:
            rgb[uncovered] = fallback_rgb

    out_rgb = np.clip(np.round(rgb), 0, 255).astype(np.uint8)
    filled_mask = island_mask.copy()
    for _ in range(12):
        nxt = cv2.dilate(filled_mask.astype(np.uint8), k3) > 0
        ring = nxt & (~filled_mask)
        if not np.any(ring):
            break
        dil_rgb = cv2.dilate(np.where(filled_mask[..., None], out_rgb, 0), k3)
        out_rgb[ring] = dil_rgb[ring]
        filled_mask = nxt

    return np.dstack([out_rgb, np.full((h, w), 255, dtype=np.uint8)])


def _project_points_to_view(
    pts: np.ndarray,
    *,
    azimuth_deg: float,
    center_world: np.ndarray,
    ortho_scale_v: float,
    width: int,
    height: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rad = math.radians(azimuth_deg)
    cam_forward = np.array([-math.sin(rad), math.cos(rad), 0.0], dtype=np.float32)
    cam_right = np.array([math.cos(rad), math.sin(rad), 0.0], dtype=np.float32)
    cam_up = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    view_to_cam = -cam_forward

    rel = pts - center_world[None, :]
    x_cam = rel @ cam_right
    z_cam = rel @ cam_up

    h_m = float(ortho_scale_v)
    w_m = h_m * (float(width) / max(1.0, float(height)))
    px = (x_cam / w_m + 0.5) * float(width) - 0.5
    py = (0.5 - z_cam / h_m) * float(height) - 0.5
    return px, py, view_to_cam


_BLENDER_EXPORT_POS_SCRIPT = r"""
import json, math, sys
from collections import deque
from pathlib import Path
import bmesh, bpy, numpy as np
from mathutils import Vector

argv = sys.argv[sys.argv.index("--") + 1 :]
glb_path = Path(argv[0]).resolve()
work_dir = Path(argv[1]).resolve()
views_spec = json.loads(Path(argv[2]).read_text(encoding="utf-8"))
base_glb_arg = argv[3] if len(argv) > 3 else ""

bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.gltf(filepath=str(glb_path))

meshes = [o for o in bpy.data.objects if o.type == "MESH" and not o.name.startswith("Icosphere")]
if not meshes:
    raise RuntimeError(f"No meshes in {glb_path}")

deps = bpy.context.evaluated_depsgraph_get()
mesh_manifest = {}
all_pos_chunks = []
for obj in meshes:
    ev = obj.evaluated_get(deps)
    me = bpy.data.meshes.new_from_object(ev)
    me.transform(ev.matrix_world.copy())
    bm = bmesh.new()
    bm.from_mesh(me)
    bmesh.ops.triangulate(bm, faces=bm.faces)
    uv_lay = bm.loops.layers.uv.active or bm.loops.layers.uv.new("UVMap")
    bm.verts.ensure_lookup_table()
    bm.faces.ensure_lookup_table()

    n_loops = len(bm.faces) * 3
    uvs = np.zeros((n_loops, 2), dtype=np.float32)
    pos = np.zeros((n_loops, 3), dtype=np.float32)
    nrm = np.zeros((n_loops, 3), dtype=np.float32)
    tris = np.arange(n_loops, dtype=np.int32).reshape(-1, 3)
    face_mat_idx = np.zeros(len(bm.faces), dtype=np.int32)
    bm.normal_update()

    idx = 0
    for fi, f in enumerate(bm.faces):
        face_mat_idx[fi] = int(f.material_index)
        fn = Vector(f.normal)
        for loop in f.loops:
            uvs[idx] = (float(loop[uv_lay].uv.x), float(loop[uv_lay].uv.y))
            pos[idx] = tuple(loop.vert.co)
            vn = Vector(loop.vert.normal)
            if vn.length < 1e-4 or vn.dot(fn) < -0.5:
                vn = fn
            nrm[idx] = tuple(vn.normalized() if vn.length > 1e-4 else Vector((0.0, -1.0, 0.0)))
            idx += 1
    bm.free()
    bpy.data.meshes.remove(me)
    if len(pos) > 0:
        all_pos_chunks.append(pos)

    npz_p = work_dir / f"mesh_{obj.name}.npz"
    np.savez(npz_p, uvs=uvs, positions=pos, normals=nrm, faces=tris, face_mat_idx=face_mat_idx)
    slots = [{"slot_idx": int(si), "mat_name": ms.material.name if ms.material else f"AI_{obj.name}_{si}"} for si, ms in enumerate(obj.material_slots)]
    mesh_manifest[obj.name] = {"npz": str(npz_p), "slots": slots or [{"slot_idx": 0, "mat_name": f"AI_{obj.name}"}]}

all_pos = np.vstack(all_pos_chunks) if all_pos_chunks else np.zeros((1, 3), dtype=np.float32)
z_c_min, z_c_max = float(all_pos[:, 2].min()), float(all_pos[:, 2].max())
mesh_h_m = max(0.1, z_c_max - z_c_min)
z_mid_3d = 0.5 * (z_c_min + z_c_max)

pos_mat = bpy.data.materials.new("_ExactPosMat")
pos_mat.use_nodes = True
nt = pos_mat.node_tree
nt.nodes.clear()
out_n = nt.nodes.new("ShaderNodeOutputMaterial")
em = nt.nodes.new("ShaderNodeEmission")
em.inputs["Strength"].default_value = 1.0
geom = nt.nodes.new("ShaderNodeNewGeometry")
nt.links.new(geom.outputs["Position"], em.inputs["Color"])
nt.links.new(em.outputs["Emission"], out_n.inputs["Surface"])

for obj in meshes:
    if len(obj.data.materials) == 0:
        obj.data.materials.append(pos_mat)
    else:
        for i in range(len(obj.data.materials)):
            obj.data.materials[i] = pos_mat

scene = bpy.context.scene
scene.render.engine = "CYCLES"
scene.cycles.device = "CPU"
scene.cycles.samples = 1
scene.cycles.use_denoising = False
scene.render.film_transparent = True
scene.view_settings.view_transform = "Raw"
scene.view_settings.look = "None"
scene.render.image_settings.file_format = "OPEN_EXR"
scene.render.image_settings.color_mode = "RGBA"
scene.render.image_settings.color_depth = "32"

world = bpy.data.worlds.new("_PosWorld")
scene.world = world
world.use_nodes = True
bg = world.node_tree.nodes.get("Background")
if bg:
    bg.inputs["Color"].default_value = (0.0, 0.0, 0.0, 0.0)
    bg.inputs["Strength"].default_value = 0.0

cam_data = bpy.data.cameras.new("_VisCam")
cam_data.type = "ORTHO"
cam_data.sensor_fit = "VERTICAL"
cam_data.clip_start = 0.01
cam_data.clip_end = 30.0
cam_obj = bpy.data.objects.new("_VisCam", cam_data)
scene.collection.objects.link(cam_obj)
scene.camera = cam_obj

exr_paths = []
views_resolved = {}
for v in views_spec:
    w_v = float(v["width"])
    h_v = float(v["height"])
    rad = math.radians(float(v["azimuth_deg"]))
    cam_right = np.array([math.cos(rad), math.sin(rad), 0.0], dtype=np.float32)
    x_proj = all_pos @ cam_right
    x_mid_3d = 0.5 * (float(x_proj.min()) + float(x_proj.max()))
    x0_px, x1_px, y0_px, y1_px = [float(val) for val in v["fg_bbox"]]
    fg_h_px = max(10.0, y1_px - y0_px)
    o_scale = mesh_h_m * (h_v / fg_h_px)
    w_m = o_scale * (w_v / max(1.0, h_v))
    fg_cx_px = 0.5 * (x0_px + x1_px)
    fg_cy_px = 0.5 * (y0_px + y1_px)
    x_cam_center = x_mid_3d + (0.5 - (fg_cx_px + 0.5) / w_v) * w_m
    z_center = z_mid_3d + ((fg_cy_px + 0.5) / h_v - 0.5) * o_scale
    cx = float(cam_right[0]) * x_cam_center
    cy = float(cam_right[1]) * x_cam_center
    cz = z_center

    scene.render.resolution_x = int(w_v)
    scene.render.resolution_y = int(h_v)
    scene.render.resolution_percentage = 100
    cam_data.ortho_scale = float(o_scale)
    center_vec = Vector((cx, cy, cz))
    cam_pos = center_vec + Vector((5.0 * math.sin(rad), -5.0 * math.cos(rad), 0.0))
    cam_obj.location = cam_pos
    cam_obj.rotation_euler = (center_vec - cam_pos).normalized().to_track_quat("-Z", "Y").to_euler()
    exr_p = work_dir / f"pos_vis_{v['index']:02d}.exr"
    scene.render.filepath = str(exr_p)
    bpy.ops.render.render(write_still=True)
    exr_paths.append(str(exr_p))
    views_resolved[str(v["index"])] = {
        "center_world": [cx, cy, cz],
        "ortho_scale_v": float(o_scale),
    }

base_o_voxel_npz = ""
base_albedo_png = ""
if base_glb_arg and Path(base_glb_arg).is_file():
    pre_imgs = set(bpy.data.images)
    pre_objs = set(bpy.data.objects)
    bpy.ops.import_scene.gltf(filepath=str(Path(base_glb_arg).resolve()))
    new_objs = [o for o in bpy.data.objects if o not in pre_objs]
    new_meshes = [o for o in new_objs if o.type == "MESH"]
    new_imgs = [im for im in bpy.data.images if im not in pre_imgs and im.size[0] > 0]

    albedo_img = None
    for bo in new_meshes:
        for ms in bo.material_slots:
            mat = ms.material
            if mat is None or not mat.use_nodes or not mat.node_tree:
                continue
            for lk in mat.node_tree.links:
                if lk.to_socket.name == "Base Color" and lk.from_node.type == "TEX_IMAGE" and lk.from_node.image in new_imgs:
                    albedo_img = lk.from_node.image
                    break
            if albedo_img is not None:
                break
        if albedo_img is not None:
            break
    if albedo_img is None and new_imgs:
        albedo_img = next((im for im in new_imgs if "image_0" in im.name.lower()), new_imgs[0])

    if new_meshes and albedo_img is not None:
        base_obj = new_meshes[0]
        for ex in new_meshes[1:]:
            bpy.ops.object.select_all(action="DESELECT")
            base_obj.select_set(True)
            ex.select_set(True)
            bpy.context.view_layer.objects.active = base_obj
            bpy.ops.object.join()
        bpy.ops.object.select_all(action="DESELECT")
        base_obj.select_set(True)
        bpy.context.view_layer.objects.active = base_obj
        base_obj.parent = None
        bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)

        bm_b = bmesh.new()
        bm_b.from_mesh(base_obj.data)
        bmesh.ops.remove_doubles(bm_b, verts=bm_b.verts, dist=1e-5)
        bm_b.faces.ensure_lookup_table()
        vis_b = [False] * len(bm_b.faces)
        comps_b = []
        for ff in bm_b.faces:
            if vis_b[ff.index]:
                continue
            q = deque([ff])
            vis_b[ff.index] = True
            comp = []
            while q:
                cur = q.popleft()
                comp.append(cur)
                for ed in cur.edges:
                    for nf in ed.link_faces:
                        if not vis_b[nf.index]:
                            vis_b[nf.index] = True
                            q.append(nf)
            comps_b.append(comp)
        if len(comps_b) > 1:
            max_fb = max(len(c) for c in comps_b)
            keep_tb = max(500, int(0.005 * max_fb))
            drop_b = [f for c in comps_b if len(c) < keep_tb for f in c]
            if drop_b and len(drop_b) < len(bm_b.faces):
                bmesh.ops.delete(bm_b, geom=drop_b, context="FACES")
                le = [e for e in bm_b.edges if not e.link_faces]
                if le:
                    bmesh.ops.delete(bm_b, geom=le, context="EDGES")
                lv = [v for v in bm_b.verts if not v.link_faces]
                if lv:
                    bmesh.ops.delete(bm_b, geom=lv, context="VERTS")
        bmesh.ops.triangulate(bm_b, faces=bm_b.faces)
        bm_b.verts.ensure_lookup_table()
        bm_b.faces.ensure_lookup_table()

        foot_c = all_pos[:, 2] < z_c_min + 0.06 * mesh_h_m
        cx_c = 0.5 * float(all_pos[foot_c, 0].min() + all_pos[foot_c, 0].max()) if np.any(foot_c) else 0.0
        cy_c = 0.5 * float(all_pos[:, 1].min() + all_pos[:, 1].max())
        b_pts = np.array([[v.co.x, v.co.y, v.co.z] for v in bm_b.verts], dtype=np.float64)
        b_zmin, b_zmax = float(b_pts[:, 2].min()), float(b_pts[:, 2].max())
        b_scale = mesh_h_m / max(b_zmax - b_zmin, 1e-4)
        b_foot = b_pts[:, 2] < b_zmin + 0.06 * (b_zmax - b_zmin)
        b_cx = 0.5 * float(b_pts[b_foot, 0].min() + b_pts[b_foot, 0].max()) * b_scale if np.any(b_foot) else 0.0
        b_cy = 0.5 * float(b_pts[:, 1].min() + b_pts[:, 1].max()) * b_scale
        for v in bm_b.verts:
            v.co = Vector((v.co.x * b_scale + (cx_c - b_cx), v.co.y * b_scale + (cy_c - b_cy), (v.co.z - b_zmin) * b_scale + z_c_min))

        uv_b = bm_b.loops.layers.uv.active
        n_bf = len(bm_b.faces)
        tri_pts = np.zeros((n_bf, 3, 3), dtype=np.float32)
        tri_uvs = np.zeros((n_bf, 3, 2), dtype=np.float32)
        for fi, f in enumerate(bm_b.faces):
            for li, lp in enumerate(f.loops[:3]):
                tri_pts[fi, li] = tuple(lp.vert.co)
                if uv_b is not None:
                    tri_uvs[fi, li] = (float(lp[uv_b].uv.x), float(lp[uv_b].uv.y))
        bm_b.free()

        p0, p1, p2 = tri_pts[:, 0], tri_pts[:, 1], tri_pts[:, 2]
        u0, u1, u2 = tri_uvs[:, 0], tri_uvs[:, 1], tri_uvs[:, 2]
        sample_pts = np.concatenate([p0, p1, p2, 0.5 * (p0 + p1), 0.5 * (p1 + p2), 0.5 * (p2 + p0), (p0 + p1 + p2) / 3.0], axis=0).astype(np.float32)
        sample_uvs = np.concatenate([u0, u1, u2, 0.5 * (u0 + u1), 0.5 * (u1 + u2), 0.5 * (u2 + u0), (u0 + u1 + u2) / 3.0], axis=0).astype(np.float32)

        npz_base_p = work_dir / "base_o_voxel.npz"
        if albedo_img.packed_file is not None:
            png_base_p = work_dir / "base_albedo.png"
            png_base_p.write_bytes(bytes(albedo_img.packed_file.data))
            base_albedo_png = str(png_base_p)
            np.savez(npz_base_p, sample_pts=sample_pts, sample_uvs=sample_uvs)
        else:
            iw, ih = int(albedo_img.size[0]), int(albedo_img.size[1])
            ch = int(albedo_img.channels)
            buf = np.empty(iw * ih * ch, dtype=np.float32)
            albedo_img.pixels.foreach_get(buf)
            arr = np.clip(np.round(buf.reshape(ih, iw, ch)[::-1, :, :4] * 255.0), 0, 255).astype(np.uint8)
            np.savez(npz_base_p, sample_pts=sample_pts, sample_uvs=sample_uvs, albedo_rgba=arr)
        base_o_voxel_npz = str(npz_base_p)

(work_dir / "export_manifest.json").write_text(
    json.dumps(
        {
            "meshes": mesh_manifest,
            "exr_paths": exr_paths,
            "views_resolved": views_resolved,
            "base_o_voxel_npz": base_o_voxel_npz,
            "base_albedo_png": base_albedo_png,
        },
        indent=2,
    ),
    encoding="utf-8",
)
"""


_BLENDER_PACK_SCRIPT = r"""
import json, sys
from pathlib import Path
import bpy
from mathutils import Vector

argv = sys.argv[sys.argv.index("--") + 1 :]
glb_in = Path(argv[0]).resolve()
out_glb = Path(argv[1]).resolve()
spec = json.loads(Path(argv[2]).read_text(encoding="utf-8"))

bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.gltf(filepath=str(glb_in))

def _apply_tex(mat, tex_path: Path, roughness: float) -> None:
    img = bpy.data.images.load(str(tex_path), check_existing=True)
    img.colorspace_settings.name = "sRGB"
    if not img.packed_file:
        img.pack()
    mat.use_nodes = True
    nt = mat.node_tree
    nt.nodes.clear()
    out_n = nt.nodes.new("ShaderNodeOutputMaterial")
    bsdf = nt.nodes.new("ShaderNodeBsdfPrincipled")
    tex_n = nt.nodes.new("ShaderNodeTexImage")
    tex_n.image = img
    tex_n.extension = "EXTEND"
    nt.links.new(tex_n.outputs["Color"], bsdf.inputs["Base Color"])
    if hasattr(mat, "blend_method"):
        mat.blend_method = "OPAQUE"
    if hasattr(mat, "surface_render_method"):
        mat.surface_render_method = "DITHERED"
    if hasattr(mat, "use_backface_culling"):
        mat.use_backface_culling = False
    bsdf.inputs["Roughness"].default_value = roughness
    nt.links.new(bsdf.outputs["BSDF"], out_n.inputs["Surface"])

tex_map = spec["textures"]
for obj_name, info in tex_map.items():
    obj = bpy.data.objects.get(obj_name)
    if obj is None or obj.type != "MESH":
        continue
    obj.pass_index = 1
    roughness = 0.78 if "Prop" not in obj_name else 0.42
    if "slots" in info:
        for sidx_s, sinfo in info["slots"].items():
            sidx = int(sidx_s)
            if 0 <= sidx < len(obj.material_slots):
                ms = obj.material_slots[sidx]
                if ms.material is None:
                    ms.material = bpy.data.materials.new(f"AI_{obj_name}_slot{sidx}")
                ms.material.pass_index = 1
                _apply_tex(ms.material, Path(sinfo["png"]).resolve(), roughness)
    else:
        tex_p = Path(info["png"]).resolve()
        if len(obj.material_slots) == 0:
            obj.data.materials.append(bpy.data.materials.new(f"AI_{obj_name}"))
        for ms in obj.material_slots:
            if ms.material is None:
                ms.material = bpy.data.materials.new(f"AI_{obj_name}")
            ms.material.pass_index = 1
            _apply_tex(ms.material, tex_p, roughness)
    if "sharp_edge" in obj.data.attributes:
        obj.data.attributes.remove(obj.data.attributes["sharp_edge"])
    obj.data.polygons.foreach_set("use_smooth", [True] * len(obj.data.polygons))
    obj.data.update()
    v_norms = [Vector(v.normal).normalized() if v.normal.length > 1e-6 else Vector((0.0, -1.0, 0.0)) for v in obj.data.vertices]
    obj.data.normals_split_custom_set([(float(v_norms[l.vertex_index].x), float(v_norms[l.vertex_index].y), float(v_norms[l.vertex_index].z)) for l in obj.data.loops])
    obj.data.update()

out_glb.parent.mkdir(parents=True, exist_ok=True)
bpy.ops.export_scene.gltf(
    filepath=str(out_glb),
    export_format="GLB",
    export_apply=False,
    export_yup=True,
    export_texcoords=True,
    export_normals=True,
    export_skins=True,
    export_all_influences=True,
    export_def_bones=False,
    export_rest_position_armature=True,
    export_materials="EXPORT",
)
print(f"FLAT_CEL_BAKE_OK:{out_glb}")
"""


def bake_flat_cel_albedo(
    *,
    glb_path: Path,
    out_glb: Path,
    view_paths: dict[str, Path],
    tex_dir: Path | None = None,
    base_glb: Path | None = None,
    delight_shadows: bool = True,
    body_res: int = 2048,
    part_res: int = 1024,
    blender_exe: Path = DEFAULT_BLENDER_EXE,
) -> dict[str, Any]:
    """Project 4 orthogonal turnaround RGBA sheets onto Stage 6 UVs and pack into GLB."""
    glb_path = Path(glb_path).resolve()
    out_glb = Path(out_glb).resolve()
    out_glb.parent.mkdir(parents=True, exist_ok=True)
    if tex_dir is None:
        tex_dir = out_glb.parent / f"{out_glb.stem}_textures"
    tex_dir = Path(tex_dir).resolve()
    tex_dir.mkdir(parents=True, exist_ok=True)

    # Load all 4 views & compute shared vertical span so ortho camera framing matches 1:1
    ordered_views = [
        ("front", 0.0, 4.0),
        ("right", 90.0, 1.2),
        ("back", 180.0, 4.0),
        ("left", 270.0, 1.2),
    ]
    loaded_rgbas: dict[str, np.ndarray] = {}
    y_mins, y_maxs = [], []
    for role, _az, _prio in ordered_views:
        vp = view_paths.get(role)
        if vp is None or not Path(vp).is_file():
            continue
        rgba = np.asarray(Image.open(vp).convert("RGBA"))
        if delight_shadows:
            rgba = _delight_rgba_soft(rgba, lift_strength=0.22)
        fg = rgba[..., 3] > 128
        ys = np.where(fg)[0]
        if len(ys):
            y_mins.append(int(ys.min()))
            y_maxs.append(int(ys.max()))
        loaded_rgbas[role] = rgba

    if not loaded_rgbas:
        raise ValueError("No valid view RGBA files supplied to bake_flat_cel_albedo")

    g_y0, g_y1 = float(min(y_mins)), float(max(y_maxs))
    full_h_px = max(60.0, g_y1 - g_y0)
    m_per_px = TARGET_HEIGHT_M / full_h_px

    # Determine spine horizontal center from front/back views
    fb_cxs = []
    for r in ("front", "back"):
        if r in loaded_rgbas:
            xs = np.where(loaded_rgbas[r][..., 3] > 128)[1]
            if len(xs):
                fb_cxs.append(0.5 * (float(xs.min()) + float(xs.max())))

    prepared_views: list[dict[str, Any]] = []
    for idx, (role, az, prio) in enumerate(ordered_views):
        if role not in loaded_rgbas:
            continue
        rgba = loaded_rgbas[role]
        h_v, w_v = rgba.shape[:2]
        ys_v, xs_v = np.where(rgba[..., 3] > 128)
        vx0 = float(xs_v.min()) if len(xs_v) else 0.0
        vx1 = float(xs_v.max()) if len(xs_v) else float(w_v - 1)
        cx_px = 0.5 * (vx0 + vx1)
        o_scale = float(h_v) * m_per_px
        rad = math.radians(az)
        right_offset_m = (0.5 * float(w_v) - cx_px) * m_per_px
        cam_right = np.array([math.cos(rad), math.sin(rad), 0.0], dtype=np.float32)
        z_center = (g_y1 - 0.5 * float(h_v)) * m_per_px
        center_world = np.array([cam_right[0] * right_offset_m, cam_right[1] * right_offset_m, z_center], dtype=np.float32)

        # Erode alpha slightly by 1px for crisp edge color without white/gray halo bleed
        fg_erode = cv2.erode((rgba[..., 3] > 128).astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
        prepared_views.append({
            "index": idx,
            "role": role,
            "azimuth_deg": az,
            "priority": prio,
            "rgba": rgba,
            "fg_erode": fg_erode,
            "fg_bbox": [vx0, vx1, g_y0, g_y1],
            "width": w_v,
            "height": h_v,
            "ortho_scale_v": o_scale,
            "center_world": center_world,
        })

    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td:
        td_p = Path(td)
        views_spec = [
            {
                "index": pv["index"],
                "azimuth_deg": pv["azimuth_deg"],
                "fg_bbox": pv["fg_bbox"],
                "width": pv["width"],
                "height": pv["height"],
                "ortho_scale_v": pv["ortho_scale_v"],
                "center_world": pv["center_world"].tolist(),
            }
            for pv in prepared_views
        ]
        spec_p = td_p / "views_spec.json"
        spec_p.write_text(json.dumps(views_spec, indent=2), encoding="utf-8")
        exp_sc = td_p / "export_pos.py"
        exp_sc.write_text(_BLENDER_EXPORT_POS_SCRIPT, encoding="utf-8")

        base_glb_str = str(Path(base_glb).resolve()) if (base_glb is not None and Path(base_glb).is_file()) else ""
        proc = subprocess.run(
            [str(blender_exe), "-b", "--factory-startup", "--python-exit-code", "1", "-P", str(exp_sc), "--", str(glb_path), str(td_p), str(spec_p), base_glb_str],
            capture_output=True,
            text=True,
        )
        if proc.returncode != 0:
            raise RuntimeError(f"Blender position EXR export failed:\n{proc.stdout}\n{proc.stderr}")

        exp_manifest = json.loads((td_p / "export_manifest.json").read_text(encoding="utf-8"))
        for pv in prepared_views:
            vres = exp_manifest.get("views_resolved", {}).get(str(pv["index"]))
            if vres:
                pv["center_world"] = np.array(vres["center_world"], dtype=np.float32)
                pv["ortho_scale_v"] = float(vres["ortho_scale_v"])
        exr_arrays = [_read_exr_rgba(Path(ep)) for ep in exp_manifest["exr_paths"]]

        base_kdtree: cKDTree | None = None
        base_uvs: np.ndarray | None = None
        base_rgb_img: np.ndarray | None = None
        base_npz_str = str(exp_manifest.get("base_o_voxel_npz") or "")
        if base_npz_str and Path(base_npz_str).is_file():
            with np.load(base_npz_str) as bnpz:
                base_pts = bnpz["sample_pts"].astype(np.float32)
                base_uvs = bnpz["sample_uvs"].astype(np.float32)
                base_png_str = str(exp_manifest.get("base_albedo_png") or "")
                if base_png_str and Path(base_png_str).is_file():
                    b_rgba = np.asarray(Image.open(base_png_str).convert("RGBA"))
                elif "albedo_rgba" in bnpz:
                    b_rgba = bnpz["albedo_rgba"].astype(np.uint8)
                else:
                    b_rgba = None
            if b_rgba is not None and len(base_pts) > 0:
                if delight_shadows:
                    b_rgba = _delight_rgba_soft(b_rgba, lift_strength=0.18)
                base_rgb_img = b_rgba[..., :3].astype(np.float32)
                base_kdtree = cKDTree(base_pts)

        front_pv = next((v for v in prepared_views if v["role"] == "front"), prepared_views[0])
        fg_m = front_pv["rgba"][..., 3] >= 128
        fallback_rgb = np.median(front_pv["rgba"][fg_m, :3], axis=0).astype(np.float32) if np.any(fg_m) else np.array([180.0, 175.0, 175.0], dtype=np.float32)

        f_rgba = front_pv["rgba"]
        y_head_lo = max(0, int(round(g_y0)))
        y_head_hi = min(f_rgba.shape[0], int(round(g_y0 + 0.22 * full_h_px)))
        head_band = f_rgba[y_head_lo:y_head_hi]
        hb_rgb = head_band[..., :3].astype(np.int32)
        skin_mask = (
            (head_band[..., 3] >= 128)
            & (hb_rgb[..., 0] > 210)
            & (hb_rgb[..., 1] > 170)
            & (hb_rgb[..., 2] > 150)
            & (hb_rgb[..., 0] > hb_rgb[..., 1])
            & (hb_rgb[..., 1] > hb_rgb[..., 2])
            & ((hb_rgb[..., 0] - hb_rgb[..., 2]) > 20)
        )
        skin_rgb = (
            np.median(hb_rgb[skin_mask], axis=0).astype(np.float32)
            if np.any(skin_mask)
            else np.array([248.0, 215.0, 196.0], dtype=np.float32)
        )
        hair_mask = (
            (head_band[..., 3] >= 128)
            & (hb_rgb[..., 2] > hb_rgb[..., 1] + 25)
            & (hb_rgb[..., 0] > hb_rgb[..., 1] + 25)
            & (hb_rgb[..., 2] > 140)
        )
        hair_rgb = (
            np.median(hb_rgb[hair_mask], axis=0).astype(np.float32)
            if np.any(hair_mask)
            else np.array([196.0, 114.0, 206.0], dtype=np.float32)
        )

        baked_textures: dict[str, Any] = {}
        bake_stats: dict[str, Any] = {}

        for obj_name, minfo in exp_manifest["meshes"].items():
            with np.load(minfo["npz"]) as npz:
                uvs, pos, nrm, faces = npz["uvs"], npz["positions"], npz["normals"], npz["faces"]
                face_mat_idx = npz["face_mat_idx"]

            slots = minfo["slots"]
            multi_slot = len(slots) > 1
            # Group AI_Body_Limbs + AI_Skirt_Coat onto shared AI_Body_Limbs.png (zero waist texture seam!)
            groups: list[dict[str, Any]] = []
            shared_map: dict[str, int] = {}
            for sinfo in slots:
                mname = str(sinfo["mat_name"]).split(".")[0]
                mlow = mname.lower()
                if multi_slot and ("body_limbs" in mlow or "skirt_coat" in mlow):
                    key = "AI_Body_Limbs"
                    if key in shared_map:
                        groups[shared_map[key]]["slots"].append(sinfo)
                        continue
                    shared_map[key] = len(groups)
                    groups.append({"slots": [sinfo], "out_name": key, "res": body_res})
                else:
                    groups.append({"slots": [sinfo], "out_name": mname, "res": part_res})

            slot_tex_map: dict[str, dict[str, Any]] = {}
            for grp in groups:
                sidxs = [int(s["slot_idx"]) for s in grp["slots"]]
                slot_faces = faces[np.isin(face_mat_idx, sidxs)] if multi_slot else faces
                if len(slot_faces) == 0:
                    continue
                res = int(grp["res"])
                pos_map, norm_map, valid_mask = _rasterize_uv_attributes(uvs, pos, nrm, slot_faces, res)
                act_ys, act_xs = np.where(valid_mask)
                n_act = len(act_ys)
                if n_act == 0:
                    continue

                pts_act = pos_map[act_ys, act_xs]
                nrms_act = norm_map[act_ys, act_xs]
                is_head_face_grp = "head_face" in grp["out_name"].lower()
                sidelock_eye_box = (
                    (pts_act[:, 2] >= 1.36)
                    & (pts_act[:, 2] <= 1.62)
                    & (pts_act[:, 1] >= -0.155)
                    & (pts_act[:, 1] <= -0.020)
                    & (np.abs(pts_act[:, 0]) >= 0.035)
                    & (np.abs(pts_act[:, 0]) <= 0.145)
                )
                accum_rgb = np.zeros((n_act, 3), dtype=np.float32)
                accum_w = np.zeros(n_act, dtype=np.float32)
                ortho_covered = np.zeros(n_act, dtype=bool)
                if base_kdtree is not None and base_uvs is not None and base_rgb_img is not None and not is_head_face_grp:
                    _, nn_idx = base_kdtree.query(pts_act, k=1)
                    bh, bw = base_rgb_img.shape[:2]
                    bu = np.clip(base_uvs[nn_idx, 0], 0.0, 1.0) * (float(bw) - 1.001)
                    bv = np.clip(1.0 - base_uvs[nn_idx, 1], 0.0, 1.0) * (float(bh) - 1.001)
                    bx0, by0 = np.floor(bu).astype(np.int32), np.floor(bv).astype(np.int32)
                    bx1, by1 = np.minimum(bx0 + 1, bw - 1), np.minimum(by0 + 1, bh - 1)
                    btx, bty = (bu - bx0)[:, None], (bv - by0)[:, None]
                    base_rgb_act = (
                        (1.0 - btx) * (1.0 - bty) * base_rgb_img[by0, bx0]
                        + btx * (1.0 - bty) * base_rgb_img[by0, bx1]
                        + (1.0 - btx) * bty * base_rgb_img[by1, bx0]
                        + btx * bty * base_rgb_img[by1, bx1]
                    )
                    if "hair" in grp["out_name"].lower():
                        is_muddy = (base_rgb_act[:, 2] < base_rgb_act[:, 1] + 15.0) | (base_rgb_act.mean(axis=1) < 95.0)
                        if np.any(is_muddy):
                            base_rgb_act[is_muddy] = hair_rgb
                    w_seed = 0.05
                    accum_rgb = base_rgb_act * w_seed
                    accum_w.fill(w_seed)

                for v_idx, pv in enumerate(prepared_views):
                    if is_head_face_grp and pv["role"] != "front":
                        continue
                    exr = exr_arrays[v_idx]
                    px_f, py_f, view_to_cam = _project_points_to_view(
                        pts_act,
                        azimuth_deg=pv["azimuth_deg"],
                        center_world=pv["center_world"],
                        ortho_scale_v=pv["ortho_scale_v"],
                        width=pv["width"],
                        height=pv["height"],
                    )
                    px_i = np.round(px_f).astype(np.int32)
                    py_i = np.round(py_f).astype(np.int32)
                    ib = (px_i >= 0) & (px_i < pv["width"]) & (py_i >= 0) & (py_i < pv["height"])
                    if not np.any(ib):
                        continue

                    ndot = nrms_act @ view_to_cam
                    ib_idx = np.where(ib)[0]
                    ex0 = np.clip(np.floor(px_f[ib_idx]).astype(np.int32), 0, pv["width"] - 1)
                    ey0 = np.clip(np.floor(py_f[ib_idx]).astype(np.int32), 0, pv["height"] - 1)
                    ex1 = np.minimum(ex0 + 1, pv["width"] - 1)
                    ey1 = np.minimum(ey0 + 1, pv["height"] - 1)
                    min_d = np.full(len(ib_idx), 999.0, dtype=np.float32)
                    for cx_a, cy_a in ((ex0, ey0), (ex1, ey0), (ex0, ey1), (ex1, ey1)):
                        epx = exr[cy_a, cx_a]
                        dc = np.where(epx[:, 3] > 0.5, np.linalg.norm(epx[:, :3] - pts_act[ib_idx], axis=1), 999.0)
                        min_d = np.minimum(min_d, dc)

                    d_tol = 0.085 if is_head_face_grp else (0.075 if pv["role"] in ("front", "back") else 0.055)
                    nd_min = -0.05 if is_head_face_grp else 0.06
                    eff_ndot_ib = np.where((not is_head_face_grp) & (min_d < 0.045), np.abs(ndot[ib_idx]), ndot[ib_idx])
                    vis_ok = (min_d < d_tol) & (eff_ndot_ib > nd_min) & pv["fg_erode"][py_i[ib_idx], px_i[ib_idx]]
                    vis_idx = ib_idx[vis_ok]
                    if len(vis_idx) == 0:
                        continue
                    eff_ndot = np.maximum(eff_ndot_ib[vis_ok], 0.05)

                    # Bilinear RGB sample from de-lit RGBA view
                    img_rgb = pv["rgba"][..., :3].astype(np.float32)
                    fx = np.clip(px_f[vis_idx], 0.0, pv["width"] - 1.001)
                    fy = np.clip(py_f[vis_idx], 0.0, pv["height"] - 1.001)
                    x0, y0 = np.floor(fx).astype(np.int32), np.floor(fy).astype(np.int32)
                    x1, y1 = np.minimum(x0 + 1, pv["width"] - 1), np.minimum(y0 + 1, pv["height"] - 1)
                    tx, ty = (fx - x0)[:, None], (fy - y0)[:, None]
                    samp_rgb = (
                        (1.0 - tx) * (1.0 - ty) * img_rgb[y0, x0]
                        + tx * (1.0 - ty) * img_rgb[y0, x1]
                        + (1.0 - tx) * ty * img_rgb[y1, x0]
                        + tx * ty * img_rgb[y1, x1]
                    )
                    if is_head_face_grp and pv["role"] == "front":
                        pt_v = pts_act[vis_idx]
                        in_eye_brow_box = (np.abs(pt_v[:, 0]) <= 0.046) & (pt_v[:, 2] >= 1.480) & (pt_v[:, 2] <= 1.565) & (eff_ndot >= 0.52)
                        in_nose_mouth_box = (np.abs(pt_v[:, 0]) <= 0.026) & (pt_v[:, 2] >= 1.452) & (pt_v[:, 2] <= 1.500) & (eff_ndot >= 0.55)
                        in_facial_features = in_eye_brow_box | in_nose_mouth_box
                        is_non_skin_bleed = (
                            (samp_rgb[:, 2] > samp_rgb[:, 1] + 12.0)
                            | (samp_rgb[:, 1] < 168.0)
                            | (eff_ndot < 0.48)
                        )
                        fix_face = (~in_facial_features) & is_non_skin_bleed
                        if np.any(fix_face):
                            shade = np.clip(0.94 + 0.06 * eff_ndot[fix_face, None], 0.90, 1.02)
                            samp_rgb[fix_face] = np.clip(skin_rgb[None, :] * shade, 0.0, 255.0)
                    if ("hair" in grp["out_name"].lower()) and pv["role"] in ("right", "left"):
                        in_box = sidelock_eye_box[vis_idx]
                        if np.any(in_box):
                            is_hair_px = (samp_rgb[:, 2] > samp_rgb[:, 1] + 18.0) & (samp_rgb[:, 0] > samp_rgb[:, 1] + 18.0) & (samp_rgb[:, 2] > 130.0)
                            is_tiara_px = (
                                (pts_act[vis_idx, 2] >= 1.54)
                                & (~((samp_rgb[:, 0] > 200.0) & (samp_rgb[:, 1] > 155.0) & (samp_rgb[:, 0] > samp_rgb[:, 2] + 15.0)))
                                & (samp_rgb.mean(axis=1) > 95.0)
                            )
                            to_hair = in_box & (~is_hair_px) & (~is_tiara_px)
                            if np.any(to_hair):
                                shade = np.clip(0.90 + 0.12 * eff_ndot[to_hair, None], 0.85, 1.04)
                                samp_rgb[to_hair] = np.clip(hair_rgb[None, :] * shade, 0.0, 255.0)
                    w = (np.maximum(0.0, eff_ndot) ** 4.0) * pv["priority"]
                    accum_rgb[vis_idx] += samp_rgb * w[:, None]
                    accum_w[vis_idx] += w
                    ortho_covered[vis_idx] = True

                cov_flat = accum_w > 1e-4
                rgb_uv = np.zeros((res, res, 3), dtype=np.float32)
                if np.any(cov_flat):
                    rgb_flat = accum_rgb[cov_flat] / accum_w[cov_flat, None]
                    rgb_uv[act_ys[cov_flat], act_xs[cov_flat]] = rgb_flat
                    part_fb = skin_rgb if is_head_face_grp else np.median(rgb_flat, axis=0).astype(np.float32)
                else:
                    part_fb = skin_rgb if is_head_face_grp else fallback_rgb

                cov_mask = np.zeros((res, res), dtype=bool)
                cov_mask[act_ys[cov_flat], act_xs[cov_flat]] = True
                final_rgba = _pull_push_inpaint(rgb_uv, cov_mask, valid_mask, fallback_rgb=part_fb)

                tex_png = tex_dir / f"{grp['out_name']}.png"
                Image.fromarray(final_rgba, mode="RGBA").save(tex_png)
                stat = {
                    "resolution": res,
                    "active_texels": int(n_act),
                    "direct_covered_pct": round(100.0 * float(np.sum(ortho_covered)) / max(1, n_act), 2),
                    "seeded_from_existing": bool(base_kdtree is not None),
                    "png": str(tex_png),
                }
                if multi_slot:
                    for gs in grp["slots"]:
                        slot_tex_map[str(gs["slot_idx"])] = {"png": str(tex_png)}
                        bake_stats[f"{obj_name}::{gs['mat_name']}"] = stat
                else:
                    baked_textures[obj_name] = {"png": str(tex_png)}
                    bake_stats[obj_name] = stat

            if multi_slot and slot_tex_map:
                baked_textures[obj_name] = {"slots": slot_tex_map}

        pack_spec = td_p / "pack_spec.json"
        pack_spec.write_text(json.dumps({"textures": baked_textures}, indent=2), encoding="utf-8")
        pack_sc = td_p / "pack_export.py"
        pack_sc.write_text(_BLENDER_PACK_SCRIPT, encoding="utf-8")
        proc2 = subprocess.run(
            [str(blender_exe), "-b", "--factory-startup", "--python-exit-code", "1", "-P", str(pack_sc), "--", str(glb_path), str(out_glb), str(pack_spec)],
            capture_output=True,
            text=True,
        )
        if proc2.returncode != 0:
            raise RuntimeError(f"Blender GLB texture pack failed:\n{proc2.stdout}\n{proc2.stderr}")

    return {"glb_in": str(glb_path), "out_glb": str(out_glb), "tex_dir": str(tex_dir), "parts": bake_stats}


def main() -> int:
    ap = argparse.ArgumentParser(description="Bake 4-view flat cel albedo onto Stage 6 UV layout")
    ap.add_argument("--glb", type=Path, required=True)
    ap.add_argument("--out-glb", type=Path, required=True)
    ap.add_argument("--tex-dir", type=Path, default=None)
    ap.add_argument("--front", type=Path, required=True)
    ap.add_argument("--right", type=Path, required=True)
    ap.add_argument("--back", type=Path, required=True)
    ap.add_argument("--left", type=Path, required=True)
    ap.add_argument("--no-delight", action="store_true")
    ap.add_argument("--base-glb", type=Path, default=None)
    ap.add_argument("--body-res", type=int, default=2048)
    ap.add_argument("--part-res", type=int, default=1024)
    ap.add_argument("--out-report", type=Path, default=None)
    ap.add_argument("--blender-exe", type=Path, default=DEFAULT_BLENDER_EXE)
    args = ap.parse_args()

    rep = bake_flat_cel_albedo(
        glb_path=args.glb,
        out_glb=args.out_glb,
        view_paths={"front": args.front, "right": args.right, "back": args.back, "left": args.left},
        tex_dir=args.tex_dir,
        base_glb=args.base_glb,
        delight_shadows=not args.no_delight,
        body_res=args.body_res,
        part_res=args.part_res,
        blender_exe=args.blender_exe,
    )
    if args.out_report is not None:
        args.out_report.parent.mkdir(parents=True, exist_ok=True)
        args.out_report.write_text(json.dumps(rep, indent=2), encoding="utf-8")
    print(json.dumps(rep, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
