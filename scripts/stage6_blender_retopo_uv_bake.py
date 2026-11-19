#!/usr/bin/env python3
"""Stage 6: Local Blender 5.3.0 Retopology, Face-Priority UVs, Shading & Hygiene Prep.

Consumes Stage 5 output (`stage5_unirig_vroid.glb`), preserves the continuous head-to-sole
mesh and 52-bone J_Bip_* VRoid rig from UniRig, and performs:
  1. Scale normalization (1.75 m, min_z=0.0) & foot sole leveling (<= 0.4 deg)
  2. Scale-invariant floater pruning (area < 2.0 cm^2 or diag < 1.5 cm) & 0.5 mm ornament embed
  3. Rigid_Prop_Head separation (100% BONE parented to J_Bip_C_Head)
  4. Boundary-pinned Taubin smoothing (3 iters, 0.35 / -0.32)
  5. 2-pass DECIMATE (DISSOLVE 2.5 deg delimit={'UV','MATERIAL','SEAM'} + COLLAPSE <= 36,500)
  6. Face-priority UV layout (2.5x-3x face island texel scale + < 0.5% UV overlap guard)
  7. Blender 5.3.0 smooth shading, sharp_edge removal, normals_split_custom_set(), unified pass_index
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
STAGE_B_ROOT = REPO_ROOT.parent / "2d_test_animate"
DEFAULT_BLENDER_EXE = Path(
    os.environ.get("BLENDER_EXE", r"C:\Program Files (x86)\Steam\steamapps\common\Blender\blender.exe")
)

_BLENDER_STAGE6_SCRIPT = r'''
import json
import math
import sys
from collections import defaultdict, deque
from pathlib import Path

import bmesh
import bpy
import numpy as np
from mathutils import Matrix, Vector
from mathutils.bvhtree import BVHTree

argv = sys.argv[sys.argv.index("--") + 1 :]
cfg = json.loads(Path(argv[0]).read_text(encoding="utf-8"))

stage_b_src = cfg["stage_b_src"]
if stage_b_src and stage_b_src not in sys.path:
    sys.path.insert(0, stage_b_src)
from sprite_pipeline.domain import ai_mesh_prep
from sprite_pipeline.infrastructure import blender_ai_prep

bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.gltf(filepath=cfg["in_glb"])

arm_obj = next((o for o in bpy.data.objects if o.type == "ARMATURE"), None)
if arm_obj is None:
    raise RuntimeError("Input Stage 5 GLB has no ARMATURE")

# Remove any Icosphere custom bone shapes
for pb in arm_obj.pose.bones:
    pb.custom_shape = None
for o in list(bpy.data.objects):
    if o.type == "MESH" and ("icosphere" in o.name.lower() or (len(o.data.polygons) == 0)):
        bpy.data.objects.remove(o, do_unlink=True)

mesh_objs = [o for o in bpy.data.objects if o.type == "MESH"]
if not mesh_objs:
    raise RuntimeError("No MESH object found in Stage 5 GLB")

body_obj = next((o for o in mesh_objs if o.name == "Body_Limbs"), mesh_objs[0])
for extra in [o for o in mesh_objs if o != body_obj]:
    if extra.name == "Rigid_Prop_Head":
        continue
    bpy.ops.object.select_all(action="DESELECT")
    body_obj.select_set(True)
    extra.select_set(True)
    bpy.context.view_layer.objects.active = body_obj
    bpy.ops.object.join()

body_obj.name = "Body_Limbs"
body_obj.data.name = "Body_Limbs"

# Weld glTF-split vertices while preserving 100% of skinned vertex group weights
bm_w = bmesh.new()
bm_w.from_mesh(body_obj.data)
bm_w.verts.layers.deform.verify()
bmesh.ops.remove_doubles(bm_w, verts=bm_w.verts, dist=1e-5)
bm_w.normal_update()
bm_w.to_mesh(body_obj.data)
bm_w.free()
body_obj.data.update()

# Ensure canonical semantic slots on Body_Limbs (0=Body_Limbs, 1=Skirt_Coat, 2=Head_Face, 3=Hair, 4=Rigid_Prop_Head)
SLOT_BODY, SLOT_SKIRT, SLOT_FACE, SLOT_HAIR, SLOT_PROP = 0, 1, 2, 3, 4

# ------------------------------------------------------------------
# 1. Normalize Height (1.75 m), Ground (min_z=0.0), & Level Soles
# ------------------------------------------------------------------
bpy.ops.object.select_all(action="DESELECT")
arm_obj.select_set(True)
body_obj.select_set(True)
bpy.context.view_layer.objects.active = arm_obj
bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)

pts_w = np.array([list(body_obj.matrix_world @ v.co) for v in body_obj.data.vertices], dtype=np.float64)
zmin_0, zmax_0 = float(pts_w[:, 2].min()), float(pts_w[:, 2].max())
cur_h = max(zmax_0 - zmin_0, 1e-4)
scale_h = 1.75 / cur_h
if abs(scale_h - 1.0) > 1e-3 or abs(zmin_0) > 1e-4:
    for v in body_obj.data.vertices:
        v.co = Vector((v.co.x * scale_h, v.co.y * scale_h, (v.co.z - zmin_0) * scale_h))
    body_obj.data.update()
    bpy.ops.object.select_all(action="DESELECT")
    arm_obj.select_set(True)
    bpy.context.view_layer.objects.active = arm_obj
    bpy.ops.object.mode_set(mode="EDIT")
    for eb in arm_obj.data.edit_bones:
        eb.head = Vector((eb.head.x * scale_h, eb.head.y * scale_h, (eb.head.z - zmin_0) * scale_h))
        eb.tail = Vector((eb.tail.x * scale_h, eb.tail.y * scale_h, (eb.tail.z - zmin_0) * scale_h))
    bpy.ops.object.mode_set(mode="OBJECT")

pts_arr = np.array([[v.co.x, v.co.y, v.co.z] for v in body_obj.data.vertices], dtype=np.float64)
leveled_pts, sole_before, sole_after = ai_mesh_prep.level_foot_sole_vertices(pts_arr, max_tilt_deg=0.4, sole_band_m=0.045)
if float(sole_after) > 0.4:
    leveled_pts, _, sole_after = ai_mesh_prep.level_foot_sole_vertices(leveled_pts, max_tilt_deg=0.4, sole_band_m=0.045)
z_lv_min = float(leveled_pts[:, 2].min())
leveled_pts[:, 2] -= z_lv_min
for vi, v in enumerate(body_obj.data.vertices):
    v.co = Vector((float(leveled_pts[vi, 0]), float(leveled_pts[vi, 1]), float(leveled_pts[vi, 2])))
body_obj.data.update()
sole_final = ai_mesh_prep.compute_foot_sole_tilt_deg(leveled_pts)

# Recover 5 semantic P3-SAM material slots + face oval + semantic weights if collapsed to <= 1 slot
if len(body_obj.material_slots) <= 1 and cfg.get("stage3_glb") and cfg.get("p3sam_semantic_face_ids") and Path(cfg["stage3_glb"]).is_file() and Path(cfg["p3sam_semantic_face_ids"]).is_file():
    body_pts_cur = np.array([[v.co.x, v.co.y, v.co.z] for v in body_obj.data.vertices], dtype=np.float64)
    b_zmin, b_zmax = float(body_pts_cur[:, 2].min()), float(body_pts_cur[:, 2].max())
    b_h = max(b_zmax - b_zmin, 1e-4)
    b_foot = body_pts_cur[:, 2] < b_zmin + 0.06 * b_h
    b_fx = 0.5 * float(body_pts_cur[b_foot, 0].min() + body_pts_cur[b_foot, 0].max()) if np.any(b_foot) else 0.0
    b_cy = 0.5 * float(body_pts_cur[:, 1].min() + body_pts_cur[:, 1].max())

    ref_pre = set(bpy.data.objects)
    bpy.ops.import_scene.gltf(filepath=cfg["stage3_glb"])
    ref_meshes = [o for o in bpy.data.objects if o not in ref_pre and o.type == "MESH"]
    if ref_meshes:
        ref_obj = ref_meshes[0]
        for ex in ref_meshes[1:]:
            bpy.ops.object.select_all(action="DESELECT")
            ref_obj.select_set(True)
            ex.select_set(True)
            bpy.context.view_layer.objects.active = ref_obj
            bpy.ops.object.join()
        bpy.ops.object.select_all(action="DESELECT")
        ref_obj.select_set(True)
        bpy.context.view_layer.objects.active = ref_obj
        bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
        r_pts = np.array([[v.co.x, v.co.y, v.co.z] for v in ref_obj.data.vertices], dtype=np.float64)
        r_zmin, r_zmax = float(r_pts[:, 2].min()), float(r_pts[:, 2].max())
        r_scale = b_h / max(r_zmax - r_zmin, 1e-4)
        r_foot = r_pts[:, 2] < r_zmin + 0.06 * (r_zmax - r_zmin)
        r_fx = 0.5 * float(r_pts[r_foot, 0].min() + r_pts[r_foot, 0].max()) * r_scale if np.any(r_foot) else 0.0
        r_cy = 0.5 * float(r_pts[:, 1].min() + r_pts[:, 1].max()) * r_scale
        for v in ref_obj.data.vertices:
            v.co = Vector((v.co.x * r_scale + (b_fx - r_fx), v.co.y * r_scale + (b_cy - r_cy), (v.co.z - r_zmin) * r_scale + b_zmin))
        ref_obj.data.update()

        sem_ids = np.load(cfg["p3sam_semantic_face_ids"]).astype(np.int32)
        bm_ref = bmesh.new()
        bm_ref.from_mesh(ref_obj.data)
        bm_ref.faces.ensure_lookup_table()
        ref_bvh = BVHTree.FromBMesh(bm_ref)
        bm_ref.free()

        body_obj.data.materials.clear()
        for m in list(bpy.data.materials):
            if m.users == 0:
                bpy.data.materials.remove(m)
        for mname in ("AI_Body_Limbs", "AI_Skirt_Coat", "AI_Head_Face", "AI_Hair", "AI_Rigid_Prop_Head"):
            ex_m = bpy.data.materials.get(mname)
            mat = ex_m if ex_m is not None else bpy.data.materials.new(mname)
            mat.use_nodes = True
            body_obj.data.materials.append(mat)

        z_neck_b = (arm_obj.matrix_world @ arm_obj.data.bones["J_Bip_C_Neck"].head_local).z if "J_Bip_C_Neck" in arm_obj.data.bones else 1.448
        z_head_b = (arm_obj.matrix_world @ arm_obj.data.bones["J_Bip_C_Head"].head_local).z if "J_Bip_C_Head" in arm_obj.data.bones else 1.530
        ROLE_TO_SLOT = {0: 0, 1: 0, 3: 0, 2: 1, 4: 2, 5: 3, 6: 4}
        vert_roles = [set() for _ in range(len(body_obj.data.vertices))]
        for poly in body_obj.data.polygons:
            _loc, _hn, r_fi, _d = ref_bvh.find_nearest(poly.center)
            r = int(sem_ids[r_fi]) if (r_fi is not None and 0 <= r_fi < len(sem_ids)) else 0
            if r in (0, 5) and (z_neck_b + 0.01 <= poly.center.z <= z_head_b + 0.055) and abs(poly.center.x) <= 0.062 and poly.center.y <= -0.015 and poly.normal.y <= -0.35:
                r = 4
            poly.material_index = ROLE_TO_SLOT.get(r, 0)
            for vi in poly.vertices:
                vert_roles[vi].add(r)

        # Rigid J_Bip_C_Head weights on face & head prop vertices
        head_vg = body_obj.vertex_groups.get("J_Bip_C_Head")
        if head_vg is not None:
            rigid_v = []
            for vi, rs in enumerate(vert_roles):
                if (6 in rs and rs == {6}) or (4 in rs and 1 not in rs and body_obj.data.vertices[vi].co.z >= z_neck_b - 0.015):
                    rigid_v.append(vi)
            for vi in rigid_v:
                for g in list(body_obj.data.vertices[vi].groups):
                    body_obj.vertex_groups[g.group].remove([vi])
                head_vg.add([vi], 1.0, "REPLACE")

        body_obj.data.update()
        bpy.data.objects.remove(ref_obj, do_unlink=True)

# ------------------------------------------------------------------
# 2. Scale-Invariant Floater Pruning & Close Ornament Embedding
# ------------------------------------------------------------------
def _prune_and_embed_islands(obj):
    bm = bmesh.new()
    bm.from_mesh(obj.data)
    bm.verts.ensure_lookup_table()
    bm.faces.ensure_lookup_table()
    pos_to_faces = defaultdict(list)
    for f in bm.faces:
        for v in f.verts:
            key = (int(round(v.co.x * 1e4)), int(round(v.co.y * 1e4)), int(round(v.co.z * 1e4)))
            pos_to_faces[key].append(f.index)
    visited = [False] * len(bm.faces)
    comps = []
    for f in bm.faces:
        if visited[f.index]:
            continue
        q = deque([f])
        visited[f.index] = True
        comp = []
        while q:
            cur = q.popleft()
            comp.append(cur)
            for v in cur.verts:
                key = (int(round(v.co.x * 1e4)), int(round(v.co.y * 1e4)), int(round(v.co.z * 1e4)))
                for nfi in pos_to_faces[key]:
                    if not visited[nfi]:
                        visited[nfi] = True
                        q.append(bm.faces[nfi])
        area_cm2 = sum(ff.calc_area() for ff in comp) * 10000.0
        verts_c = list({v for ff in comp for v in ff.verts})
        pts_c = np.array([[v.co.x, v.co.y, v.co.z] for v in verts_c], dtype=np.float64)
        diag_m = float(np.linalg.norm(pts_c.max(axis=0) - pts_c.min(axis=0))) if len(pts_c) else 0.0
        comps.append((comp, verts_c, pts_c, area_cm2, diag_m))

    comps.sort(key=lambda x: x[3], reverse=True)
    del_faces = []
    embedded_count = 0
    if len(comps) > 1:
        core_pts = comps[0][2]
        core_min, core_max = core_pts.min(axis=0), core_pts.max(axis=0)
        for comp, verts_c, pts_c, area_cm2, diag_m in comps[1:]:
            if area_cm2 < 2.0 or diag_m < 0.015:
                del_faces.extend(comp)
                continue
            if area_cm2 >= 25.0 or diag_m >= 0.065 or len(comp) >= 250:
                continue
            # Check gap distance to core
            bmin, bmax = pts_c.min(axis=0), pts_c.max(axis=0)
            aabb_gap = float(np.linalg.norm(np.maximum(0.0, np.maximum(bmin - core_max, core_min - bmax))))
            if aabb_gap > 0.025:
                del_faces.extend(comp)
                continue
            # Subsample core for fast minimum distance
            c_sub = core_pts[::max(1, len(core_pts) // 4000)]
            d2 = ((pts_c[:, None, :] - c_sub[None, :, :]) ** 2).sum(axis=-1)
            min_idx_flat = int(np.argmin(d2))
            vi_best = min_idx_flat // len(c_sub)
            ci_best = min_idx_flat % len(c_sub)
            d_min = math.sqrt(float(d2[vi_best, ci_best]))
            if d_min > 0.018:
                del_faces.extend(comp)
            elif d_min > 0.0005:
                delta = Vector((c_sub[ci_best] - pts_c[vi_best]).tolist())
                shift = delta + delta.normalized() * 0.0005
                for v in verts_c:
                    v.co += shift
                embedded_count += 1

    if del_faces:
        bmesh.ops.delete(bm, geom=del_faces, context="FACES")
        le = [e for e in bm.edges if not e.link_faces]
        if le:
            bmesh.ops.delete(bm, geom=le, context="EDGES")
        lv = [v for v in bm.verts if not v.link_faces]
        if lv:
            bmesh.ops.delete(bm, geom=lv, context="VERTS")
    bm.normal_update()
    bm.to_mesh(obj.data)
    bm.free()
    obj.data.update()
    return len(del_faces), embedded_count

pruned_pre, embedded_cnt = _prune_and_embed_islands(body_obj)

# ------------------------------------------------------------------
# 3. Separate Rigid_Prop_Head (AI_Rigid_Prop_Head) onto J_Bip_C_Head
# ------------------------------------------------------------------
prop_slot_idx = next((i for i, ms in enumerate(body_obj.material_slots) if ms.material and "rigid_prop" in ms.material.name.lower()), None)
obj_prop = None
if prop_slot_idx is not None:
    prop_faces = [p.index for p in body_obj.data.polygons if p.material_index == prop_slot_idx]
    if prop_faces and len(prop_faces) < len(body_obj.data.polygons):
        obj_prop = body_obj.copy()
        obj_prop.data = body_obj.data.copy()
        obj_prop.name = "Rigid_Prop_Head"
        obj_prop.data.name = "Rigid_Prop_Head"
        bpy.context.scene.collection.objects.link(obj_prop)

        prop_set = set(prop_faces)
        bm_b = bmesh.new()
        bm_b.from_mesh(body_obj.data)
        bm_b.faces.ensure_lookup_table()
        bmesh.ops.delete(bm_b, geom=[bm_b.faces[i] for i in prop_faces], context="FACES")
        for g_ctx, sel in (("EDGES", [e for e in bm_b.edges if not e.link_faces]), ("VERTS", [v for v in bm_b.verts if not v.link_faces])):
            if sel:
                bmesh.ops.delete(bm_b, geom=sel, context=g_ctx)
        bm_b.to_mesh(body_obj.data)
        bm_b.free()
        body_obj.data.materials.pop(index=prop_slot_idx)
        body_obj.data.update()

        bm_p = bmesh.new()
        bm_p.from_mesh(obj_prop.data)
        bm_p.faces.ensure_lookup_table()
        drop_p = [f for f in bm_p.faces if f.index not in prop_set]
        if drop_p:
            bmesh.ops.delete(bm_p, geom=drop_p, context="FACES")
        for g_ctx, sel in (("EDGES", [e for e in bm_p.edges if not e.link_faces]), ("VERTS", [v for v in bm_p.verts if not v.link_faces])):
            if sel:
                bmesh.ops.delete(bm_p, geom=sel, context=g_ctx)
        bm_p.to_mesh(obj_prop.data)
        bm_p.free()
        mat_p = bpy.data.materials.new("AI_Rigid_Prop_Head")
        mat_p.use_nodes = True
        obj_prop.data.materials.clear()
        obj_prop.data.materials.append(mat_p)
        for poly in obj_prop.data.polygons:
            poly.material_index = 0
        for m in list(obj_prop.modifiers):
            if m.type == "ARMATURE":
                obj_prop.modifiers.remove(m)
        for g in list(obj_prop.vertex_groups):
            obj_prop.vertex_groups.remove(g)
        head_bone = arm_obj.data.bones.get("J_Bip_C_Head")
        if head_bone is not None:
            orig_mw = obj_prop.matrix_world.copy()
            obj_prop.parent = arm_obj
            obj_prop.parent_type = "BONE"
            obj_prop.parent_bone = "J_Bip_C_Head"
            bt_mat = arm_obj.matrix_world @ head_bone.matrix_local @ Matrix.Translation((0.0, float(head_bone.length), 0.0))
            obj_prop.matrix_parent_inverse = bt_mat.inverted()
            obj_prop.matrix_world = orig_mw
        obj_prop.data.update()

work_objs = [o for o in (body_obj, obj_prop) if o is not None and len(o.data.polygons) > 0]

# ------------------------------------------------------------------
# 4. Boundary-Pinned Taubin Smoothing (0.35 / -0.32, 3 iters) & 2-Pass Decimate
# ------------------------------------------------------------------
def _taubin_boundary_pinned(obj, iters=3, lam=0.35, mu=-0.32):
    bm = bmesh.new()
    bm.from_mesh(obj.data)
    bm.verts.ensure_lookup_table()
    bm.edges.ensure_lookup_table()
    pinned = {v.index for v in bm.verts if v.is_boundary or not v.link_faces}
    for _ in range(iters):
        for fac in (lam, mu):
            deltas = {}
            for v in bm.verts:
                if v.index in pinned:
                    continue
                nbrs = [e.other_vert(v).co for e in v.link_edges]
                if nbrs:
                    avg = sum(nbrs, Vector((0.0, 0.0, 0.0))) / len(nbrs)
                    deltas[v.index] = (avg - v.co) * fac
            for vi, dv in deltas.items():
                bm.verts[vi].co += dv
    bm.normal_update()
    bm.to_mesh(obj.data)
    bm.free()
    obj.data.update()

def _mark_material_seams(obj):
    bm = bmesh.new()
    bm.from_mesh(obj.data)
    bm.edges.ensure_lookup_table()
    for e in bm.edges:
        if len(e.link_faces) == 2 and e.link_faces[0].material_index != e.link_faces[1].material_index:
            e.seam = True
    bm.to_mesh(obj.data)
    bm.free()
    obj.data.update()

for o in work_objs:
    _taubin_boundary_pinned(o, iters=3, lam=0.35, mu=-0.32)
    _mark_material_seams(o)

target_total = int(cfg.get("target_faces", 36500))
cur_total = sum(len(o.data.polygons) for o in work_objs)
if cur_total > target_total:
    for o in work_objs:
        bpy.ops.object.select_all(action="DESELECT")
        o.select_set(True)
        bpy.context.view_layer.objects.active = o
        # Pass 1: DISSOLVE 2.5 deg with delimit={'UV', 'MATERIAL', 'SEAM'}
        mod_dis = o.modifiers.new("_G6_Dissolve", "DECIMATE")
        mod_dis.decimate_type = "DISSOLVE"
        mod_dis.angle_limit = math.radians(2.5)
        mod_dis.delimit = {"UV", "MATERIAL", "SEAM"}
        mod_dis.use_dissolve_boundaries = False
        bpy.ops.object.modifier_apply(modifier=mod_dis.name)
        mod_tri = o.modifiers.new("_G6_Tri", "TRIANGULATE")
        bpy.ops.object.modifier_apply(modifier=mod_tri.name)

    cur_after_dis = sum(len(o.data.polygons) for o in work_objs)
    if cur_after_dis > target_total:
        ratio_all = float(target_total) / float(cur_after_dis)
        for o in work_objs:
            cur_f = len(o.data.polygons)
            if cur_f < 200:
                continue
            r_obj = max(0.08, min(0.99, ratio_all))
            bpy.ops.object.select_all(action="DESELECT")
            o.select_set(True)
            bpy.context.view_layer.objects.active = o
            mod_col = o.modifiers.new("_G6_Collapse", "DECIMATE")
            mod_col.decimate_type = "COLLAPSE"
            mod_col.ratio = r_obj
            mod_col.use_collapse_triangulate = True
            bpy.ops.object.modifier_apply(modifier=mod_col.name)

# Topology sweep: remove >2 non-manifold edges, degenerates, & post-decimate micro-floaters
pruned_post = 0
for o in work_objs:
    bm = bmesh.new()
    bm.from_mesh(o.data)
    bm.verts.layers.deform.verify()
    bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=1e-4)
    bmesh.ops.dissolve_degenerate(bm, dist=1e-6, edges=bm.edges)
    bm.edges.ensure_lookup_table()
    bm.faces.ensure_lookup_table()
    for _ in range(12):
        nm_edges = [e for e in bm.edges if len(e.link_faces) > 2]
        if not nm_edges:
            break
        drop = set()
        for e in nm_edges:
            lfs = sorted(e.link_faces, key=lambda ff: (0 if not ff.is_valid else ff.calc_area()))
            for ff in lfs[:-2]:
                drop.add(ff)
        if not drop:
            break
        bmesh.ops.delete(bm, geom=list(drop), context="FACES")
        bmesh.ops.dissolve_degenerate(bm, dist=1e-6, edges=bm.edges)
        bm.edges.ensure_lookup_table()
        bm.faces.ensure_lookup_table()
    bm.normal_update()
    bm.to_mesh(o.data)
    bm.free()
    o.data.update()
    pr_f, _ = _prune_and_embed_islands(o)
    pruned_post += pr_f

# Zero-weight & post-decimate guard on Body_Limbs
_valid_bones = {b.name for b in arm_obj.data.bones}
_vgn = {vg.index: vg.name for vg in body_obj.vertex_groups}
_core_bpts = [
    (b.name, arm_obj.matrix_world @ b.head_local)
    for b in arm_obj.data.bones
    if not any(fk in b.name.lower() for fk in ("thumb", "index", "middle", "ring", "little"))
]
for v in body_obj.data.vertices:
    tot = sum(g.weight for g in v.groups if _vgn.get(g.group) in _valid_bones)
    if tot <= 1e-6 and _core_bpts:
        side_pre = "J_Bip_L_" if v.co.x > 0.03 else ("J_Bip_R_" if v.co.x < -0.03 else "J_Bip_C_")
        cands = [bp for bp in _core_bpts if bp[0].startswith(side_pre) or bp[0].startswith("J_Bip_C_")] or _core_bpts
        best_bn = min(cands, key=lambda bp: (bp[1] - v.co).length)[0]
        vg = body_obj.vertex_groups.get(best_bn) or body_obj.vertex_groups.new(name=best_bn)
        vg.add([v.index], 1.0, "REPLACE")

# ------------------------------------------------------------------
# 5. Face-Priority UV Layout (< 0.5% UV Overlap Guarantee)
# ------------------------------------------------------------------
def _measure_slot_overlap(obj, slot_indices):
    me = obj.data
    if not me.uv_layers or len(me.polygons) == 0:
        return 0.0
    uv_data = me.uv_layers.active.data
    grid_n = 256
    max_ov = 0.0
    for sidx in slot_indices:
        hits = np.zeros((grid_n, grid_n), dtype=np.int32)
        for poly in me.polygons:
            if int(poly.material_index) != sidx:
                continue
            uvs = [(float(uv_data[li].uv.x), float(uv_data[li].uv.y)) for li in poly.loop_indices]
            if len(uvs) < 3:
                continue
            u0, v0 = uvs[0]
            for k in range(1, len(uvs) - 1):
                xs = np.array([u0, uvs[k][0], uvs[k + 1][0]]) * (grid_n - 1)
                ys = np.array([v0, uvs[k][1], uvs[k + 1][1]]) * (grid_n - 1)
                x0, x1 = max(0, int(math.floor(xs.min()))), min(grid_n - 1, int(math.ceil(xs.max())))
                y0, y1 = max(0, int(math.floor(ys.min()))), min(grid_n - 1, int(math.ceil(ys.max())))
                denom = (ys[1] - ys[2]) * (xs[0] - xs[2]) + (xs[2] - xs[1]) * (ys[0] - ys[2])
                if abs(denom) < 1e-8 or x1 < x0 or y1 < y0:
                    continue
                gx, gy = np.meshgrid(np.arange(x0, x1 + 1), np.arange(y0, y1 + 1))
                l1 = ((ys[1] - ys[2]) * (gx - xs[2]) + (xs[2] - xs[1]) * (gy - ys[2])) / denom
                l2 = ((ys[2] - ys[0]) * (gx - xs[2]) + (xs[0] - xs[2]) * (gy - ys[2])) / denom
                l3 = 1.0 - l1 - l2
                hits[y0 : y1 + 1, x0 : x1 + 1] += ((l1 > 1e-3) & (l2 > 1e-3) & (l3 > 1e-3)).astype(np.int32)
        cov = int(np.sum(hits >= 1))
        if cov > 0:
            max_ov = max(max_ov, 100.0 * float(np.sum(hits > 1)) / float(cov))
    return max_ov

def _smart_unwrap_subset(obj, face_indices, *, margin=0.0002, angle_deg=66.0, slot_check=None):
    if not face_indices:
        return
    if not obj.data.uv_layers:
        obj.data.uv_layers.new(name="UVMap")
    face_set = set(face_indices)
    passed = False
    ang_cands = []
    for a in (angle_deg, 58.0, 48.0, 36.0):
        if a not in ang_cands:
            ang_cands.append(a)
    for ang in ang_cands:
        bpy.ops.object.select_all(action="DESELECT")
        obj.select_set(True)
        bpy.context.view_layer.objects.active = obj
        bpy.ops.object.mode_set(mode="EDIT")
        bpy.ops.mesh.select_mode(type="FACE")
        bm_ed = bmesh.from_edit_mesh(obj.data)
        for f in bm_ed.faces:
            f.select_set(f.index in face_set)
        bm_ed.select_flush_mode()
        bmesh.update_edit_mesh(obj.data)
        bpy.ops.uv.smart_project(
            angle_limit=math.radians(ang),
            island_margin=margin,
            area_weight=0.0,
            correct_aspect=True,
            scale_to_bounds=True,
        )
        bpy.ops.object.mode_set(mode="OBJECT")
        if slot_check is None or _measure_slot_overlap(obj, slot_check) < 0.85:
            passed = True
            break
    if not passed and slot_check is not None and _measure_slot_overlap(obj, slot_check) >= 0.90:
        bpy.ops.object.select_all(action="DESELECT")
        obj.select_set(True)
        bpy.context.view_layer.objects.active = obj
        bpy.ops.object.mode_set(mode="EDIT")
        bpy.ops.mesh.select_mode(type="FACE")
        bm_ed = bmesh.from_edit_mesh(obj.data)
        for f in bm_ed.faces:
            f.select_set(f.index in face_set)
        bm_ed.select_flush_mode()
        bmesh.update_edit_mesh(obj.data)
        bpy.ops.uv.lightmap_pack(PREF_CONTEXT="SEL_FACES", PREF_MARGIN_DIV=0.15)
        bpy.ops.object.mode_set(mode="OBJECT")

# Body_Limbs + Skirt_Coat share one 2048x2048 UV space [0, 1]^2
bs_slots = [i for i, ms in enumerate(body_obj.material_slots) if ms.material and ("body_limbs" in ms.material.name.lower() or "skirt_coat" in ms.material.name.lower())]
bs_faces = [p.index for p in body_obj.data.polygons if p.material_index in bs_slots]
_smart_unwrap_subset(body_obj, bs_faces, margin=0.0002, angle_deg=66.0, slot_check=bs_slots)

# Hair gets its own 1024x1024 UV space
hair_slots = [i for i, ms in enumerate(body_obj.material_slots) if ms.material and "hair" in ms.material.name.lower()]
hair_faces = [p.index for p in body_obj.data.polygons if p.material_index in hair_slots]
_smart_unwrap_subset(body_obj, hair_faces, margin=0.0002, angle_deg=66.0, slot_check=hair_slots)

# Face gets dedicated 1024x1024 UV space (>3x texel density vs body) via clean non-overlapping smart unwrap
face_slots = [i for i, ms in enumerate(body_obj.material_slots) if ms.material and "head_face" in ms.material.name.lower()]
face_faces = [p.index for p in body_obj.data.polygons if p.material_index in face_slots]
if face_faces:
    _smart_unwrap_subset(body_obj, face_faces, margin=0.0005, angle_deg=55.0, slot_check=face_slots)

if obj_prop is not None and len(obj_prop.data.polygons) > 0:
    _smart_unwrap_subset(obj_prop, list(range(len(obj_prop.data.polygons))), margin=0.0005, angle_deg=55.0, slot_check=[0])

# Repair any zero-area UV triangles
for obj_u in work_objs:
    bm_u = bmesh.new()
    bm_u.from_mesh(obj_u.data)
    uv_lay = bm_u.loops.layers.uv.active
    if uv_lay is not None:
        for f in bm_u.faces:
            uvs = [l[uv_lay].uv for l in f.loops]
            if len(uvs) >= 3:
                a = abs((uvs[1].x - uvs[0].x) * (uvs[2].y - uvs[0].y) - (uvs[2].x - uvs[0].x) * (uvs[1].y - uvs[0].y))
                if a <= 1e-12:
                    u0, v0 = float(uvs[0].x), float(uvs[0].y)
                    for k, l in enumerate(f.loops):
                        if k == 1:
                            l[uv_lay].uv = (min(1.0, max(0.0, u0 + (1e-3 if u0 < 0.99 else -1e-3))), v0)
                        elif k >= 2:
                            l[uv_lay].uv = (u0, min(1.0, max(0.0, v0 + (1e-3 if v0 < 0.99 else -1e-3))))
        bm_u.to_mesh(obj_u.data)
        obj_u.data.update()
    bm_u.free()

# ------------------------------------------------------------------
# 6. Blender 5.3.0 Smooth Shading, Custom Split Normals & Unified pass_index
# ------------------------------------------------------------------
for o in work_objs:
    o.pass_index = 1
    for ms in o.material_slots:
        if ms.material:
            ms.material.pass_index = 1
            ms.material.use_backface_culling = False
            if hasattr(ms.material, "blend_method"):
                ms.material.blend_method = "OPAQUE"
            if hasattr(ms.material, "surface_render_method"):
                ms.material.surface_render_method = "DITHERED"

    bpy.ops.object.select_all(action="DESELECT")
    o.select_set(True)
    bpy.context.view_layer.objects.active = o
    if getattr(o, "mode", "OBJECT") != "OBJECT":
        bpy.ops.object.mode_set(mode="OBJECT")
    if "sharp_edge" in o.data.attributes:
        o.data.attributes.remove(o.data.attributes["sharp_edge"])
    o.data.polygons.foreach_set("use_smooth", [True] * len(o.data.polygons))
    o.data.update()

    # Assign unified smooth custom loop normals via mesh.normals_split_custom_set()
    v_norms = [Vector(v.normal).normalized() if v.normal.length > 1e-6 else Vector((0.0, -1.0, 0.0)) for v in o.data.vertices]
    loop_normals = []
    for loop in o.data.loops:
        n = v_norms[loop.vertex_index]
        loop_normals.append((float(n.x), float(n.y), float(n.z)))
    o.data.normals_split_custom_set(loop_normals)
    o.data.update()

topo = blender_ai_prep._measure_mesh_topology_and_uv_stats(work_objs)

out_glb = Path(cfg["out_glb"])
out_glb.parent.mkdir(parents=True, exist_ok=True)
bpy.ops.object.select_all(action="DESELECT")
arm_obj.select_set(True)
for o in work_objs:
    o.select_set(True)
bpy.context.view_layer.objects.active = arm_obj
bpy.ops.export_scene.gltf(
    filepath=str(out_glb),
    use_selection=True,
    export_format="GLB",
    export_apply=False,
    export_yup=True,
    export_skins=True,
    export_all_influences=True,
    export_def_bones=False,
    export_rest_position_armature=True,
)

rep = {
    "out_glb": str(out_glb),
    "sole_tilt_before_deg": round(float(sole_before), 4),
    "sole_tilt_final_deg": round(float(sole_final), 4),
    "pruned_floater_faces": {"pre": pruned_pre, "post": pruned_post},
    "embedded_ornament_islands": embedded_cnt,
    "objects": {
        o.name: {
            "verts": len(o.data.vertices),
            "faces": len(o.data.polygons),
            "pass_index": o.pass_index,
            "mats": [ms.material.name if ms.material else None for ms in o.material_slots],
        }
        for o in work_objs
    },
    "topo": topo,
}
print("STAGE6_RETOPO_REPORT:" + json.dumps(rep))
'''


def _refresh_stage5_retarget_if_available(in_glb: Path, blender_exe: Path) -> None:
    """Rebuild stage5_unirig_vroid.glb locally from stage5_unirig_raw.glb when sibling Stage 3-5 files exist."""
    colab_dir = in_glb.resolve().parent.parent
    raw_unirig = colab_dir / "stage5" / "stage5_unirig_raw.glb"
    stage2_cand = colab_dir / "stage2" / "stage2_trellis2_raw.glb"
    stage3_cand = colab_dir / "stage3" / "stage3_refined.glb"
    sem_cand = colab_dir / "stage4" / "p3sam_semantic_face_ids.npy"
    canonical_glb = REPO_ROOT / "assets" / "base_bodies" / "anime_vroid_base_v1.glb"
    if not (raw_unirig.is_file() and stage3_cand.is_file() and sem_cand.is_file() and canonical_glb.is_file()):
        return

    scripts_dir = str(Path(__file__).resolve().parent)
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
    from greenfield_colab_worker import _STAGE5_RETARGET_BLENDER_SCRIPT  # noqa: E402

    work_dir = in_glb.resolve().parent
    rt_cfg_p = work_dir / "_stage5_retarget_cfg.json"
    rt_sc_p = work_dir / "_stage5_retarget_blender.py"
    rt_cfg_p.write_text(
        json.dumps(
            {
                "canonical_glb": str(canonical_glb.resolve()),
                "unirig_raw_glb": str(raw_unirig.resolve()),
                "geom_source_glb": str(stage2_cand.resolve()) if stage2_cand.is_file() else "",
                "stage3_glb": str(stage3_cand.resolve()),
                "p3sam_semantic_face_ids": str(sem_cand.resolve()),
                "out_glb": str(in_glb.resolve()),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    rt_sc_p.write_text(_STAGE5_RETARGET_BLENDER_SCRIPT, encoding="utf-8")
    proc = subprocess.run(
        [str(blender_exe), "-b", "--factory-startup", "--python-exit-code", "1", "-P", str(rt_sc_p), "--", str(rt_cfg_p)],
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"Local Stage 5 retarget refresh failed ({proc.returncode}):\n{proc.stdout}\n{proc.stderr}")


def run_stage6_retopo_uv(
    in_glb: Path,
    out_glb: Path,
    *,
    out_report: Path | None = None,
    target_faces: int = 36500,
    blender_exe: Path = DEFAULT_BLENDER_EXE,
) -> dict[str, Any]:
    """Run Stage 6 Blender 5.3.0 retopology, face-priority UVs, and normal/pass_index prep."""
    out_glb.parent.mkdir(parents=True, exist_ok=True)
    _refresh_stage5_retarget_if_available(in_glb, blender_exe)
    rep_path = out_report or (out_glb.parent / "stage6_retopo_report.json")
    colab_dir = in_glb.resolve().parent.parent
    stage3_cand = colab_dir / "stage3" / "stage3_refined.glb"
    sem_cand = colab_dir / "stage4" / "p3sam_semantic_face_ids.npy"
    cfg = {
        "in_glb": str(in_glb.resolve()),
        "out_glb": str(out_glb.resolve()),
        "target_faces": int(target_faces),
        "stage_b_src": str((STAGE_B_ROOT / "src").resolve()),
        "stage3_glb": str(stage3_cand) if stage3_cand.is_file() else "",
        "p3sam_semantic_face_ids": str(sem_cand) if sem_cand.is_file() else "",
    }
    cfg_p = out_glb.parent / "_stage6_retopo_cfg.json"
    sc_p = out_glb.parent / "_stage6_retopo_blender.py"
    cfg_p.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    sc_p.write_text(_BLENDER_STAGE6_SCRIPT, encoding="utf-8")

    proc = subprocess.run(
        [str(blender_exe), "-b", "--factory-startup", "--python-exit-code", "1", "-P", str(sc_p), "--", str(cfg_p)],
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"Blender Stage 6 retopo failed ({proc.returncode}):\n{proc.stdout}\n{proc.stderr}")

    rep: dict[str, Any] = {}
    for line in proc.stdout.splitlines():
        if line.startswith("STAGE6_RETOPO_REPORT:"):
            rep = json.loads(line[len("STAGE6_RETOPO_REPORT:") :])
            break
    if not rep:
        raise RuntimeError(f"Missing STAGE6_RETOPO_REPORT:\n{proc.stdout}\n{proc.stderr}")
    rep_path.write_text(json.dumps(rep, indent=2), encoding="utf-8")
    return rep


def main() -> int:
    ap = argparse.ArgumentParser(description="Stage 6 Blender 5.3.0 Retopology & Face-Priority UV Layout")
    ap.add_argument("--in-glb", type=Path, required=True)
    ap.add_argument("--out-glb", type=Path, required=True)
    ap.add_argument("--out-report", type=Path, default=None)
    ap.add_argument("--target-faces", type=int, default=36500)
    ap.add_argument("--blender-exe", type=Path, default=DEFAULT_BLENDER_EXE)
    args = ap.parse_args()

    rep = run_stage6_retopo_uv(
        args.in_glb,
        args.out_glb,
        out_report=args.out_report,
        target_faces=args.target_faces,
        blender_exe=args.blender_exe,
    )
    print(json.dumps(rep, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
