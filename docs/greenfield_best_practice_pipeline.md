# Greenfield Best-Practice Pipeline: 4-View Anime Turnarounds to Rigged 3D & Godot Cel Sprites

This blueprint is designed from scratch—**independent of your current scripts, workarounds, or model licenses**—for turning 4 orthogonal 2D anime turnaround views (`0°`, `90°`, `180°`, `270°`) into a clean, rigged `<= 39,000`-face 52-bone `J_Bip_*` 3D character that renders crisp `512×768` `clean_cel` + `exr_4b` sprites for Godot 4.7.2.

## End-to-End Best-Practice Stack

| Stage | Environment | Best-Practice Model / Tool | Core Output | Why This Wins Over Alternatives |
| :--- | :--- | :--- | :--- | :--- |
| **1. Preprocessing & Priors** | Colab A100 | `BiRefNet` + shared-scale normalization + [StableNormal_turbo](https://github.com/Stable-X/StableNormal) | 4 scale-locked RGBA cutouts + 4 camera-space normal maps | Locks 1:1 height across wide front/back and narrow side views; removes fragile 2D pose-detector (`DWPose`) failures on stylized outfits. |
| **2. Multi-View 3D Geometry** | Colab A100 | Multi-View [TRELLIS.2-4B](https://github.com/microsoft/TRELLIS.2/pull/104) + [o_voxel (`remesh=False`, 120k)](https://github.com/microsoft/TRELLIS.2/blob/main/o-voxel/o_voxel/postprocess.py) | 120k-face single-layer open-cloth mesh (`doubleSided=True`) | Natively preserves open bell sleeves, open skirt hems, and thin ribbons (unlike watertight SDF models such as Hunyuan3D-2mv/Omni that seal cuffs solid). |
| **3. Silhouette & Normal Lock** | Colab A100 | 100-step `nvdiffrast` + [largesteps](https://pypi.org/project/largesteps/) (`Cholesky`) | Silhouette-locked 120k mesh (`>= 0.75` IoU across all 4 views) | Eliminates side-profile depth collapse and waist/bust distortion without triangle self-intersections. |
| **4. Native 3D Part Isolation** | Colab A100 | [tencent/Hunyuan3D-Part (`P3-SAM` + `X-Part`)](https://huggingface.co/tencent/Hunyuan3D-Part) | Per-face 3D part IDs (`hair`, `face`, `bodice/sleeves`, `skirt`, `shoes`, `props`) | Segments directly in 3D with zero 2D raycast occlusion bleed; `X-Part` (or [HoloPart](https://github.com/VAST-AI-Research/HoloPart)) amodally completes occluded upper shoes/collars. |
| **5. Neural Auto-Rigging** | Colab A100 | [VAST-AI/UniRig](https://huggingface.co/VAST-AI/UniRig) → retarget to 52-bone `J_Bip_*` VRoid rig | Continuous skinned mesh (zero ankle, knee, waist, or wrist cuts) | Replaces brittle manual mesh slicing/splicing with learned volumetric skinning weights across the unbroken body, legs, feet, and sleeves. |
| **6. Topology & Normals** | Local Blender 5.3.0 | Boundary-pinned Taubin smoothing + 2-pass `DECIMATE` + `normals_split_custom_set()` | `<= 36,500`-face clean mesh (`0` non-manifold `>2` edges, `0` floaters) | Smooths voxel lumpiness while pinning open hems/cuffs; single unified `pass_index` across torso and skirt prevents artificial `exr_4b` waist ink seams. |
| **7. Flat Cel Albedo Bake** | Local Blender 5.3.0 / Python | Direct 4-view orthographic projection (`2048×2048`) + pull-push crevice fill | De-lit flat anime albedo with 2.5×–3× UV density on face | Faithfully copies all 4 turnaround sheets without single-view PBR hallucination ([Hunyuan3D-Paint 2.1](https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1/blob/main/hy3dpaint/README.md)) or sparse-voxel sampling holes. |

---

## Stage-by-Stage Engineering Specification

### Stage 1: Shared-Scale 4-View Preprocessing & Normal Priors (Colab A100)

1. Segment all 4 orthogonal turnaround views (`view_00_front`, `view_01_right`, `view_02_back`, `view_03_left`) with `BiRefNet`.
2. Compute a **single shared vertical span** `(global_y0, global_y1)` and shared square crop size across all 4 views, centering each view horizontally on the front/back spine axis so trailing hair or skirts never shift the `90°`/`270°` turntable center.
3. Run [StableNormal_turbo](https://github.com/Stable-X/StableNormal) (`torch.hub.load("Stable-X/StableNormal", "StableNormal_turbo", trust_repo=True)`) on the 4 normalized views to produce crisp camera-space normal targets for Stage 3.

### Stage 2: Multi-View Open-Cloth Generation (Colab A100)

1. Run **TRELLIS.2-4B** with multi-view sampler injection ([PR #104](https://github.com/microsoft/TRELLIS.2/pull/104) hybrid schedule wrapped in `try...finally`):
   - **Coarse `32³` sparse structure:** Equal-weight `multidiffusion` (`[0.25, 0.25, 0.25, 0.25]`) to enforce consistent 4-view hull depth.
   - **Cascade shape SLat (`512 → 1024/1536`):** Hybrid mode (`multidiffusion` for `t >= 0.55`, `stochastic` round-robin for `t < 0.55`) to prevent velocity-averaging blur while keeping sharp garment folds.
2. Extract the intermediate mesh via [o_voxel/postprocess.py](https://github.com/microsoft/TRELLIS.2/blob/main/o-voxel/o_voxel/postprocess.py) **Branch 1**:
   - Set **`remesh=False`** and **`decimation_target=120000`** (avoiding both the double-walled UDF shell of `remesh=True` and the low-target CuMesh CUDA crash in [Issue #84](https://github.com/microsoft/TRELLIS.2/issues/84)).
   - Branch 1 automatically runs the 5-step CuMesh topology repair (`fill_holes`, `remove_duplicate_faces`, `repair_non_manifold_edges`, `remove_small_connected_components`, `unify_face_orientations`) and sets `doubleSided=True`.

### Stage 3: Differentiable 4-View Silhouette & Normal Lock (Colab A100)

1. Run 100 iterations of **`nvdiffrast` + [largesteps](https://pypi.org/project/largesteps/)** directly on the extracted 120k mesh vertices (Y-up GLB space, rotating around the Y-axis for `0°`, `90°`, `180°`, `270°`).
2. Optimize:
   - `L1` silhouette loss (`w = 1.0`) against the 4 `BiRefNet` alpha masks.
   - Interior stop-gradient normal cosine loss (`w = 0.05`) against the 4 `StableNormal_turbo` normal maps.
   - Vertex displacement regularization (`w = 0.02`).
3. Ensure clip-space Y is flipped (`-v_rot[:, :, 1:2] * 2.0`) to match top-down image row order, and pass `faces.long()` to PyTorch `index_add_`.
4. **Export refined geometry directly via `trimesh`**—never re-sample sparse `o_voxel` `attr_volume` at displaced vertex coordinates (displaced vertices leave the original sparse voxel shell and bake black patches).

### Stage 4: Native 3D Part Segmentation & Occluded Completion (Colab A100)

1. Run **[P3-SAM (`tencent/Hunyuan3D-Part`)](https://huggingface.co/tencent/Hunyuan3D-Part)** directly on the silhouette-locked 120k 3D mesh to obtain clean per-face 3D part instance IDs (`p3sam_part_ids.npy`).
2. Classify each 3D instance into semantic roles (`head_prop`, `hair`, `face`, `torso_sleeves`, `skirt`, `shoes`, `body`) purely from 3D spatial bounds and connectivity—no 2D semantic raycasting needed.
3. Only if a long skirt occludes the upper boots/ankles, run **`X-Part`** (or [HoloPart](https://github.com/VAST-AI-Research/HoloPart)) on the segmented footwear component to complete the hidden ankle geometry cleanly in 3D.

### Stage 5: Neural Auto-Rigging & Continuous Body Binding (Colab A100)

**Core rule: Do not chop and splice ankles, knees, waist, or wrists onto a separate canonical mannequin.**

1. Run **[VAST-AI/UniRig](https://huggingface.co/VAST-AI/UniRig)** on the unbroken 3D character mesh to predict the skeleton and smooth volumetric skinning weights across the torso, bell sleeves, skirt, legs, ankles, and feet.
2. Retarget/rename `UniRig`'s predicted humanoid joints 1-to-1 onto your **52-bone `J_Bip_*` VRoid skeleton** so existing `walk_15` / `run_15` animations play directly.
3. Apply two lightweight semantic weight refinements using the Stage 4 `P3-SAM` part IDs:
   - **Rigid head props (`antlers`, `tiara`):** Bind `100%` to `J_Bip_C_Head` so props never flex.
   - **Outer skirt (`skirt`) & long rear hair (`hair`):** Smooth weights with 6 Laplacian iterations (`0.65 Hips + 0.35 nearest UpperLeg` on lower skirt hems; `Head → Neck → UpperChest → Chest` on long rear hair) so running strides never clip or tear.
   - **Inner thigh webbing check:** Only if left and right thighs physically touch across `x = 0`, split the tiny sagittal bridge so left/right legs separate cleanly during walk/run cycles—never delete legs, knees, ankles, or feet.

### Stage 6: Retopology, Face-Priority UVs & Cel Albedo Projection (Local Blender 5.3.0)

1. **Floater removal:** Delete disconnected micro-islands (`area < 2.0 cm²` or `diag < 1.5 cm`) across the whole mesh; embed genuine close ornaments (`<= 1.8 cm`) by `0.5 mm`.
2. **Boundary-pinned smoothing & 2-pass decimation:** Pin open garment borders (`v.is_boundary == True`), run 3 iterations of uniform Taubin smoothing (`0.35 / -0.32`), then `DECIMATE` Pass 1 (`DISSOLVE`, `2.5°`, `delimit={'UV', 'MATERIAL', 'SEAM'}`) and Pass 2 (`COLLAPSE`) to `<= 36,500` total faces.
3. **Face-priority UV layout:** Pack UVs with **2.5×–3× texel scale on the `P3-SAM` face island** so anime eyes and mouth stay razor-sharp at `512×768` sprite resolution without splicing a separate canonical `Head_Face`.
4. **Blender 5.3.0 shading & `exr_4b` continuity:** Set smooth shading via `mesh.polygons.foreach_set("use_smooth", [True] * len(mesh.polygons))`, remove any `sharp_edge` attribute, assign unified loop normals via `mesh.normals_split_custom_set()`, and keep one unified `pass_index` across torso, belt, and skirt.
5. **Direct 4-view flat albedo bake:** Project all 4 de-lit turnaround RGBA views onto the final `2048×2048` UV layout, using multi-scale pull-push inpainting strictly for occluded crevices (inner cuffs, armpits, under-skirt). Skip single-view PBR painters completely.

---

## Stage-by-Stage Verification Gates

| Stage Gate | Metric / Inspection | Pass Target |
| :--- | :--- | :--- |
| **After Stage 3 (Geometry)** | 4-view `bbox_aligned_silhouette_iou` on raw refined mesh | `>= 0.78` front/back, **`>= 0.75` left/right side**; open bell-sleeve cuffs intact. |
| **After Stage 4 (3D Parts)** | Visual color-coded 3D render of `P3-SAM` clusters | Clean separation of hair, face, bodice/sleeves, skirt, shoes, and antlers with zero bleed. |
| **After Stage 5 (Auto-Rig)** | `walk_00`, `walk_15`, `run_15` 3/4 renders | **0 severed ankles/feet**, **0 knee notches**, **0 waist tear**, clean L/R leg separation, rigid antlers. |
| **After Stage 6 (Hygiene & Bake)** | Strict mesh validator + turnaround visual audit | `<= 39,000` faces, `0` non-manifold `>2` edges, `0` degenerate/zero-UV faces, `0` floaters, `0` cross-part texture bleed. |
| **Stage B (Godot 4.7.2 Atlas)** | 60-frame `clean_cel` + `exr_4b` sprite benchmark | Zero internal waist ink lines, `Hygiene = 100.0`, Silhouette & Total scores matching hand-authored reference (`77.4` / `75.4`). |
