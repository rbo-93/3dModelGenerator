# Deep Research Plan: 360° Arbitrary-View Anime 3D Reconstruction & See-Through Semantic Layer Pipelines

## 1. Objective & Proven Baseline

Design the **highest-quality, license-agnostic, end-to-end 3D anime character pipeline** that turns 2D anime character art (generated from an **Anima** base model—either 4 orthogonal turnarounds or a 360° multi-view sequence) into a **watertight, cleanly rigged 3D character** (`<= 39,000` faces, `52`-bone `J_Bip_*` VRoid skeleton) rendered into **`512×768` cel-shaded Stage B sprites (`clean_cel` + `exr_4b`)** for **Godot 4.7.2**.

### What Already Worked in Our Latest Prototype (`valkyrie`)
- **Whole-mesh greenfield stack** (multi-view `TRELLIS.2-4B` → `P3-SAM` 3D part clustering → `UniRig` neural auto-rigging retargeted to 52-bone `J_Bip_*` → Blender 5.3.0 retopo + multi-view flat-albedo projection) achieved **`81.3` Stage B Total Score** (beating our `75.4` hand-authored `lauma_60f` reference), **`72.9` Silhouette Score**, **`84.4` Kinematics**, **`100.0` Hygiene**, `0` non-manifold edges, `0` floaters, and `0.82–0.87` 4-view IoU.
- **Remaining bottlenecks to solve in this research:**
  1. **Full-body voxel starvation on the face:** Generating a full 1.75 m character in a single `TRELLIS.2` pass leaves only ~40×40 voxels for the anime face, causing slight brow/cheek relief roughness.
  2. **Mesh holes & thin geometry erosion:** `o_voxel(remesh=False)` leaves micro-holes and erodes thin wing feathers/ribbons, whereas watertight remeshing can close open sleeves unless handled cleanly.
  3. **Inner-layer & occluded texture quality:** Projecting only 4 cardinal views (`0°, 90°, 180°, 270°`) cannot see inner layers (behind bangs, under pauldrons/skirts, inner arms) and causes cross-part bleed (skin projected onto bangs, hair/outline projected onto cheeks) when 3D geometry deviates by 2–5 mm from 2D art.

### Hard Constraints
- **Compute split:** Colab A100 (40/80 GB VRAM, PyTorch/CUDA) + local Windows **Blender 5.3.0** (`bpy`).
- **Target asset spec:** `<= 39,000` faces, `0` non-manifold `>2` edges, `0` degenerate/zero-UV faces, `0` floating debris, `52`-bone `J_Bip_*` VRoid armature, clean L/R leg separation in `walk`/`run`, zero per-character hardcoding.
- **License policy:** **Ignore licensing completely**—optimize strictly for maximum visual, geometric, and animation quality.

### Proven Anti-Patterns (Do NOT Recommend These)
1. **No destructive canonical VRoid limb/ankle/waist/wrist splicing:** Cutting AI legs, ankles, knees, waist, or wrists to stitch onto a canonical VRoid mannequin creates severed ankles, knee notches, waist tears, and wrist splinters. Keep the body geometry continuous.
2. **No 4-cardinal-view `nvdiffrast` silhouette carving:** Optimizing `L1` silhouette loss on only `0°, 90°, 180°, 270°` without diagonal views (`±45°`, `±135°`) mathematically collapses round heads and torsos into flat-walled 90° "Minecraft" boxes.
3. **No independent per-part image-to-3D generation:** Running `TRELLIS.2` / `Hunyuan3D` separately on cropped 2D PNGs (`hair.png`, `dress.png`, `body.png`) breaks shared 3D scale and depth alignment upon reassembly.
4. **No sparse `o_voxel` attribute re-sampling after vertex deformation:** Moving vertices outside the original sparse voxel coordinates bakes black texture patches.
5. **No single-view photorealistic PBR painters replacing flat multi-view cel albedo:** Stage B outputs flat `clean_cel` + `exr_4b` anime sprites; single-view PBR painters hallucinate back/side details and fight cel shading.

---

## 2. Research Workstreams

### Workstream 1: Rigid 360° Orbital View Generation from Anima (16–32 Consistent Views)
**Goal:** Determine the best way to generate **16–32 3D-consistent camera views** (full 360° azimuth including `±45°`/`±135°` diagonals, plus optional elevated/low camera rings) from an Anima 2D anime character—mimicking a real iPhone/LiDAR walk-around scan without multi-view drift.

1. **Survey 2025–2026 3D-consistent turntable / orbital generators for anime characters:**
   - Compare multi-view diffusion models (`MV-Adapter`, `Era3D`, `Unique3D`, `CharacterGen`, `CAT3D`-style / `Zero123++`) against image-to-video 360° turntable models (`Wan` / `SV3D` / `Kling` / `Seedance` / `HunyuanVideo` camera-orbit LoRAs) conditioned on 1 front Anima image or 4 orthogonal Anima views.
2. **Quantify multi-view geometric consistency:**
   - How much hair, armor, and facial drift occurs across 16–32 generated frames, and which camera trajectory (16 azimuths at 0° elevation vs. 16 azimuths + 8 elevated views at +20°/-15°) gives the best coverage of under-skirt, under-chin, and inner-sleeve layers?
   - Evaluate whether **view-consistent normal + RGB joint generation** improves downstream reconstruction over RGB-only turntables.

### Workstream 2: Unlimited-View ("iPhone-Scan") 3D Reconstruction vs. Native 3D Generators
**Goal:** Evaluate whether feeding 16–32 orbital anime views into **arbitrary-view 3D foundation models + neural surface reconstruction** beats multi-view `TRELLIS.2-4B` / `Hunyuan3D-2mv` on roundness, face detail, and hole-free thin geometry.

1. **Benchmark feed-forward N-view reconstructors (Colab A100):**
   - Evaluate **[VGGT](https://github.com/facebookresearch/vggt)**, **[MapAnything](https://github.com/facebookresearch/map-anything)**, and **[Depth Anything 3](https://github.com/bytedance-seed/depth-anything-3)** on 16–32 synthetic anime orbital views for camera pose recovery, multi-view depth accuracy, and robustness to flat cel-shaded/textureless anime skin (where classical `COLMAP` fails).
2. **Benchmark surface-accurate mesh extraction from N views:**
   - Compare **2D Gaussian Splatting (2DGS)**, **GOF (Gaussian Opacity Fields)**, **PGSR**, and **[MVG-Splatting](https://mvgsplatting.github.io/)** (TSDF / Marching Cubes mesh extraction) against **multi-view `TRELLIS.2-4B` (`remesh=True, remesh_project=0.9`)** and **16–32-view `nvdiffrast` + `largesteps` refinement** (where diagonal views eliminate the 4-view Minecraft-box visual hull).
3. **Failure-mode analysis:**
   - How tolerant is each reconstructor (`VGGT + 2DGS` vs. `TRELLIS.2` + 16-view refinement) to **minor 2D diffusion inconsistencies** (2–5% strand/strap shift between orbital frames)?
   - Which representation preserves thin anime accessories (winged headpieces, ribbons, open bell sleeves) with **zero mesh holes**?

### Workstream 3: 2D See-Through Semantic Decomposition (`SemanticSam` / Live2D-Style) for Inner-Layer Painting & Multi-Layer 3D
**Goal:** Investigate using a **2D see-through semantic decomposition model** (such as `24yearsold/l2d_sam_iter2` / Live2D layer splitters + 2D amodal inpainting) **before** 3D generation or texturing so the pipeline knows exactly which 2D layer/texture belongs to which 3D body part—solving inner-layer paint and cross-part bleed permanently.

1. **Approach 3A — Joint 3D Geometry + Semantic Layer-Routed Texturing (Immediate Practical Path):**
   - Decompose the 2D input views into clean, amodally inpainted semantic layers (`face_only` with bangs removed and forehead/cheeks inpainted, `hair_only`, `garment_only`, `body_only` with skirt/sleeves removed and limbs inpainted, `prop_only`).
   - Route each 2D layer strictly to its matching 3D part (segmented via **[P3-SAM / `tencent/Hunyuan3D-Part`](https://huggingface.co/tencent/Hunyuan3D-Part)**) during multi-view texture baking, paired with depth-discontinuity edge rejection and `dot(n, view_dir) >= 0.55` grazing-angle gating.
   - Identify the best open-weight 2D anime layer decomposition + amodal inpainting models (2025–2026) that produce clean, artifact-free hidden skin/hair/cloth layers across multiple angles.
2. **Approach 3B — Shared-Camera Multi-Layer 3D Reconstruction (True Layered 3D):**
   - Evaluate **[StdGEN](https://github.com/hyz317/StdGEN)** / `StdGEN++`, **[PartCrafter](https://github.com/wgsxm/PartCrafter)**, and **[Hunyuan3D-Part (`P3-SAM` + `X-Part`)](https://huggingface.co/tencent/Hunyuan3D-Part)** for reconstructing separate, non-clipping 3D `body`, `clothes`, and `hair` layers inside a **single locked 3D camera coordinate system** (from `VGGT` / `Depth Anything 3` or joint latents).
   - Compare geometric fidelity, cloth-to-body clearance, and Colab A100 runtime vs. Approach 3A.

### Workstream 4: High-Resolution Anime Face Fidelity & Artifact-Free Neural Auto-Rigging
**Goal:** Achieve crisp, artifact-free anime faces (both in `512×768` sprites and 3/4 3D viewer close-ups) and clean walk/run deformation with zero cross-leg stretching.

1. **Anime face resolution upgrade:**
   - Compare **16–32 view dense face reconstruction (`VGGT`/`2DGS`)** vs. **cropped high-res head-only 3D generation** (cropping views tightly to head+hair so the face gets ~400×400 voxels instead of 40×40, then seamlessly merging at the single neck collar ring) vs. **face-priority 3× UV density + layer-routed `face_only` projection**.
2. **Production rigging (`UniRig` → 52-bone `J_Bip_*`):**
   - Document best practices for **[VAST-AI/UniRig](https://huggingface.co/VAST-AI/UniRig)** retargeting to a 52-bone VRoid rig: preventing cross-leg weight leaks on close knee guards/boots, handling wide bell sleeves and layered skirts without calf poke-through in `run_15`, and binding `P3-SAM`-isolated head props (`wings`, `antlers`) rigidly to `J_Bip_C_Head`.

---

## 3. Required Deliverables & Acceptance Criteria

Ask the research synthesis to provide:
1. **End-to-End Architecture Decision Matrix:** Concrete comparison table across (a) `16–32 view Turntable + VGGT/DA3 + 2DGS/GOF`, (b) `Multi-View TRELLIS.2 (remesh=True, project=0.9) + P3-SAM + Layer-Routed Projection`, and (c) `Shared-Camera Multi-Layer 3D (StdGEN++ / X-Part)`—covering Colab A100 VRAM/runtime, inconsistency tolerance, hole-free thin geometry, inner-layer paint quality, and face crispness.
2. **Recommended Production Pipeline Blueprint:** Exact model checkpoints, data flow between Colab A100 and local Blender 5.3.0, and where 2D see-through semantic decomposition plugs in.
3. **Concrete Implementation Snippets & Pitfall Guards:** Exact API calls (verified against upstream repos, no unverified flags), depth-edge / layer-routed projection logic, and `UniRig` cross-leg weight clamping.
4. **Target Quantitative Gates (across `valkyrie`, `deer_elf`, and `lauma_ref`):**
   - Stage A 4-view `bbox_aligned_silhouette_iou >= 0.85` (all views) with round, organic 3/4 curvature.
   - `0` mesh holes, `0` non-manifold `>2` edges, `0` floaters, `<= 39,000` faces.
   - `0` cross-part texture bleed (face/bangs/inner layers) and `0` cross-leg skinning stretches in `walk_15`/`run_15`.
   - Stage B 60-frame sprite benchmark: **`Total Score >= 85.0`**, **`Silhouette Score >= 78.0`**, **`Hygiene = 100.0`**.
