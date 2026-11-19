# 3dModelGenerator — Greenfield Best-Practice Anime 3D & Cel Sprite Pipeline

Implements the end-to-end 7-stage pipeline specified in [docs/greenfield_best_practice_pipeline.md](docs/greenfield_best_practice_pipeline.md):

1. **Stage 1 (Colab GPU):** `BiRefNet` segmentation + shared-scale spine-centered 4-view normalization + `StableNormal_turbo` camera-space normals.
2. **Stage 2 (Colab GPU):** Multi-view `TRELLIS.2-4B` (PR #104 hybrid sampler) + `o_voxel` Branch 1 (`remesh=False`, `decimation_target=120000`, `doubleSided=True`).
3. **Stage 3 (Colab GPU):** 100-step `nvdiffrast` + `largesteps` (`Cholesky`) silhouette & normal lock, exported directly via `trimesh`.
4. **Stage 4 (Colab GPU):** Native 3D part isolation via `tencent/Hunyuan3D-Part` (`P3-SAM` + conditional `X-Part`/`HoloPart`).
5. **Stage 5 (Colab GPU):** Neural auto-rigging via `VAST-AI/UniRig` on the unbroken character mesh, retargeted 1-to-1 onto the 52-bone `J_Bip_*` VRoid skeleton with `P3-SAM` semantic weight refinements.
6. **Stage 6 (Local Blender 5.3.0):** Floater pruning, boundary-pinned Taubin smoothing, 2-pass `DECIMATE` (`<= 36,500` faces), face-priority UV layout, custom split normals + unified `pass_index`, and direct 4-view flat cel albedo projection (`2048x2048`) with pull-push crevice inpainting.
7. **Evaluation & Stage B Gates:** Turnaround IoU / Delta-E76 (`eval_greenfield_pipeline.py`), `walk_00` / `walk_15` / `run_15` `clean_cel` + `exr_4b` previews, and `prepare_ai_character.py --strict`.

## Usage

```powershell
# 1. Split a 3-view or 4-view turnaround sheet into 4 calibrated views:
..\2d_test_animate\.venv\Scripts\python.exe scripts\split_turnaround_sheet.py `
  --sheet <path_to_turnaround.jpg> --char-id valkyrie `
  --out-inputs-dir inputs\valkyrie_4view --out-fixture-dir eval_fixtures\valkyrie_4view

# 2. Run Colab L4 Stages 1-5 + Local Stage 6 + Eval end-to-end:
..\2d_test_animate\.venv\Scripts\python.exe scripts\run_greenfield_pipeline.py `
  --char-id valkyrie --mode full --gpu L4

# 3. Re-run Local Stage 6 + Eval from downloaded Colab Stage 1-5 artifacts:
..\2d_test_animate\.venv\Scripts\python.exe scripts\run_greenfield_pipeline.py `
  --char-id valkyrie --mode local-stage6
```
