#!/usr/bin/env python3
"""Colab L4/A100 GPU Worker for the Greenfield Best-Practice Pipeline (Stages 1-5).

Strictly implements the blueprint stack in docs/greenfield_best_practice_pipeline.md:
  - Stage 1: ZhengPeng7/BiRefNet + shared-scale spine-centered normalization + StableNormal_turbo
  - Stage 2: Multi-View TRELLIS.2-4B (PR #104 hybrid sampler) + o_voxel Branch 1 (remesh=False, 120k)
  - Stage 3: 100-step nvdiffrast + largesteps (Cholesky) silhouette & StableNormal lock -> trimesh GLB
  - Stage 4: Tencent-Hunyuan/Hunyuan3D-Part (P3-SAM + conditional X-Part/HoloPart) -> p3sam_part_ids.npy
  - Stage 5: VAST-AI/UniRig on unbroken mesh -> 1-to-1 52-bone J_Bip_* retarget + P3-SAM weights

Designed for headless `colab_exec_safe.py` execution (never touches google.colab.drive/userdata,
always sets MPLBACKEND=Agg and PYTHONUNBUFFERED=1).
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tarfile
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np
from PIL import Image

IN_COLAB = Path("/content").exists()
WORKSPACE_ROOT = Path(
    os.environ.get(
        "GREENFIELD_WORKSPACE",
        "/content/greenfield_workspace" if IN_COLAB else str(Path.cwd() / "greenfield_workspace"),
    )
)
TORCH_CACHE_DIR = WORKSPACE_ROOT / "torch_cache"
PIP_CACHE_DIR = WORKSPACE_ROOT / "pip_cache"
UV_CACHE_DIR = WORKSPACE_ROOT / "uv_cache"
HF_CACHE_DIR = Path(os.environ.get("HF_HOME") or (WORKSPACE_ROOT / "hf_cache"))
REPOS_DIR = Path("/content/repos") if IN_COLAB else (WORKSPACE_ROOT / "repos")
VENVS_DIR = Path("/content/venvs") if IN_COLAB else (WORKSPACE_ROOT / "venvs")

TRELLIS2_REPO = "https://github.com/microsoft/TRELLIS.2.git"
HUNYUAN3D_PART_REPO = "https://github.com/Tencent-Hunyuan/Hunyuan3D-Part.git"
UNIRIG_REPO = "https://github.com/VAST-AI-Research/UniRig.git"
HOLOPART_REPO = "https://github.com/VAST-AI-Research/HoloPart.git"

TRELLIS2_CUDA_EXTENSIONS: tuple[dict[str, Any], ...] = (
    {"pkg": "nvdiffrast", "import_name": "nvdiffrast", "repo": "https://github.com/NVlabs/nvdiffrast.git", "branch": "v0.4.0", "recursive": False, "cpp20": False},
    {"pkg": "nvdiffrec_render", "import_name": "nvdiffrec_render", "repo": "https://github.com/JeffreyXiang/nvdiffrec.git", "branch": "renderutils", "recursive": False, "cpp20": False},
    {"pkg": "cumesh", "import_name": "cumesh", "repo": "https://github.com/JeffreyXiang/CuMesh.git", "branch": None, "recursive": True, "cpp20": True},
    {"pkg": "flex_gemm", "import_name": "flex_gemm", "repo": "https://github.com/JeffreyXiang/FlexGEMM.git", "branch": None, "recursive": True, "cpp20": True},
    {"pkg": "o_voxel", "import_name": "o_voxel", "local_subpath": "o-voxel", "repo": None, "branch": None, "recursive": False, "cpp20": True},
)

ROLE_IDS: dict[str, int] = {
    "body": 0,
    "torso_sleeves": 1,
    "skirt": 2,
    "shoes": 3,
    "face": 4,
    "hair": 5,
    "head_prop": 6,
}


def setup_env_and_caches(wheel_cache_dir: Path | str | None = None) -> dict[str, Any]:
    """Configure caches, UTF-8/Agg/CUDA environment, and probe live GPU hardware."""
    resolved_wheel_dir = Path(wheel_cache_dir) if wheel_cache_dir else (WORKSPACE_ROOT / "wheel_cache")
    for d in (WORKSPACE_ROOT, HF_CACHE_DIR, TORCH_CACHE_DIR, PIP_CACHE_DIR, UV_CACHE_DIR, resolved_wheel_dir, REPOS_DIR, VENVS_DIR):
        d.mkdir(parents=True, exist_ok=True)

    os.environ["HF_HOME"] = str(HF_CACHE_DIR)
    os.environ["HUGGINGFACE_HUB_CACHE"] = str(HF_CACHE_DIR / "hub")
    os.environ["TORCH_HOME"] = str(TORCH_CACHE_DIR)
    os.environ["PIP_CACHE_DIR"] = str(PIP_CACHE_DIR)
    os.environ["UV_CACHE_DIR"] = str(UV_CACHE_DIR)
    os.environ["GREENFIELD_WHEEL_CACHE"] = str(resolved_wheel_dir)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["MPLBACKEND"] = "Agg"
    os.environ["PYTHONUNBUFFERED"] = "1"
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    cuda_candidates = [Path("/usr/local/cuda")] + sorted(Path("/usr/local").glob("cuda-12*"), reverse=True)
    cuda_home = next((p for p in cuda_candidates if (p / "bin" / "nvcc").is_file()), None)
    if cuda_home is not None:
        os.environ["CUDA_HOME"] = str(cuda_home)
        os.environ["PATH"] = f"{cuda_home / 'bin'}:{os.environ.get('PATH', '')}"

    if not os.environ.get("HF_TOKEN"):
        for tf in (WORKSPACE_ROOT / "hf_token", Path.home() / ".cache" / "huggingface" / "token"):
            if tf.is_file():
                tok = tf.read_text(encoding="utf-8-sig").strip()
                if "=" in tok:
                    tok = tok.split("=", 1)[1].strip()
                tok = tok.strip("\"'")
                if tok:
                    os.environ["HF_TOKEN"] = tok
                    break
    if os.environ.get("HF_TOKEN"):
        os.environ.setdefault("HUGGING_FACE_HUB_TOKEN", os.environ["HF_TOKEN"])

    gpu_name, vram_gb, sm_arch = "CPU_ONLY", 0.0, "0.0"
    try:
        smi = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,memory.total,compute_cap", "--format=csv,noheader,nounits"],
            text=True,
        ).strip()
        if smi:
            parts = [p.strip() for p in smi.splitlines()[0].split(",")]
            gpu_name = parts[0]
            vram_gb = float(parts[1]) / 1024.0
            if len(parts) >= 3:
                sm_arch = parts[2]
    except Exception:
        pass

    return {
        "gpu_name": gpu_name,
        "vram_gb": round(vram_gb, 2),
        "sm_arch": sm_arch,
        "cuda_home": os.environ.get("CUDA_HOME", ""),
        "wheel_cache_dir": str(resolved_wheel_dir),
        "hf_cache_dir": str(HF_CACHE_DIR),
    }


def _resolve_uv() -> str:
    uv_bin = shutil.which("uv")
    if not uv_bin:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "uv"], check=True)
        uv_bin = shutil.which("uv")
    return uv_bin or "uv"


def _run(cmd: list[str], cwd: Path | None = None, env_extra: dict[str, str] | None = None) -> None:
    env = os.environ.copy()
    env["MPLBACKEND"] = "Agg"
    env["PYTHONUNBUFFERED"] = "1"
    if cwd is not None:
        prev = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = f"{cwd}{os.pathsep}{prev}" if prev else str(cwd)
    if env_extra:
        env.update(env_extra)
    print(f"$ {' '.join(str(x) for x in cmd)}", flush=True)
    subprocess.run(cmd, cwd=str(cwd) if cwd else None, env=env, check=True)


def ensure_uv_env(
    name: str,
    repo_url: str | None,
    *,
    python_version: str = "3.10",
    clone_recursive: bool = False,
    install_cmds: list[list[str]] | None = None,
    post_install_fn: Callable[[Path, Path | None], Any] | None = None,
) -> tuple[Path, Path | None]:
    """Create or reuse a pinned `uv` venv and optional git clone with `.installed.ok` marker."""
    uv_bin = _resolve_uv()
    repo_dir = (REPOS_DIR / name) if repo_url else None
    venv_dir = VENVS_DIR / name
    py_bin = venv_dir / "bin" / "python"
    marker = venv_dir / ".installed.ok"

    if repo_dir is not None and not repo_dir.exists():
        c_cmd = ["git", "clone", "--depth", "1"]
        if clone_recursive:
            c_cmd.append("--recursive")
        c_cmd.extend([repo_url, str(repo_dir)])
        _run(c_cmd)

    if not marker.is_file():
        if not py_bin.exists():
            _run([uv_bin, "venv", "--python", python_version, str(venv_dir)])
        _run([uv_bin, "pip", "install", "--python", str(py_bin), "setuptools<81", "wheel", "pip"])
        for tpl in install_cmds or []:
            cmd = [tok.replace("{PY}", str(py_bin)).replace("{UV}", uv_bin) for tok in tpl]
            _run(cmd, cwd=repo_dir)
        if post_install_fn is not None:
            post_install_fn(py_bin, repo_dir)
        marker.write_text("ok\n", encoding="utf-8")

    return py_bin, repo_dir


def resolve_headless_blender(blender_version: str = "4.2.3") -> Path:
    """Locate or install portable headless Linux Blender on Colab VM for Stage 5 retargeting."""
    sys_blender = shutil.which("blender")
    if sys_blender:
        return Path(sys_blender)
    tools_dir = WORKSPACE_ROOT / "tools"
    tools_dir.mkdir(parents=True, exist_ok=True)
    blender_bin = tools_dir / f"blender-{blender_version}-linux-x64" / "blender"
    if not blender_bin.is_file():
        tar_url = f"https://download.blender.org/release/Blender4.2/blender-{blender_version}-linux-x64.tar.xz"
        tar_path = tools_dir / f"blender-{blender_version}-linux-x64.tar.xz"
        _run(["wget", "-q", "-O", str(tar_path), tar_url])
        _run(["tar", "-xf", str(tar_path), "-C", str(tools_dir)])
    probe = subprocess.run([str(blender_bin), "--version"], capture_output=True, text=True)
    if probe.returncode != 0 and shutil.which("apt-get"):
        _run(
            [
                "bash",
                "-lc",
                "apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq "
                "libx11-6 libxi6 libxrender1 libxxf86vm1 libxfixes3 libgl1 libxkbcommon0 libsm6",
            ]
        )
    return blender_bin


def _write_status(status_path: Path | None, state: str, step: str, detail: dict[str, Any] | None = None) -> None:
    if status_path is None:
        return
    payload = {
        "state": state,
        "step": step,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        **(detail or {}),
    }
    status_path.parent.mkdir(parents=True, exist_ok=True)
    status_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


# ---------------------------------------------------------------------------
# Pure-Python Multi-View Turnaround Normalization
# ---------------------------------------------------------------------------
def preprocess_multiview_turnaround(
    view_paths: dict[str, Path | str],
    out_dir: Path,
    *,
    target_size: int = 1024,
    fill_frac: float = 0.88,
) -> dict[str, Any]:
    """Normalize 4 turnaround RGBA views onto a shared-scale, spine-centered square canvas.

    Computes a single global vertical span `(global_y0, global_y1)` across all 4 views and
    anchors horizontal centering to the front/back spine axis so wide front/back views and
    narrow side views remain 1:1 locked in height and turntable axis.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    loaded: dict[str, np.ndarray] = {}
    fgs: dict[str, np.ndarray] = {}
    bboxes: dict[str, tuple[int, int, int, int]] = {}

    for role, vp in view_paths.items():
        p = Path(vp)
        if not p.is_file():
            continue
        arr = np.asarray(Image.open(p).convert("RGBA"))
        fg = arr[:, :, 3] > 128
        ys, xs = np.where(fg)
        if len(ys) == 0:
            continue
        loaded[role] = arr
        fgs[role] = fg
        bboxes[role] = (int(xs.min()), int(xs.max()) + 1, int(ys.min()), int(ys.max()) + 1)

    if not loaded:
        raise ValueError("No valid foreground RGBA views supplied to preprocess_multiview_turnaround")

    global_y0 = min(b[2] for b in bboxes.values())
    global_y1 = max(b[3] for b in bboxes.values())
    global_h = max(1, global_y1 - global_y0)
    max_w = max(b[1] - b[0] for b in bboxes.values())

    shared_box_size = int(math.ceil(max(global_h, max_w) / max(fill_frac, 0.1)))
    if shared_box_size % 2 != 0:
        shared_box_size += 1
    half = shared_box_size // 2
    cy = int(round(0.5 * (global_y0 + global_y1)))

    ref_role = "front" if "front" in loaded else ("back" if "back" in loaded else next(iter(loaded)))
    ref_w = loaded[ref_role].shape[1]
    fb_cxs = [0.5 * (bboxes[r][0] + bboxes[r][1]) for r in ("front", "back") if r in bboxes and loaded[r].shape[1] == ref_w]
    spine_cx = int(round(sum(fb_cxs) / len(fb_cxs))) if fb_cxs else int(round(0.5 * (bboxes[ref_role][0] + bboxes[ref_role][1])))

    sq_paths: dict[str, str] = {}
    per_view_meta: dict[str, Any] = {}
    for role, arr in loaded.items():
        h_v, w_v = arr.shape[:2]
        if w_v == ref_w:
            cx = spine_cx
        elif role in ("front", "back"):
            cx = int(round(0.5 * (bboxes[role][0] + bboxes[role][1])))
        else:
            fg = fgs[role]
            y0_v, y1_v = bboxes[role][2], bboxes[role][3]
            span_v = max(10, y1_v - y0_v)
            band_mask = np.zeros_like(fg)
            band_mask[y0_v : min(h_v, y0_v + int(0.15 * span_v)), :] = True
            band_mask[max(0, y1_v - int(0.15 * span_v)) : y1_v, :] = True
            spine_xs = np.where(fg & band_mask)[1]
            cx = int(round(float(np.median(spine_xs)))) if len(spine_xs) else (w_v // 2)

        x0 = cx - half
        y0 = cy - half
        x1 = cx + half
        y1 = cy + half

        canvas = np.zeros((shared_box_size, shared_box_size, 4), dtype=np.uint8)
        sx0 = max(0, x0)
        sy0 = max(0, y0)
        sx1 = min(w_v, x1)
        sy1 = min(h_v, y1)
        if sx1 > sx0 and sy1 > sy0:
            dx0 = sx0 - x0
            dy0 = sy0 - y0
            canvas[dy0 : dy0 + (sy1 - sy0), dx0 : dx0 + (sx1 - sx0)] = arr[sy0:sy1, sx0:sx1]

        out_im = Image.fromarray(canvas, mode="RGBA").resize((target_size, target_size), Image.Resampling.LANCZOS)
        out_p = out_dir / f"view_{ {'front':'00','right':'01','back':'02','left':'03'}[role] }_{role}_sq.png"
        out_im.save(out_p)
        sq_paths[role] = str(out_p)
        per_view_meta[role] = {"cx": cx, "cy": cy, "src_size": [w_v, h_v]}

    meta = {
        "target_size": target_size,
        "shared_box_size": shared_box_size,
        "global_y0": global_y0,
        "global_y1": global_y1,
        "spine_cx": spine_cx,
        "sq_paths": sq_paths,
        "views": per_view_meta,
    }
    (out_dir / "sq_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta


# ---------------------------------------------------------------------------
# Sub-Scripts Run Inside Isolated UV Environments
# ---------------------------------------------------------------------------
_STAGE1_BIREFNET_NORMAL_SCRIPT = r'''
import argparse, gc, json
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from torchvision import transforms
from transformers import AutoModelForImageSegmentation

ap = argparse.ArgumentParser()
ap.add_argument("--inputs-json", required=True)
ap.add_argument("--cutout-dir", required=True)
ap.add_argument("--sq-dir", required=True)
ap.add_argument("--normal-dir", required=True)
args = ap.parse_args()

views_cfg = json.loads(Path(args.inputs_json).read_text(encoding="utf-8"))
cutout_dir = Path(args.cutout_dir)
sq_dir = Path(args.sq_dir)
normal_dir = Path(args.normal_dir)
for d in (cutout_dir, sq_dir, normal_dir):
    d.mkdir(parents=True, exist_ok=True)

# 1A. BiRefNet foreground segmentation
birefnet = AutoModelForImageSegmentation.from_pretrained("ZhengPeng7/BiRefNet", trust_remote_code=True).to("cuda").eval()
tfm = transforms.Compose([
    transforms.Resize((1024, 1024)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

cutouts = {}
for role in ("front", "right", "back", "left"):
    src_p = Path(views_cfg[role])
    im = Image.open(src_p).convert("RGBA")
    rgb = Image.alpha_composite(Image.new("RGBA", im.size, (255, 255, 255, 255)), im).convert("RGB")
    arr = np.asarray(im)
    corners_trans = (
        arr[0, 0, 3] < 32 and arr[0, -1, 3] < 32 and arr[-1, 0, 3] < 32 and arr[-1, -1, 3] < 32
    )
    out_p = cutout_dir / f"{src_p.stem}_rgba.png"
    if corners_trans:
        im.save(out_p)
    else:
        with torch.no_grad():
            pred = birefnet(tfm(rgb).unsqueeze(0).to("cuda"))[-1].sigmoid().cpu()[0].squeeze()
        mask_pil = transforms.ToPILImage()(pred).resize(im.size, Image.BILINEAR)
        out_rgba = rgb.convert("RGBA")
        out_rgba.putalpha(mask_pil)
        out_rgba.save(out_p)
    cutouts[role] = str(out_p)

del birefnet
gc.collect()
torch.cuda.empty_cache()

# 1B. Shared-scale spine-centered normalization
import math
loaded, fgs, bboxes = {}, {}, {}
for role, vp in cutouts.items():
    arr = np.asarray(Image.open(vp).convert("RGBA"))
    fg = arr[:, :, 3] > 128
    ys, xs = np.where(fg)
    loaded[role] = arr
    fgs[role] = fg
    bboxes[role] = (int(xs.min()), int(xs.max()) + 1, int(ys.min()), int(ys.max()) + 1)

global_y0 = min(b[2] for b in bboxes.values())
global_y1 = max(b[3] for b in bboxes.values())
global_h = max(1, global_y1 - global_y0)
max_w = max(b[1] - b[0] for b in bboxes.values())
shared_box_size = int(math.ceil(max(global_h, max_w) / 0.88))
if shared_box_size % 2 != 0:
    shared_box_size += 1
half = shared_box_size // 2
cy = int(round(0.5 * (global_y0 + global_y1)))

ref_w = loaded["front"].shape[1]
fb_cxs = [0.5 * (bboxes[r][0] + bboxes[r][1]) for r in ("front", "back") if loaded[r].shape[1] == ref_w]
spine_cx = int(round(sum(fb_cxs) / len(fb_cxs)))

sq_paths = {}
for role in ("front", "right", "back", "left"):
    arr = loaded[role]
    h_v, w_v = arr.shape[:2]
    cx = spine_cx if w_v == ref_w else int(round(0.5 * (bboxes[role][0] + bboxes[role][1])))
    x0, y0, x1, y1 = cx - half, cy - half, cx + half, cy + half
    canvas = np.zeros((shared_box_size, shared_box_size, 4), dtype=np.uint8)
    sx0, sy0, sx1, sy1 = max(0, x0), max(0, y0), min(w_v, x1), min(h_v, y1)
    if sx1 > sx0 and sy1 > sy0:
        canvas[sy0 - y0 : sy0 - y0 + (sy1 - sy0), sx0 - x0 : sx0 - x0 + (sx1 - sx0)] = arr[sy0:sy1, sx0:sx1]
    out_im = Image.fromarray(canvas, mode="RGBA").resize((1024, 1024), Image.Resampling.LANCZOS)
    idx_tag = {"front": "00", "right": "01", "back": "02", "left": "03"}[role]
    out_p = sq_dir / f"view_{idx_tag}_{role}_sq.png"
    out_im.save(out_p)
    sq_paths[role] = str(out_p)

(sq_dir / "sq_meta.json").write_text(json.dumps({
    "shared_box_size": shared_box_size,
    "global_y0": global_y0,
    "global_y1": global_y1,
    "spine_cx": spine_cx,
    "cutouts": cutouts,
    "sq_paths": sq_paths,
}, indent=2), encoding="utf-8")

# 1C. StableNormal_turbo camera-space normal priors on normalized 4 views
predictor = torch.hub.load("Stable-X/StableNormal", "StableNormal_turbo", trust_repo=True)
normal_paths = {}
for role in ("front", "right", "back", "left"):
    idx_tag = {"front": "00", "right": "01", "back": "02", "left": "03"}[role]
    sq_im = Image.open(sq_paths[role]).convert("RGBA")
    try:
        n_pil = predictor(sq_im, data_type="object")
    except TypeError:
        white_rgb = Image.alpha_composite(Image.new("RGBA", sq_im.size, (255, 255, 255, 255)), sq_im).convert("RGB")
        n_pil = predictor(white_rgb)
    if not isinstance(n_pil, Image.Image):
        n_pil = Image.fromarray(np.asarray(n_pil, dtype=np.uint8))
    n_pil = n_pil.convert("RGB").resize((1024, 1024), Image.Resampling.BILINEAR)
    n_png = normal_dir / f"normal_{idx_tag}_{role}.png"
    n_pil.save(n_png)

    # Convert [0, 255] -> [-1, 1] unit camera-space normal masked by alpha
    a_mask = np.asarray(sq_im)[:, :, 3] > 128
    n_vec = (np.asarray(n_pil, dtype=np.float32) / 127.5) - 1.0
    n_len = np.maximum(np.linalg.norm(n_vec, axis=-1, keepdims=True), 1e-6)
    n_vec = n_vec / n_len
    n_vec[~a_mask] = 0.0
    n_npy = normal_dir / f"normal_{idx_tag}_{role}.npy"
    np.save(n_npy, n_vec.astype(np.float32))
    normal_paths[role] = {"png": str(n_png), "npy": str(n_npy)}

(normal_dir / "normals_meta.json").write_text(json.dumps(normal_paths, indent=2), encoding="utf-8")
print("STAGE1_OK", flush=True)
'''


_STAGE2_TRELLIS2_SCRIPT = r'''
import argparse, gc, inspect, json, os
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from transformers import DINOv3ViTModel

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ.setdefault("ATTN_BACKEND", "xformers")
os.environ.setdefault("SPARSE_ATTN_BACKEND", "xformers")

from trellis2.pipelines import Trellis2ImageTo3DPipeline
from trellis2.pipelines.samplers import FlowEulerSampler
import o_voxel

# Fallback for gated DINOv3 model + transformers>=5 layer attribute
_orig_dinov3_from_pretrained = DINOv3ViTModel.from_pretrained.__func__
@classmethod
def _safe_dinov3_from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
    try:
        m = _orig_dinov3_from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs)
    except Exception as exc:
        fallback_repo = os.environ.get("DINOV3_FALLBACK_REPO", "camenduru/dinov3-vitl16-pretrain-lvd1689m")
        print(f"[TRELLIS.2] {pretrained_model_name_or_path} unavailable ({exc}); loading {fallback_repo}", flush=True)
        m = _orig_dinov3_from_pretrained(cls, fallback_repo, *model_args, **kwargs)
    if not hasattr(m, "layer") and hasattr(m, "model") and hasattr(m.model, "layer"):
        m.layer = m.model.layer
    return m
DINOv3ViTModel.from_pretrained = _safe_dinov3_from_pretrained


def _slice_cond(cond, idx: int):
    if isinstance(cond, torch.Tensor):
        return cond[idx : idx + 1]
    if isinstance(cond, dict):
        return {k: (v[idx : idx + 1] if isinstance(v, torch.Tensor) and v.shape[0] > 1 else v) for k, v in cond.items()}
    return cond


def _cond_batch_len(cond) -> int:
    if isinstance(cond, torch.Tensor):
        return int(cond.shape[0])
    if isinstance(cond, dict):
        for v in cond.values():
            if isinstance(v, torch.Tensor):
                return int(v.shape[0])
    return 1


def _is_uniform_cond(cond) -> bool:
    if isinstance(cond, torch.Tensor) and cond.shape[0] > 1:
        return bool(torch.allclose(cond[0], cond[1], atol=1e-5))
    return False


_ORIG_FLOW_EULER_BASE_INFERENCE = FlowEulerSampler._inference_model


def _dispatched_flow_euler_base_inference(self, model, x_t, t, cond=None, **kwargs):
    hook = getattr(self, "_mv_base_inference", None)
    if hook is not None:
        return hook(model, x_t, t, cond=cond, **kwargs)
    return _ORIG_FLOW_EULER_BASE_INFERENCE(self, model, x_t, t, cond=cond, **kwargs)


FlowEulerSampler._inference_model = _dispatched_flow_euler_base_inference


def inject_trellis2_sampler_multi_image(sampler, weights: list[float], mode: str = "hybrid", hybrid_split_t: float = 0.55):
    """Hook FlowEulerSampler._inference_model below CFG/GuidanceInterval mixins for PR #104 multi-view fusion."""
    prev_hook = getattr(sampler, "_mv_base_inference", None)
    state = {"step_idx": 0}
    w_norm = [float(w) for w in weights]
    w_sum = max(sum(w_norm), 1e-6)
    w_norm = [w / w_sum for w in w_norm]

    def _mv_inference(self_sampler, model, x_t, t, cond=None, **kwargs):
        n_views = _cond_batch_len(cond)
        if cond is None or n_views <= 1:
            return _ORIG_FLOW_EULER_BASE_INFERENCE(self_sampler, model, x_t, t, cond=cond, **kwargs)
        if _is_uniform_cond(cond):
            # Identical negative conditioning tensor: evaluate once cleanly
            return _ORIG_FLOW_EULER_BASE_INFERENCE(self_sampler, model, x_t, t, cond=_slice_cond(cond, 0), **kwargs)

        t_val = float(t.item() if isinstance(t, torch.Tensor) else t)
        eff_mode = mode
        if mode == "hybrid":
            eff_mode = "multidiffusion" if t_val >= hybrid_split_t else "stochastic"

        if eff_mode == "stochastic":
            idx = state["step_idx"] % n_views
            state["step_idx"] += 1
            return _ORIG_FLOW_EULER_BASE_INFERENCE(self_sampler, model, x_t, t, cond=_slice_cond(cond, idx), **kwargs)

        preds = []
        for i in range(n_views):
            p_i = _ORIG_FLOW_EULER_BASE_INFERENCE(self_sampler, model, x_t, t, cond=_slice_cond(cond, i), **kwargs)
            w_i = w_norm[i] if i < len(w_norm) else (1.0 / n_views)
            preds.append(p_i * w_i)
        out = preds[0]
        for p_i in preds[1:]:
            out = out + p_i
        return out

    sampler._mv_base_inference = _mv_inference.__get__(sampler, type(sampler))

    def _restore():
        if prev_hook is None:
            if hasattr(sampler, "_mv_base_inference"):
                delattr(sampler, "_mv_base_inference")
        else:
            sampler._mv_base_inference = prev_hook

    return _restore


ap = argparse.ArgumentParser()
ap.add_argument("--sq-meta", required=True)
ap.add_argument("--out-glb", required=True)
ap.add_argument("--pipeline-type", default="1024_cascade")
ap.add_argument("--decimation-target", type=int, default=120000)
ap.add_argument("--seed", type=int, default=42)
args = ap.parse_args()

sq_meta = json.loads(Path(args.sq_meta).read_text(encoding="utf-8"))
sq_paths = sq_meta["sq_paths"]
mv_imgs = [Image.open(sq_paths[r]).convert("RGBA") for r in ("front", "right", "back", "left")]

pipeline = Trellis2ImageTo3DPipeline.from_pretrained("microsoft/TRELLIS.2-4B")
pipeline.cuda()

# Run pipeline.run with monkey-patched get_cond (4 normalized views) & PR #104 samplers inside try...finally
orig_get_cond = pipeline.get_cond
mv_imgs_pre = [pipeline.preprocess_image(im) for im in mv_imgs]

def _mv_get_cond(image, resolution: int, include_neg_cond: bool = True):
    return orig_get_cond(mv_imgs_pre, resolution, include_neg_cond=include_neg_cond)

restores = []
try:
    pipeline.get_cond = _mv_get_cond
    restores.append(inject_trellis2_sampler_multi_image(pipeline.sparse_structure_sampler, [0.25, 0.25, 0.25, 0.25], mode="multidiffusion"))
    restores.append(inject_trellis2_sampler_multi_image(pipeline.shape_slat_sampler, [0.25, 0.25, 0.25, 0.25], mode="hybrid", hybrid_split_t=0.55))
    restores.append(inject_trellis2_sampler_multi_image(pipeline.tex_slat_sampler, [0.25, 0.25, 0.25, 0.25], mode="stochastic"))
    out_meshes = pipeline.run(
        mv_imgs_pre[0],
        seed=int(args.seed),
        pipeline_type=args.pipeline_type,
        preprocess_image=False,
    )
finally:
    pipeline.get_cond = orig_get_cond
    for r_fn in restores:
        try:
            r_fn()
        except Exception:
            pass

mesh = out_meshes[0]
mesh.simplify(16777216)

# Branch 1 in o_voxel/postprocess.py: remesh=False preserves single-layer open cuffs/skirts + CuMesh repair + doubleSided=True
glb = o_voxel.postprocess.to_glb(
    vertices=mesh.vertices,
    faces=mesh.faces,
    attr_volume=mesh.attrs,
    coords=mesh.coords,
    attr_layout=mesh.layout,
    voxel_size=mesh.voxel_size,
    aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
    decimation_target=int(args.decimation_target),
    texture_size=2048,
    remesh=False,
    verbose=True,
)
out_p = Path(args.out_glb)
out_p.parent.mkdir(parents=True, exist_ok=True)
glb.export(str(out_p))
print(f"STAGE2_TRELLIS2_OK:{out_p} faces={len(glb.faces) if hasattr(glb, 'faces') else 'scene'}", flush=True)
'''


_STAGE3_NVDIFFRAST_SCRIPT = r'''
import argparse, json, math, os
from pathlib import Path
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
import trimesh
import nvdiffrast.torch as dr
import largesteps.geometry as lg
from largesteps.parameterize import to_differential, from_differential
from largesteps.optimize import AdamUniform


def _y_rot_mat(az_deg: float, device: torch.device) -> torch.Tensor:
    """World (Y-up, +Z front, +X char-left) -> Camera [right, up, toward]."""
    th = math.radians(az_deg)
    c, s = math.cos(th), math.sin(th)
    return torch.tensor([
        [ c, 0.0, -s],
        [0.0, 1.0, 0.0],
        [ s, 0.0,  c],
    ], dtype=torch.float32, device=device)


def _compute_vertex_normals(verts: torch.Tensor, faces_long: torch.Tensor) -> torch.Tensor:
    v0 = verts[faces_long[:, 0]]
    v1 = verts[faces_long[:, 1]]
    v2 = verts[faces_long[:, 2]]
    fn = torch.cross(v1 - v0, v2 - v0, dim=-1)
    vn = torch.zeros_like(verts)
    for k in range(3):
        vn.index_add_(0, faces_long[:, k], fn)
    return F.normalize(vn, dim=-1, eps=1e-6)


def _render_views(glctx, verts: torch.Tensor, vn: torch.Tensor, faces_i32: torch.Tensor, faces_long: torch.Tensor, cam_rots: list[torch.Tensor], res: int = 512):
    sils, norms, covs = [], [], []
    ones = torch.ones((verts.shape[0], 1), dtype=torch.float32, device=verts.device)
    for R in cam_rots:
        v_cam = verts @ R.T
        n_cam = vn @ R.T
        # Clip Y is negated so +Y (head) maps to clip_y = -1 (top row 0)
        clip = torch.cat([
            v_cam[:, 0:1] * 2.0,
            -v_cam[:, 1:2] * 2.0,
            -v_cam[:, 2:3] * 0.5,
            ones,
        ], dim=-1).unsqueeze(0).contiguous()
        rast, _ = dr.rasterize(glctx, clip, faces_i32, resolution=[res, res])
        cov = (rast[0, :, :, 3] > 0).float()
        sil = dr.antialias(cov.unsqueeze(0).unsqueeze(-1).contiguous(), rast, clip, faces_i32)
        n_interp, _ = dr.interpolate(n_cam.unsqueeze(0).contiguous(), rast, faces_long.to(torch.int32))
        n_interp = F.normalize(n_interp[0], dim=-1, eps=1e-6)
        sils.append(sil[0, :, :, 0])
        norms.append(n_interp)
        covs.append(cov)
    return sils, norms, covs


def _calc_ious(pred_sils, tgt_alphas) -> dict[str, float]:
    roles = ("front", "right", "back", "left")
    out = {}
    for role, p, t in zip(roles, pred_sils, tgt_alphas):
        pm = p > 0.5
        tm = t > 0.5
        inter = float((pm & tm).sum().item())
        union = max(1.0, float((pm | tm).sum().item()))
        out[role] = round(inter / union, 4)
    return out


def _calc_bbox_aligned_ious(pred_sils, tgt_alphas) -> dict[str, float]:
    roles = ("front", "right", "back", "left")
    out = {}
    for role, p, t in zip(roles, pred_sils, tgt_alphas):
        pm = (p > 0.5).float()
        tm = (t > 0.5).float()
        py, px = torch.where(pm > 0.5)
        ty, tx = torch.where(tm > 0.5)
        if len(py) == 0 or len(ty) == 0:
            out[role] = 0.0
            continue
        p_crop = pm[py.min() : py.max() + 1, px.min() : px.max() + 1][None, None]
        t_crop = tm[ty.min() : ty.max() + 1, tx.min() : tx.max() + 1][None, None]
        p_box = F.interpolate(p_crop, size=(256, 256), mode="bilinear", align_corners=False)[0, 0] > 0.5
        t_box = F.interpolate(t_crop, size=(256, 256), mode="bilinear", align_corners=False)[0, 0] > 0.5
        inter = float((p_box & t_box).sum().item())
        union = max(1.0, float((p_box | t_box).sum().item()))
        out[role] = round(inter / union, 4)
    return out


ap = argparse.ArgumentParser()
ap.add_argument("--in-glb", required=True)
ap.add_argument("--sq-meta", required=True)
ap.add_argument("--normals-meta", required=True)
ap.add_argument("--out-glb", required=True)
ap.add_argument("--out-metrics", required=True)
ap.add_argument("--out-grid", required=True)
ap.add_argument("--iters", type=int, default=100)
args = ap.parse_args()

device = torch.device("cuda")
sq_meta = json.loads(Path(args.sq_meta).read_text(encoding="utf-8"))
norm_meta = json.loads(Path(args.normals_meta).read_text(encoding="utf-8"))
sq_paths = sq_meta["sq_paths"]

roles = ("front", "right", "back", "left")
azimuths = (0.0, 90.0, 180.0, 270.0)
cam_rots = [_y_rot_mat(az, device) for az in azimuths]

tgt_alphas = []
tgt_norms = []
res_opt = 512
for r in roles:
    im = Image.open(sq_paths[r]).convert("RGBA").resize((res_opt, res_opt), Image.Resampling.BILINEAR)
    a_np = np.asarray(im, dtype=np.float32)[:, :, 3] / 255.0
    tgt_alphas.append(torch.tensor(a_np, dtype=torch.float32, device=device))

    n_np = np.load(norm_meta[r]["npy"])
    n_t = torch.tensor(n_np, dtype=torch.float32, device=device).permute(2, 0, 1).unsqueeze(0)
    n_ds = F.interpolate(n_t, size=(res_opt, res_opt), mode="bilinear", align_corners=False)[0].permute(1, 2, 0)
    n_ds = F.normalize(n_ds, dim=-1, eps=1e-6)
    tgt_norms.append(n_ds)

# Target height & 3D center in normalized [-0.5, 0.5] square space
front_fg = tgt_alphas[0] > 0.5
ys_f, xs_f = torch.where(front_fg)
tgt_y_top = 0.5 - (float(ys_f.min().item()) / res_opt)
tgt_y_bot = 0.5 - (float(ys_f.max().item()) / res_opt)
tgt_h = max(0.1, tgt_y_top - tgt_y_bot)
tgt_y_mid = 0.5 * (tgt_y_top + tgt_y_bot)
tgt_x_mid = (0.5 * (float(xs_f.min().item()) + float(xs_f.max().item())) / res_opt) - 0.5
right_fg = tgt_alphas[1] > 0.5
_, xs_r = torch.where(right_fg)
tgt_z_mid = -((0.5 * (float(xs_r.min().item()) + float(xs_r.max().item())) / res_opt) - 0.5) if len(xs_r) else 0.0

mesh = trimesh.load(args.in_glb, force="mesh", process=False)
mesh.merge_vertices(merge_tex=True, merge_norm=True)
ccs = mesh.split(only_watertight=False)
if len(ccs) > 1:
    max_f = max(len(c.faces) for c in ccs)
    kept = [c for c in ccs if len(c.faces) >= max(500, int(0.005 * max_f))]
    if kept:
        mesh = trimesh.util.concatenate(kept)
        mesh.merge_vertices(merge_tex=True, merge_norm=True)
verts_np = np.asarray(mesh.vertices, dtype=np.float32).copy()
faces_np = np.asarray(mesh.faces, dtype=np.int32).copy()

# Ensure Y is up
ext = verts_np.max(axis=0) - verts_np.min(axis=0)
if ext[2] > ext[1] * 1.15:
    verts_np = np.stack([verts_np[:, 0], verts_np[:, 2], -verts_np[:, 1]], axis=-1)

# Fit scale & Y-center to normalized silhouette height, then select best 90-deg yaw alignment
vmin, vmax = verts_np.min(axis=0), verts_np.max(axis=0)
mesh_h = max(float(vmax[1] - vmin[1]), 1e-4)
scale_init = tgt_h / mesh_h
verts_np = (verts_np - 0.5 * (vmin + vmax)[None, :]) * scale_init
verts_np[:, 0] += tgt_x_mid
verts_np[:, 1] += tgt_y_mid
verts_np[:, 2] += tgt_z_mid

glctx = dr.RasterizeCudaContext()
faces_i32 = torch.tensor(faces_np, dtype=torch.int32, device=device)
faces_long = faces_i32.long()

best_yaw, best_score = 0.0, -1.0
best_verts = verts_np
for cand_deg in (0.0, 90.0, 180.0, 270.0):
    R_cand = _y_rot_mat(cand_deg, device)
    v_cand = torch.tensor(verts_np, dtype=torch.float32, device=device) @ R_cand.T
    vn_cand = _compute_vertex_normals(v_cand, faces_long)
    with torch.no_grad():
        sils, _, _ = _render_views(glctx, v_cand, vn_cand, faces_i32, faces_long, cam_rots, res=256)
        # downsample targets for fast orientation selection
        t_ds = [F.interpolate(t[None, None], size=(256, 256), mode="bilinear", align_corners=False)[0, 0] for t in tgt_alphas]
        ious = _calc_ious(sils, t_ds)
        score = sum(ious.values())
    if score > best_score:
        best_score = score
        best_yaw = cand_deg
        best_verts = v_cand.detach().cpu().numpy()

# Fine 2D bbox alignment per axis before differential optimization
v_t = torch.tensor(best_verts, dtype=torch.float32, device=device)
with torch.no_grad():
    vn_0 = _compute_vertex_normals(v_t, faces_long)
    sils_pre, _, _ = _render_views(glctx, v_t, vn_0, faces_i32, faces_long, cam_rots, res=res_opt)
    pre_ious = _calc_ious(sils_pre, tgt_alphas)

M = lg.compute_matrix(v_t, faces_long, lambda_=19.0)
u = to_differential(M, v_t).detach().clone().requires_grad_(True)
v_init = v_t.detach().clone()

enable_nvdiffrast = os.environ.get("GREENFIELD_ENABLE_NVDIFFRAST", "0") == "1" and int(args.iters) > 0
if enable_nvdiffrast:
    opt = AdamUniform([u], lr=0.012)
    max_disp = 0.025
    for step in range(int(args.iters)):
        opt.zero_grad()
        v_raw = from_differential(M, u, method="Cholesky")
        v_cur = v_init + (v_raw - v_init).clamp(-max_disp, max_disp)
        vn_cur = _compute_vertex_normals(v_cur, faces_long)
        sils, norms, covs = _render_views(glctx, v_cur, vn_cur, faces_i32, faces_long, cam_rots, res=res_opt)

        loss_sil = torch.tensor(0.0, device=device)
        loss_norm = torch.tensor(0.0, device=device)
        for sil, n_pred, cov, a_tgt, n_tgt in zip(sils, norms, covs, tgt_alphas, tgt_norms):
            loss_sil = loss_sil + F.l1_loss(sil, a_tgt)
            fg_mask = (a_tgt > 0.5) & (cov > 0.5)
            if fg_mask.any():
                cos_sim = (n_pred[fg_mask] * n_tgt[fg_mask].detach()).sum(dim=-1).clamp(-1.0, 1.0)
                loss_norm = loss_norm + (1.0 - cos_sim).mean()

        loss_disp = F.mse_loss(v_raw, v_init)
        loss = 1.0 * (loss_sil / 4.0) + 0.05 * (loss_norm / 4.0) + 0.10 * loss_disp
        loss.backward()
        opt.step()

    with torch.no_grad():
        v_raw = from_differential(M, u, method="Cholesky")
        v_final = v_init + (v_raw - v_init).clamp(-max_disp, max_disp)
        vn_final = _compute_vertex_normals(v_final, faces_long)
        sils_post, norms_post, covs_post = _render_views(glctx, v_final, vn_final, faces_i32, faces_long, cam_rots, res=res_opt)
        post_ious = _calc_ious(sils_post, tgt_alphas)
else:
    v_final = v_init
    sils_post = sils_pre
    post_ious = dict(pre_ious)

# Save silhouette comparison grid (Target vs Pre vs Post)
tiles = []
for idx, r in enumerate(roles):
    t_img = (tgt_alphas[idx].cpu().numpy() * 255).astype(np.uint8)
    p_pre = (sils_pre[idx].cpu().numpy() * 255).astype(np.uint8)
    p_post = (sils_post[idx].cpu().numpy() * 255).astype(np.uint8)
    # RGB overlay: R=Target, G=Refined, B=Refined -> cyan/white match, red=missing, green=overhang
    ov = np.stack([t_img, p_post, p_post], axis=-1)
    tiles.append(Image.fromarray(ov, mode="RGB"))
grid = Image.new("RGB", (res_opt * 4, res_opt))
for i, tl in enumerate(tiles):
    grid.paste(tl, (i * res_opt, 0))
Path(args.out_grid).parent.mkdir(parents=True, exist_ok=True)
grid.save(args.out_grid)

# Export refined mesh directly via trimesh (no o_voxel attr resampling!)
refined_mesh = trimesh.Trimesh(vertices=v_final.detach().cpu().numpy(), faces=faces_np, process=False)
out_glb = Path(args.out_glb)
out_glb.parent.mkdir(parents=True, exist_ok=True)
refined_mesh.export(str(out_glb))

metrics = {
    "nvdiffrast_bypassed": not enable_nvdiffrast,
    "selected_yaw_deg": best_yaw,
    "iters": int(args.iters) if enable_nvdiffrast else 0,
    "verts": int(len(refined_mesh.vertices)),
    "faces": int(len(refined_mesh.faces)),
    "pre_iou": pre_ious,
    "post_iou": post_ious,
    "pre_bbox_iou": _calc_bbox_aligned_ious(sils_pre, tgt_alphas),
    "bbox_aligned_silhouette_iou": _calc_bbox_aligned_ious(sils_post, tgt_alphas),
}
Path(args.out_metrics).write_text(json.dumps(metrics, indent=2), encoding="utf-8")
print("STAGE3_NVDIFFRAST_OK:" + json.dumps(metrics), flush=True)
'''


_STAGE4_P3SAM_SCRIPT = r'''
import argparse, inspect, json, os, sys
from collections import defaultdict, deque
from pathlib import Path
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import numpy as np
from PIL import Image, ImageDraw
from scipy.spatial import cKDTree
import torch
import trimesh
from huggingface_hub import hf_hub_download, list_repo_files

ap = argparse.ArgumentParser()
ap.add_argument("--in-glb", required=True)
ap.add_argument("--repo-dir", required=True)
ap.add_argument("--out-dir", required=True)
ap.add_argument("--complete-occluded-shoes", choices=("auto", "always", "never"), default="auto")
args = ap.parse_args()

out_dir = Path(args.out_dir)
out_dir.mkdir(parents=True, exist_ok=True)
repo_dir = Path(args.repo_dir)

# Locate P3-SAM checkpoint cleanly from tencent/Hunyuan3D-Part
files = list_repo_files("tencent/Hunyuan3D-Part")
ckpt_rel = next((f for f in files if "p3sam" in f.lower() and f.endswith((".pt", ".ckpt", ".safetensors"))), None)
if ckpt_rel is None:
    raise FileNotFoundError(f"Could not find P3-SAM checkpoint in tencent/Hunyuan3D-Part: {files}")
ckpt_path = hf_hub_download(repo_id="tencent/Hunyuan3D-Part", filename=ckpt_rel)
print(f"[P3-SAM] Using checkpoint: {ckpt_path}", flush=True)

p3sam_root = repo_dir / "P3-SAM"
for sub in (p3sam_root / "demo", p3sam_root, repo_dir / "XPart" / "partgen", repo_dir):
    if sub.is_dir() and str(sub) not in sys.path:
        sys.path.insert(0, str(sub))

mesh = trimesh.load(args.in_glb, force="mesh", process=False)
if len(mesh.faces) == 0:
    raise RuntimeError("Empty input mesh in Stage 4")

# Disable flash attention in Sonata when flash_attn package is not present
import models.sonata as _sonata_mod
if getattr(_sonata_mod.model, "flash_attn", None) is None:
    _orig_sonata_load = _sonata_mod.load
    def _safe_sonata_load(*a, **kw):
        cfg = dict(kw.get("custom_config") or {})
        cfg["enable_flash"] = False
        kw["custom_config"] = cfg
        return _orig_sonata_load(*a, **kw)
    _sonata_mod.load = _safe_sonata_load
    _sonata_mod.model.load = _safe_sonata_load

from auto_mask import AutoMask
p3_mid_dir = out_dir / "p3sam_mid"
p3_mid_dir.mkdir(parents=True, exist_ok=True)
auto_mask = AutoMask(ckpt_path, point_num=100000, prompt_num=400, threshold=0.95, post_process=True)
_aabb, face_ids, mesh_out = auto_mask.predict_aabb(
    mesh,
    save_path=str(p3_mid_dir),
    point_num=100000,
    prompt_num=400,
    threshold=0.95,
    post_process=True,
    save_mid_res=False,
    show_info=True,
    clean_mesh_flag=False,
    seed=42,
    is_parallel=False,
    prompt_bs=6,
)
face_part_ids = np.asarray(face_ids).reshape(-1)
if len(face_part_ids) != len(mesh.faces):
    _, nn = cKDTree(mesh_out.triangles_center).query(mesh.triangles_center)
    face_part_ids = face_part_ids[nn]

face_part_ids = face_part_ids.astype(np.int32)
np.save(out_dir / "p3sam_part_ids.npy", face_part_ids)

# Classify each 3D P3-SAM cluster into semantic roles purely from 3D spatial bounds & connectivity
# Y-up space: X=left/right, Y=up, Z=front(+)/back(-)
verts = np.asarray(mesh.vertices, dtype=np.float64)
centers = np.asarray(mesh.triangles_center, dtype=np.float64)
normals = np.asarray(mesh.face_normals, dtype=np.float64)
areas = np.asarray(mesh.area_faces, dtype=np.float64)
ymin, ymax = float(verts[:, 1].min()), float(verts[:, 1].max())
H = max(ymax - ymin, 1e-4)
y_rel = (centers[:, 1] - ymin) / H

# Build connected components per P3-SAM label so disconnected L/R parts classify cleanly
ROLE_TO_ID = {
    "body": 0,
    "torso_sleeves": 1,
    "skirt": 2,
    "shoes": 3,
    "face": 4,
    "hair": 5,
    "head_prop": 6,
}

face_roles = np.zeros(len(mesh.faces), dtype=np.int32)
cluster_report = []

# Crown dome top estimate (below protruding props/wings/antlers)
head_center_mask = (y_rel >= 0.78) & (np.abs(centers[:, 0]) <= 0.11 * H)
head_x_span = np.ptp(centers[head_center_mask, 0]) if np.any(head_center_mask) else 0.14 * H

for cid in sorted(int(v) for v in np.unique(face_part_ids)):
    mask = face_part_ids == cid
    c_pts = centers[mask]
    c_nrm = normals[mask]
    c_y = y_rel[mask]
    y_lo, y_hi = float(c_y.min()), float(c_y.max())
    y_mean = float(c_y.mean())
    x_abs_max = float(np.abs(c_pts[:, 0]).max()) / H
    x_abs_mean = float(np.abs(c_pts[:, 0]).mean()) / H
    x_signed_mean = float(c_pts[:, 0].mean()) / H
    x_min_rel = float(c_pts[:, 0].min()) / H
    x_max_rel = float(c_pts[:, 0].max()) / H
    z_mean = float(c_pts[:, 2].mean()) / H
    r_xy = float(np.percentile(np.hypot(c_pts[:, 0], c_pts[:, 2]), 90)) / H
    front_dot = float(c_nrm[:, 2].mean())

    if y_lo >= 0.84 and (x_abs_max > 0.12 or y_hi >= 0.94 or y_lo >= 0.90):
        role = "head_prop"
    elif y_lo >= 0.72 and y_hi <= 0.94 and front_dot > 0.18 and x_abs_mean <= 0.085 and z_mean > 0.0:
        role = "face"
    elif y_hi >= 0.88 and (front_dot < 0.05 or z_mean <= 0.01 or y_mean >= 0.80):
        role = "hair"
    elif y_lo <= 0.04 and y_hi <= 0.40:
        role = "shoes"
    elif y_lo >= 0.30 and y_hi <= 0.68 and r_xy >= 0.085 and abs(x_signed_mean) <= 0.04 and x_min_rel < -0.05 and x_max_rel > 0.05:
        role = "skirt"
    elif y_hi >= 0.42 and y_lo >= 0.35:
        role = "torso_sleeves"
    else:
        role = "body"

    face_roles[mask] = ROLE_TO_ID[role]
    cluster_report.append({
        "cluster_id": cid,
        "role": role,
        "faces": int(mask.sum()),
        "area_pct": round(100.0 * float(areas[mask].sum()) / max(float(areas.sum()), 1e-6), 2),
        "y_rel_range": [round(y_lo, 3), round(y_hi, 3)],
        "x_abs_max_rel": round(x_abs_max, 3),
        "front_dot": round(front_dot, 3),
    })

np.save(out_dir / "p3sam_semantic_face_ids.npy", face_roles)

# Check occluded upper footwear condition for X-Part / HoloPart trigger
shoes_mask = face_roles == ROLE_TO_ID["shoes"]
skirt_mask = face_roles == ROLE_TO_ID["skirt"]
need_shoe_completion = False
if args.complete_occluded_shoes == "always":
    need_shoe_completion = True
elif args.complete_occluded_shoes == "auto" and np.any(shoes_mask) and np.any(skirt_mask):
    shoe_top = float(y_rel[shoes_mask].max())
    skirt_bot = float(y_rel[skirt_mask].min())
    need_shoe_completion = bool(skirt_bot < shoe_top - 0.02)

report = {
    "num_clusters": len(cluster_report),
    "occluded_shoes_detected": need_shoe_completion,
    "clusters": cluster_report,
}
(out_dir / "p3sam_clusters_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

# Render 4-view color-coded 3D cluster preview via software z-buffer rasterization
PALETTE = np.array([
    [180, 185, 195],  # 0 body
    [70, 165, 235],   # 1 torso_sleeves
    [245, 195, 70],   # 2 skirt
    [210, 110, 60],   # 3 shoes
    [250, 215, 190],  # 4 face
    [175, 110, 235],  # 5 hair
    [235, 70, 120],   # 6 head_prop
], dtype=np.uint8)
W_vis, H_vis = 384, 512
tiles = []
for az_deg in (0.0, 90.0, 180.0, 270.0):
    th = np.radians(az_deg)
    R = np.array([[np.cos(th), 0, -np.sin(th)], [0, 1, 0], [np.sin(th), 0, np.cos(th)]])
    c_rot = (centers - verts.mean(axis=0)) @ R.T
    px = np.clip(((c_rot[:, 0] / (H * 0.78) + 0.5) * W_vis).astype(np.int32), 0, W_vis - 1)
    py = np.clip(((0.5 - c_rot[:, 1] / (H * 1.12)) * H_vis).astype(np.int32), 0, H_vis - 1)
    order = np.argsort(c_rot[:, 2])  # back to front
    canvas = np.full((H_vis, W_vis, 3), 232, dtype=np.uint8)
    cols = PALETTE[face_roles[order]]
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            canvas[np.clip(py[order] + dy, 0, H_vis - 1), np.clip(px[order] + dx, 0, W_vis - 1)] = cols
    tiles.append(Image.fromarray(canvas, mode="RGB"))
grid = Image.new("RGB", (W_vis * 4, H_vis))
for i, t in enumerate(tiles):
    grid.paste(t, (i * W_vis, 0))
grid.save(out_dir / "stage4_p3sam_clusters.png")
print("STAGE4_P3SAM_OK:" + json.dumps({"num_clusters": len(cluster_report), "occluded_shoes": need_shoe_completion}), flush=True)
'''


_STAGE5_UNIRIG_RUNNER = r'''
import argparse, os, subprocess
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--input", required=True)
ap.add_argument("--skel-fbx", required=True)
ap.add_argument("--skin-fbx", required=True)
ap.add_argument("--out-glb", required=True)
ap.add_argument("--seed", type=int, default=12345)
args = ap.parse_args()

env = os.environ.copy()
venv_bin = str(Path(os.sys.executable).parent)
env["PATH"] = f"{venv_bin}:{env.get('PATH', '')}"

sys_src = Path("configs/system/ar_inference_articulationxl.yaml").read_text(encoding="utf-8")
Path("configs/system/ar_inference_vroid.yaml").write_text(
    sys_src.replace("assign_cls: articulationxl", "assign_cls: vroid"),
    encoding="utf-8",
)
order_p = Path("src/data/order.py")
if order_p.is_file():
    order_txt = order_p.read_text(encoding="utf-8")
    old_assert = 'assert len(names) <= num_bones, "number of bones in required skeleton is more than existing bones"'
    if old_assert in order_txt:
        order_p.write_text(order_txt.replace(old_assert, "names = names[:num_bones]"), encoding="utf-8")

task_src = Path("configs/task/quick_inference_skeleton_articulationxl_ar_256.yaml").read_text(encoding="utf-8")
Path("configs/task/quick_inference_skeleton_vroid.yaml").write_text(
    task_src.replace("system: ar_inference_articulationxl", "system: ar_inference_vroid"),
    encoding="utf-8",
)

def _has_file(p: str) -> bool:
    fp = Path(p)
    return fp.is_file() and fp.stat().st_size > 0

subprocess.run(["bash", "launch/inference/generate_skeleton.sh", "--skeleton_task", "configs/task/quick_inference_skeleton_vroid.yaml", "--force_override", "true", "--input", args.input, "--output", args.skel_fbx, "--seed", str(args.seed)], env=env, check=True)
if not _has_file(args.skel_fbx):
    print("[UniRig] vroid skeleton task produced no FBX; retrying articulationxl task", flush=True)
    subprocess.run(["bash", "launch/inference/generate_skeleton.sh", "--skeleton_task", "configs/task/quick_inference_skeleton_articulationxl_ar_256.yaml", "--force_override", "true", "--input", args.input, "--output", args.skel_fbx, "--seed", str(args.seed)], env=env, check=True)
if not _has_file(args.skel_fbx):
    raise RuntimeError(f"UniRig generate_skeleton failed to create {args.skel_fbx}")

subprocess.run(["bash", "launch/inference/generate_skin.sh", "--force_override", "true", "--input", args.skel_fbx, "--output", args.skin_fbx, "--seed", str(args.seed)], env=env, check=True)
if not _has_file(args.skin_fbx):
    raise RuntimeError(f"UniRig generate_skin failed to create {args.skin_fbx}")

subprocess.run(["bash", "launch/inference/merge.sh", "--source", args.skin_fbx, "--target", args.input, "--output", args.out_glb], env=env, check=True)
if not _has_file(args.out_glb):
    raise RuntimeError(f"UniRig merge failed to create {args.out_glb}")
print("UNIRIG_RAW_OK:" + args.out_glb, flush=True)
'''


_STAGE5_RETARGET_BLENDER_SCRIPT = r'''
import json, math, sys
from collections import defaultdict, deque
from pathlib import Path
import bmesh
import bpy
import numpy as np
from mathutils import Matrix, Vector
from mathutils.bvhtree import BVHTree
from mathutils.kdtree import KDTree

cfg = json.loads(Path(sys.argv[sys.argv.index("--") + 1]).read_text(encoding="utf-8"))
bpy.ops.wm.read_factory_settings(use_empty=True)

# 1. Import canonical 52-bone J_Bip_* armature (skeleton hierarchy + bone names only; delete canonical meshes & custom shapes!)
bpy.ops.import_scene.gltf(filepath=cfg["canonical_glb"])
can_arm = next(o for o in bpy.data.objects if o.type == "ARMATURE")
can_arm.name = "Armature"
for pb in can_arm.pose.bones:
    pb.custom_shape = None
for o in list(bpy.data.objects):
    if (o.type == "MESH" and ("icosphere" in o.name.lower() or len(o.data.polygons) == 0)) or o.type == "EMPTY":
        bpy.data.objects.remove(o, do_unlink=True)
can_meshes = [o for o in bpy.data.objects if o.type == "MESH"]
can_pts = [o.matrix_world @ v.co for o in can_meshes for v in o.data.vertices]
can_min_z = min(p.z for p in can_pts)
can_max_z = max(p.z for p in can_pts)
can_h = max(can_max_z - can_min_z, 0.5)
for o in can_meshes:
    bpy.data.objects.remove(o, do_unlink=True)

canonical_bones_52 = [b.name for b in can_arm.data.bones]

# 2. Import UniRig rigged mesh + Stage 3 ref mesh + P3-SAM labels
pre = set(bpy.data.objects)
bpy.ops.import_scene.gltf(filepath=cfg["unirig_raw_glb"])
uni_arm = next((o for o in bpy.data.objects if o not in pre and o.type == "ARMATURE"), None)
uni_meshes = [o for o in bpy.data.objects if o not in pre and o.type == "MESH"]
if uni_arm is None or not uni_meshes:
    raise RuntimeError("UniRig raw GLB missing Armature or Mesh")

# Join if multiple mesh primitives exist
uni_obj = uni_meshes[0]
for extra in uni_meshes[1:]:
    bpy.ops.object.select_all(action="DESELECT")
    uni_obj.select_set(True)
    extra.select_set(True)
    bpy.context.view_layer.objects.active = uni_obj
    bpy.ops.object.join()

bpy.ops.object.select_all(action="DESELECT")
uni_obj.select_set(True)
bpy.context.view_layer.objects.active = uni_obj
uni_obj.parent = None
bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)

# Weld glTF-split vertices immediately while preserving skinned vertex group weights
bm_w = bmesh.new()
bm_w.from_mesh(uni_obj.data)
bm_w.verts.layers.deform.verify()
bmesh.ops.remove_doubles(bm_w, verts=bm_w.verts, dist=1e-5)
bm_w.normal_update()
bm_w.to_mesh(uni_obj.data)
bm_w.free()
uni_obj.data.update()

# Scale/ground unbroken mesh cleanly to 1.75 m matching canonical armature Z-up height
pts_u = np.array([[v.co.x, v.co.y, v.co.z] for v in uni_obj.data.vertices], dtype=np.float64)
zmin, zmax = float(pts_u[:, 2].min()), float(pts_u[:, 2].max())
scale = can_h / max(zmax - zmin, 1e-4)
foot_mask = pts_u[:, 2] < zmin + 0.06 * (zmax - zmin)
dx_c = -0.5 * float(pts_u[foot_mask, 0].min() + pts_u[foot_mask, 0].max()) * scale if np.any(foot_mask) else 0.0
dy_c = -0.5 * float(pts_u[:, 1].min() + pts_u[:, 1].max()) * scale

# Preserve Stage 3 locked yaw orientation (-Y forward in Blender Z-up)
s_xy = 1.0

for v in uni_obj.data.vertices:
    v.co = Vector((s_xy * (v.co.x * scale + dx_c), s_xy * (v.co.y * scale + dy_c), (v.co.z - zmin) * scale + can_min_z))
uni_obj.data.update()

# Optional clean geometry source override (e.g. uncarved Stage 2 TRELLIS.2 GLB instead of Stage 3 box-carved GLB)
geom_src = str(cfg.get("geom_source_glb") or "").strip()
if geom_src and Path(geom_src).is_file():
    uni_donor_obj = uni_obj
    uni_donor_obj.name = "_UniRig_Weight_Donor"
    g_pre = set(bpy.data.objects)
    bpy.ops.import_scene.gltf(filepath=geom_src)
    g_new = [o for o in bpy.data.objects if o not in g_pre]
    g_meshes = [o for o in g_new if o.type == "MESH"]
    for o in g_new:
        if o.type != "MESH":
            bpy.data.objects.remove(o, do_unlink=True)
    if not g_meshes:
        raise RuntimeError(f"geom_source_glb has no MESH objects: {geom_src}")
    char_obj = g_meshes[0]
    for extra in g_meshes[1:]:
        bpy.ops.object.select_all(action="DESELECT")
        char_obj.select_set(True)
        extra.select_set(True)
        bpy.context.view_layer.objects.active = char_obj
        bpy.ops.object.join()
    bpy.ops.object.select_all(action="DESELECT")
    char_obj.select_set(True)
    bpy.context.view_layer.objects.active = char_obj
    char_obj.parent = None
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
    char_obj.name = "Body_Limbs"
    char_obj.data.name = "Body_Limbs"

    bm_g = bmesh.new()
    bm_g.from_mesh(char_obj.data)
    bmesh.ops.remove_doubles(bm_g, verts=bm_g.verts, dist=1e-5)
    bm_g.faces.ensure_lookup_table()
    visited_g = [False] * len(bm_g.faces)
    comps_g = []
    for ff in bm_g.faces:
        if visited_g[ff.index]:
            continue
        q = deque([ff])
        visited_g[ff.index] = True
        comp = []
        while q:
            cur = q.popleft()
            comp.append(cur)
            for ed in cur.edges:
                for nf in ed.link_faces:
                    if not visited_g[nf.index]:
                        visited_g[nf.index] = True
                        q.append(nf)
        comps_g.append(comp)
    if len(comps_g) > 1:
        max_fg = max(len(c) for c in comps_g)
        keep_thresh = max(500, int(0.005 * max_fg))
        drop_faces = [f for c in comps_g if len(c) < keep_thresh for f in c]
        if drop_faces and len(drop_faces) < len(bm_g.faces):
            bmesh.ops.delete(bm_g, geom=drop_faces, context="FACES")
            le = [e for e in bm_g.edges if not e.link_faces]
            if le:
                bmesh.ops.delete(bm_g, geom=le, context="EDGES")
            lv = [v for v in bm_g.verts if not v.link_faces]
            if lv:
                bmesh.ops.delete(bm_g, geom=lv, context="VERTS")
    bm_g.normal_update()
    bm_g.to_mesh(char_obj.data)
    bm_g.free()
    char_obj.data.update()

    pts_g = np.array([[v.co.x, v.co.y, v.co.z] for v in char_obj.data.vertices], dtype=np.float64)
    g_zmin, g_zmax = float(pts_g[:, 2].min()), float(pts_g[:, 2].max())
    g_scale = can_h / max(g_zmax - g_zmin, 1e-4)
    g_foot = pts_g[:, 2] < g_zmin + 0.06 * (g_zmax - g_zmin)
    g_dx = -0.5 * float(pts_g[g_foot, 0].min() + pts_g[g_foot, 0].max()) * g_scale if np.any(g_foot) else 0.0
    g_dy = -0.5 * float(pts_g[:, 1].min() + pts_g[:, 1].max()) * g_scale
    for v in char_obj.data.vertices:
        v.co = Vector((s_xy * (v.co.x * g_scale + g_dx), s_xy * (v.co.y * g_scale + g_dy), (v.co.z - g_zmin) * g_scale + can_min_z))
    char_obj.data.update()
else:
    uni_donor_obj = None
    char_obj = uni_obj
    char_obj.name = "Body_Limbs"
    char_obj.data.name = "Body_Limbs"

# Scale/translate UniRig armature identically, centering bone Y at J_Bip_C_Hips so UniRig raw Y offsets align with char_obj
uni_arm.parent = None
bpy.ops.object.select_all(action="DESELECT")
uni_arm.select_set(True)
bpy.context.view_layer.objects.active = uni_arm
bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
hips_b = uni_arm.data.bones.get("J_Bip_C_Hips")
arm_y0 = float((uni_arm.matrix_world @ hips_b.head_local).y) if hips_b is not None else float(np.median([(uni_arm.matrix_world @ b.head_local).y for b in uni_arm.data.bones]))
dy_arm = -arm_y0 * scale
uni_heads_w = {}
for b in uni_arm.data.bones:
    p = uni_arm.matrix_world @ b.head_local
    uni_heads_w[b.name] = Vector((s_xy * (p.x * scale + dx_c), s_xy * (p.y * scale + dy_arm), (p.z - zmin) * scale + can_min_z))

# Map UniRig bone names -> canonical 52 J_Bip_* bone names (excluding truncated finger names on <52-bone rigs)
def _norm(name: str) -> str:
    return name.split(":")[-1].lower().replace("_", "").replace(".", "").replace("-", "")

FINGER_KEYS = ("thumb", "index", "middle", "ring", "little")
core_cands = [bn for bn in canonical_bones_52 if not any(fk in bn.lower() for fk in FINGER_KEYS)]
can_by_norm_core = {_norm(bn): bn for bn in core_cands}
can_heads_w = {b.name: (can_arm.matrix_world @ b.head_local) for b in can_arm.data.bones}
has_full_hands = len(uni_arm.data.bones) >= 52

def _side_cands(u_pos: Vector):
    z_neck_c = can_heads_w["J_Bip_C_Neck"].z if "J_Bip_C_Neck" in can_heads_w else 1.44
    if float(u_pos.z) >= z_neck_c - 0.04:
        sub = [bn for bn in core_cands if bn in ("J_Bip_C_Head", "J_Bip_C_Neck")]
        if sub:
            return sub
    if float(u_pos.x) > 0.04:
        sub = [bn for bn in core_cands if bn.startswith("J_Bip_L_") or bn.startswith("J_Bip_C_")]
    elif float(u_pos.x) < -0.04:
        sub = [bn for bn in core_cands if bn.startswith("J_Bip_R_") or bn.startswith("J_Bip_C_")]
    else:
        sub = [bn for bn in core_cands if bn.startswith("J_Bip_C_")]
    return sub or core_cands

uni_bones_by_name = {b.name: b for b in uni_arm.data.bones}
uni_to_can = {}
matched_uni_pos = {}
for u_bn, u_pos in uni_heads_w.items():
    is_finger_name = any(fk in u_bn.lower() for fk in FINGER_KEYS)
    if not (is_finger_name and not has_full_hands):
        if u_bn in core_cands and (can_heads_w[u_bn] - u_pos).length <= 0.45:
            uni_to_can[u_bn] = u_bn
            matched_uni_pos[u_bn] = u_pos
            continue
        nu = _norm(u_bn)
        if nu in can_by_norm_core and (can_heads_w[can_by_norm_core[nu]] - u_pos).length <= 0.45:
            cbn = can_by_norm_core[nu]
            uni_to_can[u_bn] = cbn
            matched_uni_pos[cbn] = u_pos
            continue
    # Walk up UniRig parent chain only while parent link is spatially coherent (<= 0.25 m)
    mapped_parent = None
    cur_b = uni_bones_by_name.get(u_bn)
    cur_p = cur_b.parent if cur_b is not None else None
    while cur_p is not None:
        cp_pos = uni_heads_w.get(cur_p.name)
        if cp_pos is None or (u_pos - cp_pos).length > 0.25:
            break
        if cur_p.name in core_cands and (can_heads_w[cur_p.name] - cp_pos).length <= 0.45:
            mapped_parent = cur_p.name
            break
        np_name = _norm(cur_p.name)
        if np_name in can_by_norm_core and (can_heads_w[can_by_norm_core[np_name]] - cp_pos).length <= 0.45:
            mapped_parent = can_by_norm_core[np_name]
            break
        cur_p = cur_p.parent
    if mapped_parent is not None:
        uni_to_can[u_bn] = mapped_parent
    else:
        z_neck_c = can_heads_w["J_Bip_C_Neck"].z if "J_Bip_C_Neck" in can_heads_w else 1.44
        if float(u_pos.z) >= z_neck_c:
            uni_to_can[u_bn] = "J_Bip_C_Head"
        else:
            uni_to_can[u_bn] = min(_side_cands(u_pos), key=lambda cbn: (can_heads_w[cbn] - u_pos).length)

# Align canonical 52-bone joint heads, limb tails, & rotated hand finger clusters to UniRig predicted rest pose
bpy.ops.object.select_all(action="DESELECT")
can_arm.select_set(True)
bpy.context.view_layer.objects.active = can_arm
bpy.ops.object.mode_set(mode="EDIT")
inv_can = can_arm.matrix_world.inverted()
eb_map = can_arm.data.edit_bones
old_hand_info = {}
for side in ("L", "R"):
    eb_h = eb_map.get(f"J_Bip_{side}_Hand")
    eb_l = eb_map.get(f"J_Bip_{side}_LowerArm")
    if eb_h is not None and eb_l is not None:
        old_hand_info[side] = (eb_h.head.copy(), eb_h.tail.copy(), (eb_h.head - eb_l.head).normalized())

for eb in eb_map:
    if eb.name in matched_uni_pos and not any(fk in eb.name.lower() for fk in FINGER_KEYS):
        target_h = matched_uni_pos[eb.name]
        old_h = can_arm.matrix_world @ eb.head
        if (target_h - old_h).length <= 0.45:
            delta = (inv_can @ target_h) - eb.head
            eb.head += delta
            eb.tail += delta

for side, (old_h, old_t, old_dir) in old_hand_info.items():
    eb_h = eb_map.get(f"J_Bip_{side}_Hand")
    eb_l = eb_map.get(f"J_Bip_{side}_LowerArm")
    if eb_h is None or eb_l is None or (eb_h.head - eb_l.head).length < 1e-4:
        continue
    new_dir = (eb_h.head - eb_l.head).normalized()
    R_hand = old_dir.rotation_difference(new_dir)
    eb_h.tail = eb_h.head + (R_hand @ (old_t - old_h))
    prefix = f"J_Bip_{side}_"
    for eb in eb_map:
        if eb.name.startswith(prefix) and any(fk in eb.name.lower() for fk in FINGER_KEYS):
            eb.head = eb_h.head + (R_hand @ (eb.head - old_h))
            eb.tail = eb_h.head + (R_hand @ (eb.tail - old_h))

for p_bn, c_bn in (
    ("J_Bip_C_Hips", "J_Bip_C_Spine"),
    ("J_Bip_C_Spine", "J_Bip_C_Chest"),
    ("J_Bip_C_Chest", "J_Bip_C_UpperChest"),
    ("J_Bip_C_UpperChest", "J_Bip_C_Neck"),
    ("J_Bip_C_Neck", "J_Bip_C_Head"),
    ("J_Bip_L_Shoulder", "J_Bip_L_UpperArm"),
    ("J_Bip_L_UpperArm", "J_Bip_L_LowerArm"),
    ("J_Bip_L_LowerArm", "J_Bip_L_Hand"),
    ("J_Bip_R_Shoulder", "J_Bip_R_UpperArm"),
    ("J_Bip_R_UpperArm", "J_Bip_R_LowerArm"),
    ("J_Bip_R_LowerArm", "J_Bip_R_Hand"),
    ("J_Bip_L_UpperLeg", "J_Bip_L_LowerLeg"),
    ("J_Bip_L_LowerLeg", "J_Bip_L_Foot"),
    ("J_Bip_L_Foot", "J_Bip_L_ToeBase"),
    ("J_Bip_R_UpperLeg", "J_Bip_R_LowerLeg"),
    ("J_Bip_R_LowerLeg", "J_Bip_R_Foot"),
    ("J_Bip_R_Foot", "J_Bip_R_ToeBase"),
):
    eb_p = eb_map.get(p_bn)
    eb_c = eb_map.get(c_bn)
    if eb_p is not None and eb_c is not None and (eb_c.head - eb_p.head).length > 0.02:
        eb_p.tail = eb_c.head.copy()

bpy.ops.object.mode_set(mode="OBJECT")
can_heads_w = {b.name: (can_arm.matrix_world @ b.head_local) for b in can_arm.data.bones}

# Map Stage 4 P3-SAM semantic face labels onto char_obj faces via BVH nearest centroid
ref_pre = set(bpy.data.objects)
bpy.ops.import_scene.gltf(filepath=cfg["stage3_glb"])
ref_meshes = [o for o in bpy.data.objects if o not in ref_pre and o.type == "MESH"]
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
ref_pts = np.array([[v.co.x, v.co.y, v.co.z] for v in ref_obj.data.vertices], dtype=np.float64)
r_zmin, r_zmax = float(ref_pts[:, 2].min()), float(ref_pts[:, 2].max())
r_scale = can_h / max(r_zmax - r_zmin, 1e-4)
r_foot = ref_pts[:, 2] < r_zmin + 0.06 * (r_zmax - r_zmin)
r_dx = -0.5 * float(ref_pts[r_foot, 0].min() + ref_pts[r_foot, 0].max()) * r_scale if np.any(r_foot) else 0.0
r_dy = -0.5 * float(ref_pts[:, 1].min() + ref_pts[:, 1].max()) * r_scale
r_sxy = 1.0
for v in ref_obj.data.vertices:
    v.co = Vector((r_sxy * (v.co.x * r_scale + r_dx), r_sxy * (v.co.y * r_scale + r_dy), (v.co.z - r_zmin) * r_scale + can_min_z))
ref_obj.data.update()

sem_face_ids = np.load(cfg["p3sam_semantic_face_ids"]).astype(np.int32)
bm_ref = bmesh.new()
bm_ref.from_mesh(ref_obj.data)
bm_ref.faces.ensure_lookup_table()
ref_bvh = BVHTree.FromBMesh(bm_ref)
bm_ref.free()

# Transfer UniRig weights onto canonical 52 J_Bip_* vertex groups
src_obj_w = uni_donor_obj if uni_donor_obj is not None else char_obj
vg_names_u = {vg.index: vg.name for vg in src_obj_w.vertex_groups}
uni_wlist = []
for v in src_obj_w.data.vertices:
    acc = {}
    for g in v.groups:
        u_bn = vg_names_u.get(g.group)
        if not u_bn or g.weight <= 1e-5:
            continue
        c_bn = uni_to_can.get(u_bn, "J_Bip_C_Hips")
        acc[c_bn] = acc.get(c_bn, 0.0) + float(g.weight)
    tot = sum(acc.values())
    uni_wlist.append({bn: w / tot for bn, w in acc.items()} if tot > 1e-5 else {"J_Bip_C_Hips": 1.0})

if uni_donor_obj is not None:
    kd = KDTree(len(uni_donor_obj.data.vertices))
    for vi, dv in enumerate(uni_donor_obj.data.vertices):
        kd.insert(dv.co, vi)
    kd.balance()
    wlist = []
    for v in char_obj.data.vertices:
        acc = {}
        wsum = 0.0
        for _co, d_vi, dist in kd.find_n(v.co, 4):
            wt = 1.0 / (max(float(dist), 1e-4) ** 2)
            wsum += wt
            for bn, bval in uni_wlist[d_vi].items():
                acc[bn] = acc.get(bn, 0.0) + wt * bval
        tot = sum(acc.values())
        wlist.append({bn: w / tot for bn, w in acc.items()} if tot > 1e-5 else {"J_Bip_C_Hips": 1.0})
    bpy.data.objects.remove(uni_donor_obj, do_unlink=True)
else:
    wlist = uni_wlist

# Apply P3-SAM semantic roles & Stage 5 weight refinements
ROLE_BODY, ROLE_TORSO, ROLE_SKIRT, ROLE_SHOES, ROLE_FACE, ROLE_HAIR, ROLE_HEAD_PROP = 0, 1, 2, 3, 4, 5, 6
bm = bmesh.new()
bm.from_mesh(char_obj.data)
bm.verts.ensure_lookup_table()
bm.faces.ensure_lookup_table()

role_lay = bm.faces.layers.int.new("p3sam_role")
for f in bm.faces:
    _loc, _hn, r_fi, _d = ref_bvh.find_nearest(f.calc_center_median())
    f[role_lay] = int(sem_face_ids[r_fi]) if (r_fi is not None and 0 <= r_fi < len(sem_face_ids)) else ROLE_BODY

bm.normal_update()
z_hip = can_heads_w["J_Bip_C_Hips"].z
z_spine = can_heads_w["J_Bip_C_Spine"].z
z_ul = 0.5 * (can_heads_w["J_Bip_L_UpperLeg"].z + can_heads_w["J_Bip_R_UpperLeg"].z)
z_knee = 0.5 * (can_heads_w["J_Bip_L_LowerLeg"].z + can_heads_w["J_Bip_R_LowerLeg"].z)
z_ankle = 0.5 * (can_heads_w["J_Bip_L_Foot"].z + can_heads_w["J_Bip_R_Foot"].z)
z_toe = 0.5 * (can_heads_w["J_Bip_L_ToeBase"].z + can_heads_w["J_Bip_R_ToeBase"].z)
z_neck = can_heads_w["J_Bip_C_Neck"].z
z_head = can_heads_w["J_Bip_C_Head"].z
z_uchest = can_heads_w["J_Bip_C_UpperChest"].z
z_chest = can_heads_w["J_Bip_C_Chest"].z

# Ensure front face oval is labeled ROLE_FACE, and front-lateral lower-jaw skin stays ROLE_BODY (never ROLE_HAIR)
for f in bm.faces:
    c = f.calc_center_median()
    if f[role_lay] == ROLE_FACE and (abs(c.x) > 0.062 or c.y > -0.045 or f.normal.y > -0.38 or c.z < z_neck + 0.018):
        if (z_neck + 0.015 <= c.z <= z_head + 0.012) and abs(c.x) <= 0.064 and c.y <= -0.048 and f.normal.y <= 0.10:
            f[role_lay] = ROLE_BODY
        else:
            f[role_lay] = ROLE_HAIR if c.z >= z_neck + 0.025 else ROLE_TORSO
    elif f[role_lay] in (ROLE_HAIR, ROLE_BODY, ROLE_TORSO) and (z_neck + 0.020 <= c.z <= z_head + 0.075) and abs(c.x) <= 0.058 and c.y <= -0.050 and f.normal.y <= -0.42:
        f[role_lay] = ROLE_FACE
    elif f[role_lay] == ROLE_HAIR and (z_neck + 0.015 <= c.z <= z_head + 0.012) and abs(c.x) <= 0.064 and c.y <= -0.048 and f.normal.y <= 0.10:
        f[role_lay] = ROLE_BODY

z_crotch_w = z_ul - 0.06
z_crotch = z_ul - 0.12

def _clamp_cross_leg_weights(wdict, pos_x: float, pos_z: float, has_skirt: bool, side_override: str | None = None):
    if has_skirt or pos_z >= z_crotch_w:
        return wdict
    keep_s = side_override if side_override in ("L", "R") else ("L" if pos_x >= 0.0 else "R")
    drop_tok = "_R_" if keep_s == "L" else "_L_"
    filtered = {bn: wt for bn, wt in wdict.items() if drop_tok not in bn}
    tot = sum(filtered.values())
    if tot > 1e-5:
        return {bn: wt / tot for bn, wt in filtered.items()}
    return {f"J_Bip_{keep_s}_UpperLeg": 1.0}

for vi, v in enumerate(bm.verts):
    has_sk = any(int(f[role_lay]) == ROLE_SKIRT for f in v.link_faces)
    wlist[vi] = _clamp_cross_leg_weights(wlist[vi], v.co.x, v.co.z, has_sk)

def _seg_dist(pt: Vector, a: Vector, b: Vector) -> float:
    ab = b - a
    d2 = ab.length_squared
    if d2 <= 1e-10:
        return (pt - a).length
    t = max(0.0, min(1.0, (pt - a).dot(ab) / d2))
    return (pt - (a + ab * t)).length

def _poly_dist(pt: Vector, poly_pts) -> float:
    return min(_seg_dist(pt, poly_pts[i], poly_pts[i + 1]) for i in range(len(poly_pts) - 1))

arm_polys = {}
leg_polys = {}
for s in ("L", "R"):
    p_ua = can_heads_w[f"J_Bip_{s}_UpperArm"]
    p_la = can_heads_w[f"J_Bip_{s}_LowerArm"]
    p_hd = can_heads_w[f"J_Bip_{s}_Hand"]
    h_ext = p_hd + (p_hd - p_la).normalized() * 0.08
    arm_polys[s] = [p_ua, p_la, p_hd, h_ext]
    leg_polys[s] = [
        can_heads_w[f"J_Bip_{s}_UpperLeg"],
        can_heads_w[f"J_Bip_{s}_LowerLeg"],
        can_heads_w[f"J_Bip_{s}_Foot"],
        can_heads_w[f"J_Bip_{s}_ToeBase"],
    ]

x_ua_abs = 0.5 * (abs(can_heads_w["J_Bip_L_UpperArm"].x) + abs(can_heads_w["J_Bip_R_UpperArm"].x))
x_la_abs = 0.5 * (abs(can_heads_w["J_Bip_L_LowerArm"].x) + abs(can_heads_w["J_Bip_R_LowerArm"].x))
x_hd_abs = 0.5 * (abs(can_heads_w["J_Bip_L_Hand"].x) + abs(can_heads_w["J_Bip_R_Hand"].x))
z_ua_avg = 0.5 * (can_heads_w["J_Bip_L_UpperArm"].z + can_heads_w["J_Bip_R_UpperArm"].z)
z_la_avg = 0.5 * (can_heads_w["J_Bip_L_LowerArm"].z + can_heads_w["J_Bip_R_LowerArm"].z)
z_hd_avg = 0.5 * (can_heads_w["J_Bip_L_Hand"].z + can_heads_w["J_Bip_R_Hand"].z)

def _x_arm_sep(z_val: float) -> float:
    if z_val < z_la_avg:
        t = (z_la_avg - z_val) / max(z_la_avg - z_hd_avg, 0.05)
        x_b = x_la_abs + t * (x_hd_abs - x_la_abs)
    else:
        t = (z_val - z_la_avg) / max(z_ua_avg - z_la_avg, 0.05)
        x_b = x_la_abs - t * (x_la_abs - x_ua_abs)
    return max(0.118, min(0.182, x_b - 0.030))

# Preserve original vertex indices so split vertices inherit UniRig weights cleanly, then split below-armpit arm/tasset bridge seams
orig_vi_lay = bm.verts.layers.int.new("orig_vi")
for v in bm.verts:
    v[orig_vi_lay] = v.index

arm_lay = bm.faces.layers.int.new("is_arm_face")
for f in bm.faces:
    c = f.calc_center_median()
    r = int(f[role_lay])
    s_f = "L" if c.x >= 0.0 else "R"
    d_af = _poly_dist(c, arm_polys[s_f])
    d_lf = _poly_dist(c, leg_polys[s_f])
    is_af = (
        r not in (ROLE_HAIR, ROLE_HEAD_PROP, ROLE_FACE, ROLE_SKIRT)
        and (z_hd_avg - 0.12 <= c.z <= z_uchest + 0.03)
        and (abs(c.x) >= _x_arm_sep(c.z))
        and (d_af < min(0.120, d_lf + 0.005))
    )
    f[arm_lay] = 1 if is_af else 0

seam_edges = [
    e for e in bm.edges
    if len(e.link_faces) == 2
    and (e.link_faces[0][arm_lay] != e.link_faces[1][arm_lay])
    and (0.5 * (e.verts[0].co.z + e.verts[1].co.z) < z_uchest - 0.04)
]
if seam_edges:
    bmesh.ops.split_edges(bm, edges=seam_edges)
    bm.verts.ensure_lookup_table()
    bm.edges.ensure_lookup_table()
    bm.faces.ensure_lookup_table()

# Disconnect fused knee/leg midline geometry below z_crotch so L and R legs never bridge during wide strides
wide_cross_faces = [
    f for f in bm.faces
    if int(f[role_lay]) != ROLE_SKIRT
    and f.calc_center_median().z < z_crotch
    and min(v.co.x for v in f.verts) < -0.005
    and max(v.co.x for v in f.verts) > 0.005
]
if wide_cross_faces:
    bmesh.ops.delete(bm, geom=wide_cross_faces, context="FACES")
    le = [e for e in bm.edges if not e.link_faces]
    if le:
        bmesh.ops.delete(bm, geom=le, context="EDGES")
    lv = [v for v in bm.verts if not v.link_faces]
    if lv:
        bmesh.ops.delete(bm, geom=lv, context="VERTS")
    bm.verts.ensure_lookup_table()
    bm.edges.ensure_lookup_table()
    bm.faces.ensure_lookup_table()

leg_side_lay = bm.faces.layers.int.new("leg_side")
for f in bm.faces:
    if int(f[role_lay]) != ROLE_SKIRT and f.calc_center_median().z < z_crotch:
        f[leg_side_lay] = 1 if f.calc_center_median().x >= 0.0 else -1
    else:
        f[leg_side_lay] = 0

leg_mid_edges = [
    e for e in bm.edges
    if len(e.link_faces) == 2
    and (e.link_faces[0][leg_side_lay] * e.link_faces[1][leg_side_lay] < 0)
]
if leg_mid_edges:
    bmesh.ops.split_edges(bm, edges=leg_mid_edges)
    bm.verts.ensure_lookup_table()
    bm.edges.ensure_lookup_table()
    bm.faces.ensure_lookup_table()

for _bt_pass in range(6):
    bad_bt_faces = set()
    for v in bm.verts:
        if not v.is_valid:
            continue
        pos_f = [f for f in v.link_faces if f.is_valid and f[leg_side_lay] > 0]
        neg_f = [f for f in v.link_faces if f.is_valid and f[leg_side_lay] < 0]
        if pos_f and neg_f:
            bad_bt_faces.update(pos_f if len(pos_f) <= len(neg_f) else neg_f)
    if not bad_bt_faces:
        break
    bmesh.ops.delete(bm, geom=list(bad_bt_faces), context="FACES")
    le = [e for e in bm.edges if not e.link_faces]
    if le:
        bmesh.ops.delete(bm, geom=le, context="EDGES")
    lv = [v for v in bm.verts if not v.link_faces]
    if lv:
        bmesh.ops.delete(bm, geom=lv, context="VERTS")
    bm.verts.ensure_lookup_table()
    bm.edges.ensure_lookup_table()
    bm.faces.ensure_lookup_table()

wlist = [dict(wlist[v[orig_vi_lay]]) for v in bm.verts]
bm.normal_update()

# Nudge separated leg vertices slightly away from x=0 and below-armpit arm vertices slightly outward so later welds never re-fuse
for v in bm.verts:
    leg_sides_v = {f[leg_side_lay] for f in v.link_faces if f[leg_side_lay] != 0}
    if leg_sides_v == {1}:
        v.co.x = max(v.co.x, 0.0018)
    elif leg_sides_v == {-1}:
        v.co.x = min(v.co.x, -0.0018)
    if v.co.z < z_uchest - 0.04 and any(f[arm_lay] for f in v.link_faces) and not any(not f[arm_lay] for f in v.link_faces):
        if any(e.is_boundary for e in v.link_edges):
            v.co.x += 0.0008 if v.co.x >= 0.0 else -0.0008

# Store material slot index from P3-SAM role (keeps continuous mesh unbroken while isolating material slots)
# Slot 0: AI_Body_Limbs (body/torso_sleeves/shoes), Slot 1: AI_Skirt_Coat, Slot 2: AI_Head_Face, Slot 3: AI_Hair, Slot 4: AI_Rigid_Prop_Head
ROLE_TO_SLOT = {
    ROLE_BODY: 0,
    ROLE_TORSO: 0,
    ROLE_SHOES: 0,
    ROLE_SKIRT: 1,
    ROLE_FACE: 2,
    ROLE_HAIR: 3,
    ROLE_HEAD_PROP: 4,
}
vert_roles = [set() for _ in range(len(bm.verts))]
for f in bm.faces:
    r = int(f[role_lay])
    f.material_index = ROLE_TO_SLOT.get(r, 0)
    for v in f.verts:
        vert_roles[v.index].add(r)

skirt_verts, hair_verts, limb_repair_verts = [], [], []
for vi, v in enumerate(bm.verts):
    p = v.co
    rs = vert_roles[vi]
    leg_sides_v = {f[leg_side_lay] for f in v.link_faces if f[leg_side_lay] != 0}
    if 1 in leg_sides_v and -1 not in leg_sides_v:
        s_a = "L"
    elif -1 in leg_sides_v and 1 not in leg_sides_v:
        s_a = "R"
    else:
        s_a = "L" if p.x >= 0.0 else "R"
    z_ua = can_heads_w[f"J_Bip_{s_a}_UpperArm"].z
    z_la = can_heads_w[f"J_Bip_{s_a}_LowerArm"].z
    z_hd = can_heads_w[f"J_Bip_{s_a}_Hand"].z
    link_arm = [f[arm_lay] for f in v.link_faces]
    is_arm_vert = bool(link_arm) and (
        any(link_arm) if p.z < z_uchest - 0.04 else all(link_arm)
    )

    if ROLE_HEAD_PROP in rs and rs == {ROLE_HEAD_PROP}:
        wlist[vi] = {"J_Bip_C_Head": 1.0}
    elif (ROLE_FACE in rs and p.z >= z_neck - 0.015) or (ROLE_BODY in rs and p.z >= z_neck + 0.015 and p.y <= -0.042 and abs(p.x) <= 0.068 and ROLE_HEAD_PROP not in rs):
        if p.z >= z_neck + 0.020:
            wlist[vi] = {"J_Bip_C_Head": 1.0}
        else:
            u = max(0.0, min(1.0, (p.z - z_neck) / 0.020))
            wlist[vi] = {"J_Bip_C_Head": u, "J_Bip_C_Neck": 1.0 - u}
    elif ROLE_HAIR in rs and ROLE_TORSO not in rs and ROLE_BODY not in rs:
        hair_verts.append(vi)
        if p.z >= z_head - 0.01:
            wlist[vi] = {"J_Bip_C_Head": 1.0}
        elif p.z >= z_neck:
            u = max(0.0, min(1.0, (p.z - z_neck) / max(z_head - 0.01 - z_neck, 0.03)))
            wlist[vi] = {"J_Bip_C_Head": u, "J_Bip_C_Neck": 1.0 - u}
        elif p.z >= z_uchest - 0.05:
            u = max(0.0, min(1.0, (p.z - (z_uchest - 0.05)) / max(z_neck - (z_uchest - 0.05), 0.05)))
            wlist[vi] = {"J_Bip_C_Neck": u, "J_Bip_C_UpperChest": 1.0 - u}
        else:
            wlist[vi] = {"J_Bip_C_UpperChest": 0.5, "J_Bip_C_Chest": 0.5}
    elif is_arm_vert:
        dom_orig = max(wlist[vi].items(), key=lambda kv: kv[1])[0] if wlist[vi] else ""
        bad_arm_dom = (
            dom_orig in ("J_Bip_C_Hips", "J_Bip_C_Spine", "J_Bip_C_Chest")
            or ("Leg" in dom_orig)
            or ("Foot" in dom_orig)
            or ("Toe" in dom_orig)
        )
        allowed_arm = {
            f"J_Bip_{s_a}_Shoulder",
            f"J_Bip_{s_a}_UpperArm",
            f"J_Bip_{s_a}_LowerArm",
            f"J_Bip_{s_a}_Hand",
            "J_Bip_C_UpperChest",
        }
        kept = {bn: wt for bn, wt in wlist[vi].items() if bn in allowed_arm}
        ktot = sum(kept.values())
        if ktot >= 0.50 and not bad_arm_dom:
            wlist[vi] = {bn: wt / ktot for bn, wt in kept.items()}
        else:
            if p.z >= z_la + 0.045:
                t_su = max(0.0, min(1.0, (p.z - (z_ua - 0.02)) / 0.06))
                seg_w = {f"J_Bip_{s_a}_Shoulder": t_su * 0.25, f"J_Bip_{s_a}_UpperArm": 1.0 - t_su * 0.25}
            elif p.z >= z_la - 0.045:
                t_el = (p.z - (z_la - 0.045)) / 0.09
                seg_w = {f"J_Bip_{s_a}_UpperArm": t_el, f"J_Bip_{s_a}_LowerArm": 1.0 - t_el}
            elif p.z >= z_hd + 0.045:
                seg_w = {f"J_Bip_{s_a}_LowerArm": 1.0}
            elif p.z >= z_hd - 0.045:
                t_wr = (p.z - (z_hd - 0.045)) / 0.09
                seg_w = {f"J_Bip_{s_a}_LowerArm": t_wr, f"J_Bip_{s_a}_Hand": 1.0 - t_wr}
            else:
                seg_w = {f"J_Bip_{s_a}_Hand": 1.0}
            wlist[vi] = {bn: wt for bn, wt in seg_w.items() if wt > 1e-5}
            limb_repair_verts.append(vi)
    else:
        if p.z < z_uchest - 0.02:
            wlist[vi] = {
                bn: wt
                for bn, wt in wlist[vi].items()
                if not any(ak in bn for ak in ("UpperArm", "LowerArm", "Hand"))
                and not (p.z < z_chest and "Shoulder" in bn)
            }
        if ROLE_SKIRT in rs and p.z <= z_hip + 0.06:
            skirt_verts.append(vi)
            if p.z >= z_hip:
                u = max(0.0, min(1.0, (p.z - z_hip) / 0.06))
                wlist[vi] = {"J_Bip_C_Spine": u * 0.35, "J_Bip_C_Hips": 1.0 - u * 0.35}
            else:
                t_lr = max(0.0, min(1.0, (p.x + 0.06) / 0.12))
                wlist[vi] = {"J_Bip_C_Hips": 0.65, "J_Bip_L_UpperLeg": 0.35 * t_lr, "J_Bip_R_UpperLeg": 0.35 * (1.0 - t_lr)}
        elif p.z >= z_hip - 0.03:
            rtot = sum(wlist[vi].values())
            if rtot >= 0.35:
                wlist[vi] = {bn: wt / rtot for bn, wt in wlist[vi].items()}
            else:
                if p.z <= z_spine:
                    u = max(0.0, min(1.0, (p.z - (z_hip - 0.03)) / max(z_spine - (z_hip - 0.03), 0.03)))
                    seg_w = {"J_Bip_C_Spine": u, "J_Bip_C_Hips": 1.0 - u}
                elif p.z <= z_chest:
                    u = max(0.0, min(1.0, (p.z - z_spine) / max(z_chest - z_spine, 0.03)))
                    seg_w = {"J_Bip_C_Chest": u, "J_Bip_C_Spine": 1.0 - u}
                elif p.z <= z_uchest:
                    u = max(0.0, min(1.0, (p.z - z_chest) / max(z_uchest - z_chest, 0.03)))
                    seg_w = {"J_Bip_C_UpperChest": u, "J_Bip_C_Chest": 1.0 - u}
                else:
                    u = max(0.0, min(1.0, (p.z - z_uchest) / max(z_neck - z_uchest, 0.03)))
                    seg_w = {"J_Bip_C_Neck": u, "J_Bip_C_UpperChest": 1.0 - u}
                wlist[vi] = {bn: wt for bn, wt in seg_w.items() if wt > 1e-5}
                limb_repair_verts.append(vi)
        else:
            b_ul = f"J_Bip_{s_a}_UpperLeg"
            b_ll = f"J_Bip_{s_a}_LowerLeg"
            b_ft = f"J_Bip_{s_a}_Foot"
            b_tb = f"J_Bip_{s_a}_ToeBase"
            allowed_leg = {b_ul, b_ll, b_ft, b_tb}
            if p.z >= z_ul - 0.06:
                allowed_leg.add("J_Bip_C_Hips")
            if p.z > z_knee - 0.04:
                allowed_leg.discard(b_ft)
                allowed_leg.discard(b_tb)
            if p.z < z_knee - 0.10:
                allowed_leg.discard(b_ul)
            if p.z > z_ul - 0.05:
                allowed_leg.discard(b_ll)

            kept = {bn: wt for bn, wt in wlist[vi].items() if bn in allowed_leg}
            if "J_Bip_C_Hips" in kept:
                max_hip_w = max(0.0, min(1.0, (p.z - (z_ul - 0.06)) / 0.09))
                if kept["J_Bip_C_Hips"] > max_hip_w:
                    kept["J_Bip_C_Hips"] = max_hip_w
                    if max_hip_w <= 1e-5:
                        kept.pop("J_Bip_C_Hips", None)
            ktot = sum(kept.values())
            if ktot >= 0.35:
                wlist[vi] = {bn: wt / ktot for bn, wt in kept.items()}
            else:
                if p.z >= z_knee + 0.05:
                    seg_w = {b_ul: 1.0}
                    if p.z > z_ul - 0.06:
                        hw = max(0.0, min(0.75, (p.z - (z_ul - 0.06)) / 0.10))
                        seg_w = {"J_Bip_C_Hips": hw, b_ul: 1.0 - hw}
                elif p.z >= z_knee - 0.05:
                    t_kn = (p.z - (z_knee - 0.05)) / 0.10
                    seg_w = {b_ul: t_kn, b_ll: 1.0 - t_kn}
                elif p.z >= z_ankle + 0.04:
                    seg_w = {b_ll: 1.0}
                elif p.z >= z_ankle - 0.03:
                    t_an = (p.z - (z_ankle - 0.03)) / 0.07
                    seg_w = {b_ll: t_an, b_ft: 1.0 - t_an}
                else:
                    if p.z <= z_toe + 0.02 and p.y < -0.02:
                        t_to = max(0.0, min(1.0, (-0.02 - p.y) / 0.06))
                        seg_w = {b_ft: 1.0 - t_to, b_tb: t_to}
                    else:
                        seg_w = {b_ft: 1.0}
                wlist[vi] = {bn: wt for bn, wt in seg_w.items() if wt > 1e-5}
                limb_repair_verts.append(vi)

# Smooth repaired weights across connected limb neighborhoods & smooth skirt/hair weights
for subset, iters in ((limb_repair_verts, 4), (skirt_verts, 6), (hair_verts, 6)):
    if not subset:
        continue
    sset = set(subset)
    for _ in range(iters):
        upd = {}
        for vi in subset:
            vb = bm.verts[vi]
            acc = {bn: val * 2.0 for bn, val in wlist[vi].items()}
            for e in vb.link_edges:
                ni = e.other_vert(vb).index
                if ni in sset:
                    if (
                        vb.co.z < z_crotch_w
                        and ROLE_SKIRT not in vert_roles[vi]
                        and ROLE_SKIRT not in vert_roles[ni]
                        and (vb.co.x >= 0.0) != (bm.verts[ni].co.x >= 0.0)
                    ):
                        continue
                    for bn, val in wlist[ni].items():
                        acc[bn] = acc.get(bn, 0.0) + val
            tot = sum(acc.values())
            if tot > 1e-5:
                upd[vi] = {bn: val / tot for bn, val in acc.items()}
        for vi, wn in upd.items():
            wlist[vi] = wn

# Final single-side leg-weight hard clamp below z_crotch_w for all non-skirt vertices
for vi, v in enumerate(bm.verts):
    leg_sides_v = {f[leg_side_lay] for f in v.link_faces if f[leg_side_lay] != 0}
    s_ov = "L" if (1 in leg_sides_v and -1 not in leg_sides_v) else ("R" if (-1 in leg_sides_v and 1 not in leg_sides_v) else None)
    wlist[vi] = _clamp_cross_leg_weights(wlist[vi], v.co.x, v.co.z, ROLE_SKIRT in vert_roles[vi], side_override=s_ov)

# Assign 5 semantic materials BEFORE bm.to_mesh so polygons keep material_index slots cleanly
char_obj.data.materials.clear()
for mname in ("AI_Body_Limbs", "AI_Skirt_Coat", "AI_Head_Face", "AI_Hair", "AI_Rigid_Prop_Head"):
    m = bpy.data.materials.new(mname)
    m.use_nodes = True
    char_obj.data.materials.append(m)

bm.normal_update()
bm.to_mesh(char_obj.data)
bm.free()
char_obj.data.update()

# Write 52-bone vertex groups and parent to canonical Armature
for g in list(char_obj.vertex_groups):
    char_obj.vertex_groups.remove(g)
for bn in canonical_bones_52:
    char_obj.vertex_groups.new(name=bn)
for vi, w in enumerate(wlist):
    for bn, val in w.items():
        if bn in char_obj.vertex_groups and val > 1e-5:
            char_obj.vertex_groups[bn].add([vi], float(val), "REPLACE")

for m in list(char_obj.modifiers):
    if m.type == "ARMATURE":
        char_obj.modifiers.remove(m)
amod = char_obj.modifiers.new("Armature", "ARMATURE")
amod.object = can_arm
char_obj.parent = can_arm
char_obj.parent_type = "OBJECT"

# Remove UniRig raw armature and ref_obj before export
bpy.data.objects.remove(uni_arm, do_unlink=True)
bpy.data.objects.remove(ref_obj, do_unlink=True)

out_glb = Path(cfg["out_glb"])
out_glb.parent.mkdir(parents=True, exist_ok=True)
bpy.ops.object.select_all(action="DESELECT")
can_arm.select_set(True)
char_obj.select_set(True)
bpy.context.view_layer.objects.active = can_arm
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
print(f"STAGE5_RETARGET_OK:{out_glb}", flush=True)
'''


# ---------------------------------------------------------------------------
# Environment Installers & Stage Runners
# ---------------------------------------------------------------------------
def _install_vision_env() -> Path:
    py_vis, _ = ensure_uv_env(
        "greenfield_vision",
        None,
        python_version="3.10",
        install_cmds=[
            ["{UV}", "pip", "install", "--python", "{PY}", "torch==2.6.0", "torchvision==0.21.0", "--index-url", "https://download.pytorch.org/whl/cu124"],
            ["{UV}", "pip", "install", "--python", "{PY}", "transformers==4.48.3", "huggingface_hub==0.28.1", "diffusers==0.32.2", "accelerate==1.3.0", "timm", "einops", "kornia", "opencv-python-headless", "pillow", "numpy", "scipy", "matplotlib"],
        ],
    )
    return py_vis


def _format_cuda_wheel_tag(cuda_ver: str | None) -> str:
    if not cuda_ver:
        return "cpu"
    cleaned = str(cuda_ver).strip().lower()
    if cleaned.startswith("cu") and cleaned[2:].isdigit():
        return cleaned
    parts = [p for p in re.split(r"[^0-9]+", cleaned) if p]
    if len(parts) >= 2:
        return f"cu{parts[0]}{parts[1]}"
    return f"cu{parts[0]}0" if parts else "cpu"


def _compute_wheel_key(cuda_ver: str | None, torch_ver: str | None, sm_arch: str | None, py_tag: str = "cp310") -> str:
    cu_tag = _format_cuda_wheel_tag(cuda_ver)
    t_base = str(torch_ver or "0.0").split("+", 1)[0]
    t_parts = [p for p in re.split(r"[^0-9]+", t_base) if p]
    torch_mm = f"{t_parts[0]}{t_parts[1]}" if len(t_parts) >= 2 else (t_parts[0] if t_parts else "0")
    sm_digits = re.sub(r"[^0-9]+", "", str(sm_arch or "00")) or "00"
    return f"{cu_tag}_torch{torch_mm}_sm{sm_digits}_{py_tag.lower()}"


def _probe_python_torch_cuda(py_bin: Path) -> dict[str, str]:
    probe_code = (
        "import json, sys, torch; "
        "cap = torch.cuda.get_device_capability(0) if torch.cuda.is_available() else (0, 0); "
        "print(json.dumps({'torch_ver': torch.__version__, 'cuda_ver': torch.version.cuda or '', "
        "'sm_arch': f'{cap[0]}.{cap[1]}', 'py_tag': f'cp{sys.version_info.major}{sys.version_info.minor}'}))"
    )
    out = subprocess.check_output([str(py_bin), "-c", probe_code], text=True).strip()
    return json.loads(out.splitlines()[-1])


def _find_cached_wheel(env_wheel_dir: Path, pkg_name: str) -> Path | None:
    if not env_wheel_dir.is_dir():
        return None
    norm_target = re.sub(r"[-_.]+", "_", pkg_name).lower()
    candidates = []
    for p in sorted(env_wheel_dir.glob("*.whl")):
        if p.stat().st_size <= 0:
            continue
        dist_part = re.split(r"-(?=[0-9])", p.stem, maxsplit=1)[0]
        if re.sub(r"[-_.]+", "_", dist_part).lower() == norm_target:
            candidates.append(p)
    return candidates[-1] if candidates else None


def _patch_cpp17_to_cpp20(src_dir: Path) -> None:
    for setup_name in ("setup.py", "pyproject.toml"):
        p = src_dir / setup_name
        if p.is_file():
            txt = p.read_text(encoding="utf-8", errors="replace")
            if "c++17" in txt:
                p.write_text(txt.replace("c++17", "c++20"), encoding="utf-8")


def _can_import_module(py_bin: Path, mod_name: str) -> bool:
    return subprocess.run([str(py_bin), "-c", f"import {mod_name}"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0


def _build_or_install_trellis2_extensions(py_tr: Path, repo_tr: Path | None, wheel_cache_dir: Path | str | None = None) -> None:
    uv_bin = _resolve_uv()
    info = _probe_python_torch_cuda(py_tr)
    cu_tag = _format_cuda_wheel_tag(info.get("cuda_ver"))
    env_key = _compute_wheel_key(info.get("cuda_ver"), info.get("torch_ver"), info.get("sm_arch"), info.get("py_tag", "cp310"))
    env_wheel_dir = (Path(wheel_cache_dir) if wheel_cache_dir else (WORKSPACE_ROOT / "wheel_cache")) / env_key
    env_wheel_dir.mkdir(parents=True, exist_ok=True)

    for idx_tag in dict.fromkeys([cu_tag, "cu128", "cu126"]):
        if idx_tag == "cpu":
            continue
        try:
            _run([uv_bin, "pip", "install", "--python", str(py_tr), "-U", "mslk", "--index-url", f"https://download.pytorch.org/whl/{idx_tag}"])
            break
        except subprocess.CalledProcessError:
            continue

    nproc = max(2, min(8, os.cpu_count() or 4))
    build_env = {
        "PATH": f"{py_tr.parent}:/usr/local/cuda/bin:{os.environ.get('PATH', '')}",
        "CUDA_HOME": os.environ.get("CUDA_HOME", "/usr/local/cuda"),
        "LIBRARY_PATH": f"/usr/local/cuda/lib64/stubs:/usr/lib/x86_64-linux-gnu:{os.environ.get('LIBRARY_PATH', '')}",
        "TORCH_CUDA_ARCH_LIST": info["sm_arch"] if info.get("sm_arch", "0.0") != "0.0" else "8.0;8.9",
        "MAX_JOBS": str(nproc),
        "NINJA_MAX_JOBS": str(nproc),
    }

    ext_build_root = Path("/tmp/trellis2_ext_src")
    shutil.rmtree(ext_build_root, ignore_errors=True)
    ext_build_root.mkdir(parents=True, exist_ok=True)

    for ext in TRELLIS2_CUDA_EXTENSIONS:
        pkg, imp = ext["pkg"], ext["import_name"]
        if _can_import_module(py_tr, imp):
            continue
        cached_whl = _find_cached_wheel(env_wheel_dir, pkg)
        if cached_whl is not None:
            try:
                _run([uv_bin, "pip", "install", "--python", str(py_tr), "--no-deps", str(cached_whl)], env_extra=build_env)
                if _can_import_module(py_tr, imp):
                    continue
            except subprocess.CalledProcessError:
                pass
            cached_whl.unlink(missing_ok=True)

        if ext.get("local_subpath") and repo_tr is not None:
            src_dir = repo_tr / ext["local_subpath"]
        else:
            src_dir = ext_build_root / pkg
            c_cmd = ["git", "clone", "--depth", "1"]
            if ext.get("branch"):
                c_cmd.extend(["-b", str(ext["branch"])])
            if ext.get("recursive"):
                c_cmd.append("--recursive")
            c_cmd.extend([str(ext["repo"]), str(src_dir)])
            _run(c_cmd)

        if ext.get("cpp20"):
            _patch_cpp17_to_cpp20(src_dir)
        _run([str(py_tr), "-m", "pip", "wheel", "--no-deps", "--no-build-isolation", "--wheel-dir", str(env_wheel_dir), str(src_dir)], cwd=src_dir, env_extra=build_env)
        built_whl = _find_cached_wheel(env_wheel_dir, pkg)
        if built_whl is None:
            raise RuntimeError(f"Failed to compile wheel for {pkg}")
        _run([uv_bin, "pip", "install", "--python", str(py_tr), "--no-deps", str(built_whl)], env_extra=build_env)

    # Install cholespy & largesteps inside trellis2 venv for Stage 3 differentiable lock
    _run([uv_bin, "pip", "install", "--python", str(py_tr), "cholespy", "largesteps"])


def _install_trellis2_env(wheel_cache_dir: Path | str | None = None) -> tuple[Path, Path]:
    py_tr, repo_tr = ensure_uv_env(
        "trellis2",
        TRELLIS2_REPO,
        python_version="3.10",
        clone_recursive=True,
        install_cmds=[
            ["{UV}", "pip", "install", "--python", "{PY}", "torch", "torchvision", "torchaudio"],
            ["{UV}", "pip", "install", "--python", "{PY}", "xformers"],
            ["{UV}", "pip", "install", "--python", "{PY}", "imageio", "imageio-ffmpeg", "tqdm", "easydict", "opencv-python-headless", "ninja", "trimesh", "transformers", "pandas", "lpips", "zstandard", "kornia", "timm", "pillow", "huggingface_hub", "plyfile", "scipy", "git+https://github.com/EasternJournalist/utils3d.git@9a4eb15e4021b67b12c460c7057d642626897ec8"],
        ],
        post_install_fn=lambda p_bin, r_dir: _build_or_install_trellis2_extensions(p_bin, r_dir, wheel_cache_dir),
    )
    assert repo_tr is not None
    return py_tr, repo_tr


def _install_p3sam_env() -> tuple[Path, Path]:
    def _post_p3sam(py_p3: Path, repo_p3: Path | None) -> None:
        uv_bin = _resolve_uv()
        if repo_p3 is None:
            return
        req_p = repo_p3 / "P3-SAM" / "requirements.txt"
        if req_p.is_file():
            try:
                _run([uv_bin, "pip", "install", "--python", str(py_p3), "-r", str(req_p)], cwd=repo_p3 / "P3-SAM")
            except subprocess.CalledProcessError as exc:
                print(f"[P3-SAM] optional post-install note (requirements.txt): {exc}", flush=True)
        chamfer_dir = repo_p3 / "P3-SAM" / "utils" / "chamfer3D"
        if chamfer_dir.is_dir() and (chamfer_dir / "setup.py").is_file():
            try:
                _run([str(py_p3), "setup.py", "install"], cwd=chamfer_dir)
            except subprocess.CalledProcessError as exc:
                print(f"[P3-SAM] optional post-install note (chamfer3D): {exc}", flush=True)

    py_p3, repo_p3 = ensure_uv_env(
        "hunyuan3d_part",
        HUNYUAN3D_PART_REPO,
        python_version="3.10",
        install_cmds=[
            ["{UV}", "pip", "install", "--python", "{PY}", "torch==2.6.0", "torchvision==0.21.0", "--index-url", "https://download.pytorch.org/whl/cu124"],
            ["{UV}", "pip", "install", "--python", "{PY}", "trimesh", "open3d", "scipy", "numpy<2.0", "numba", "fpsample==0.3.3", "timm", "einops", "transformers==4.48.3", "huggingface_hub==0.28.1", "scikit-learn", "scikit-image", "omegaconf", "pymeshlab", "viser", "gradio", "ninja", "easydict", "pyyaml", "tqdm", "opencv-python-headless", "matplotlib", "safetensors", "addict", "spconv-cu124"],
            ["{UV}", "pip", "install", "--python", "{PY}", "torch_scatter", "-f", "https://data.pyg.org/whl/torch-2.6.0+cu124.html"],
        ],
        post_install_fn=_post_p3sam,
    )
    uv_bin = _resolve_uv()
    if not (
        _can_import_module(py_p3, "addict")
        and _can_import_module(py_p3, "spconv")
        and _can_import_module(py_p3, "torch_scatter")
        and _can_import_module(py_p3, "omegaconf")
        and _can_import_module(py_p3, "skimage")
    ):
        _run([uv_bin, "pip", "install", "--python", str(py_p3), "addict", "spconv-cu124", "omegaconf", "scikit-image"])
        _run([uv_bin, "pip", "install", "--python", str(py_p3), "torch_scatter", "-f", "https://data.pyg.org/whl/torch-2.6.0+cu124.html"])
    assert repo_p3 is not None
    return py_p3, repo_p3


def _install_unirig_env() -> tuple[Path, Path]:
    py_u, repo_u = ensure_uv_env(
        "unirig",
        UNIRIG_REPO,
        python_version="3.11",
        install_cmds=[
            ["{UV}", "pip", "install", "--python", "{PY}", "torch==2.4.0", "torchvision==0.19.0", "--index-url", "https://download.pytorch.org/whl/cu124"],
            ["{UV}", "pip", "install", "--python", "{PY}", "spconv-cu124", "numpy==1.26.4", "psutil"],
            ["{UV}", "pip", "install", "--python", "{PY}", "torch_scatter", "torch_cluster", "-f", "https://data.pyg.org/whl/torch-2.4.0+cu124.html"],
            ["{UV}", "pip", "install", "--python", "{PY}", "--no-deps", "https://github.com/Dao-AILab/flash-attention/releases/download/v2.6.3/flash_attn-2.6.3%2Bcu123torch2.4cxx11abiFALSE-cp311-cp311-linux_x86_64.whl"],
            ["bash", "-lc", "grep -viE '^flash[_-]attn' requirements.txt > /tmp/unirig_req.txt && {UV} pip install --python {PY} -r /tmp/unirig_req.txt"],
        ],
    )
    assert repo_u is not None
    return py_u, repo_u


def run_stages_1_to_5(
    input_dir: Path,
    canonical_glb: Path,
    out_dir: Path,
    *,
    stage: str = "all",
    resume: bool = True,
    complete_occluded_shoes: str = "auto",
    wheel_cache_dir: Path | str | None = None,
) -> dict[str, Any]:
    """Execute blueprint Stages 1-5 on Colab L4/A100 GPU."""
    out_dir.mkdir(parents=True, exist_ok=True)
    status_p = out_dir / "status.json"
    hw = setup_env_and_caches(wheel_cache_dir=wheel_cache_dir)
    _write_status(status_p, "running", "init", hw)

    def _find_view(role: str) -> str:
        cands = sorted(input_dir.glob(f"*{role}*"))
        if not cands:
            raise FileNotFoundError(f"Missing {role} view inside {input_dir}")
        return str(cands[0])

    inputs_cfg = {r: _find_view(r) for r in ("front", "right", "back", "left")}
    s1_dir = out_dir / "stage1"
    cutout_dir = s1_dir / "cutouts"
    sq_dir = s1_dir / "normalized_sq"
    norm_dir = s1_dir / "normals"
    sq_meta_p = sq_dir / "sq_meta.json"
    norm_meta_p = norm_dir / "normals_meta.json"

    # Stage 1: BiRefNet + Shared-Scale Normalization + StableNormal_turbo
    if stage in ("all", "stage1"):
        if not (resume and sq_meta_p.is_file() and norm_meta_p.is_file()):
            _write_status(status_p, "running", "stage1_birefnet_stablenormal", hw)
            s1_dir.mkdir(parents=True, exist_ok=True)
            in_json = s1_dir / "inputs.json"
            in_json.write_text(json.dumps(inputs_cfg, indent=2), encoding="utf-8")
            py_vis = _install_vision_env()
            sc1 = s1_dir / "_run_stage1.py"
            sc1.write_text(_STAGE1_BIREFNET_NORMAL_SCRIPT, encoding="utf-8")
            _run([str(py_vis), str(sc1), "--inputs-json", str(in_json), "--cutout-dir", str(cutout_dir), "--sq-dir", str(sq_dir), "--normal-dir", str(norm_dir)])

    # Stage 2: Multi-View TRELLIS.2-4B + o_voxel Branch 1 (remesh=False, 120k)
    s2_dir = out_dir / "stage2"
    stage2_glb = s2_dir / "stage2_trellis2_raw.glb"
    if stage in ("all", "stage2"):
        if not (resume and stage2_glb.is_file()):
            _write_status(status_p, "running", "stage2_trellis2_multiview", hw)
            s2_dir.mkdir(parents=True, exist_ok=True)
            py_tr, repo_tr = _install_trellis2_env(wheel_cache_dir=hw["wheel_cache_dir"])
            sc2 = s2_dir / "_run_stage2.py"
            sc2.write_text(_STAGE2_TRELLIS2_SCRIPT, encoding="utf-8")
            ptype = "1536_cascade" if hw["vram_gb"] >= 32.0 else "1024_cascade"
            _run(
                [str(py_tr), str(sc2), "--sq-meta", str(sq_meta_p), "--out-glb", str(stage2_glb), "--pipeline-type", ptype, "--decimation-target", "120000"],
                cwd=repo_tr,
                env_extra={"ATTN_BACKEND": "xformers", "SPARSE_ATTN_BACKEND": "xformers"},
            )

    # Stage 3: 100-step nvdiffrast + largesteps (Cholesky) silhouette & normal lock
    s3_dir = out_dir / "stage3"
    stage3_glb = s3_dir / "stage3_refined.glb"
    stage3_metrics = s3_dir / "stage3_silhouette_metrics.json"
    stage3_grid = s3_dir / "stage3_silhouette_grid.png"
    if stage in ("all", "stage3"):
        if not (resume and stage3_glb.is_file() and stage3_metrics.is_file()):
            _write_status(status_p, "running", "stage3_nvdiffrast_largesteps", hw)
            s3_dir.mkdir(parents=True, exist_ok=True)
            py_tr, _ = _install_trellis2_env(wheel_cache_dir=hw["wheel_cache_dir"])
            sc3 = s3_dir / "_run_stage3.py"
            sc3.write_text(_STAGE3_NVDIFFRAST_SCRIPT, encoding="utf-8")
            _run([str(py_tr), str(sc3), "--in-glb", str(stage2_glb), "--sq-meta", str(sq_meta_p), "--normals-meta", str(norm_meta_p), "--out-glb", str(stage3_glb), "--out-metrics", str(stage3_metrics), "--out-grid", str(stage3_grid), "--iters", "100"])

    # Stage 4: Native 3D Part Isolation via Hunyuan3D-Part (P3-SAM)
    s4_dir = out_dir / "stage4"
    p3sam_part_ids = s4_dir / "p3sam_part_ids.npy"
    p3sam_sem_ids = s4_dir / "p3sam_semantic_face_ids.npy"
    p3sam_rep = s4_dir / "p3sam_clusters_report.json"
    if stage in ("all", "stage4"):
        if not (resume and p3sam_part_ids.is_file() and p3sam_sem_ids.is_file()):
            _write_status(status_p, "running", "stage4_p3sam_parts", hw)
            s4_dir.mkdir(parents=True, exist_ok=True)
            py_p3, repo_p3 = _install_p3sam_env()
            sc4 = s4_dir / "_run_stage4.py"
            sc4.write_text(_STAGE4_P3SAM_SCRIPT, encoding="utf-8")
            _run([str(py_p3), str(sc4), "--in-glb", str(stage3_glb), "--repo-dir", str(repo_p3), "--out-dir", str(s4_dir), "--complete-occluded-shoes", complete_occluded_shoes])

    # Stage 5: UniRig on unbroken 3D character mesh + 1-to-1 52-bone J_Bip_* retarget
    s5_dir = out_dir / "stage5"
    unirig_raw_glb = s5_dir / "stage5_unirig_raw.glb"
    stage5_vroid_glb = s5_dir / "stage5_unirig_vroid.glb"
    if stage in ("all", "stage5"):
        s5_dir.mkdir(parents=True, exist_ok=True)
        if not (resume and unirig_raw_glb.is_file()):
            _write_status(status_p, "running", "stage5_unirig_inference", hw)
            py_u, repo_u = _install_unirig_env()
            sc5 = s5_dir / "_run_unirig.py"
            sc5.write_text(_STAGE5_UNIRIG_RUNNER, encoding="utf-8")
            _run(
                [
                    str(py_u), str(sc5),
                    "--input", str(stage3_glb),
                    "--skel-fbx", str(s5_dir / "unirig_skeleton.fbx"),
                    "--skin-fbx", str(s5_dir / "unirig_skin.fbx"),
                    "--out-glb", str(unirig_raw_glb),
                ],
                cwd=repo_u,
            )
        if not (resume and stage5_vroid_glb.is_file()):
            _write_status(status_p, "running", "stage5_vroid_retarget", hw)
            blender_bin = resolve_headless_blender()
            retarget_cfg = s5_dir / "retarget_cfg.json"
            retarget_cfg.write_text(json.dumps({
                "canonical_glb": str(canonical_glb.resolve()),
                "unirig_raw_glb": str(unirig_raw_glb.resolve()),
                "stage3_glb": str(stage3_glb.resolve()),
                "p3sam_semantic_face_ids": str(p3sam_sem_ids.resolve()),
                "out_glb": str(stage5_vroid_glb.resolve()),
            }, indent=2), encoding="utf-8")
            sc5b = s5_dir / "_run_retarget_blender.py"
            sc5b.write_text(_STAGE5_RETARGET_BLENDER_SCRIPT, encoding="utf-8")
            _run([str(blender_bin), "-b", "--factory-startup", "--python-exit-code", "1", "-P", str(sc5b), "--", str(retarget_cfg)])

    summary = {
        "hw": hw,
        "stage1_sq_meta": str(sq_meta_p) if sq_meta_p.is_file() else None,
        "stage1_normals_meta": str(norm_meta_p) if norm_meta_p.is_file() else None,
        "stage2_glb": str(stage2_glb) if stage2_glb.is_file() else None,
        "stage3_glb": str(stage3_glb) if stage3_glb.is_file() else None,
        "stage3_metrics": json.loads(stage3_metrics.read_text(encoding="utf-8")) if stage3_metrics.is_file() else None,
        "stage4_report": json.loads(p3sam_rep.read_text(encoding="utf-8")) if p3sam_rep.is_file() else None,
        "stage5_unirig_raw_glb": str(unirig_raw_glb) if unirig_raw_glb.is_file() else None,
        "stage5_vroid_glb": str(stage5_vroid_glb) if stage5_vroid_glb.is_file() else None,
    }
    _write_status(status_p, "completed", "done", summary)
    return summary


def main() -> int:
    ap = argparse.ArgumentParser(description="Colab GPU worker for Greenfield Stages 1-5")
    ap.add_argument("--input-dir", type=Path, required=True)
    ap.add_argument("--canonical-glb", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--stage", choices=("all", "stage1", "stage2", "stage3", "stage4", "stage5"), default="all")
    ap.add_argument("--no-resume", action="store_true")
    ap.add_argument("--complete-occluded-shoes", choices=("auto", "always", "never"), default="auto")
    ap.add_argument("--wheel-cache-dir", type=Path, default=None)
    args = ap.parse_args()

    try:
        res = run_stages_1_to_5(
            args.input_dir,
            args.canonical_glb,
            args.out_dir,
            stage=args.stage,
            resume=not args.no_resume,
            complete_occluded_shoes=args.complete_occluded_shoes,
            wheel_cache_dir=args.wheel_cache_dir,
        )
        print(json.dumps(res, indent=2), flush=True)
        return 0
    except Exception as exc:
        import traceback
        traceback.print_exc()
        _write_status(args.out_dir / "status.json", "failed", args.stage, {"error": str(exc)})
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
