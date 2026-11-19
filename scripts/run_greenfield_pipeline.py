#!/usr/bin/env python3
"""End-to-End Orchestrator for the Greenfield Best-Practice Pipeline.

Supports:
  - Full automated execution (`--mode full`)
  - Granular Colab lifecycle (`--mode launch-colab`, `--mode status`, `--mode download-colab`)
  - Local Blender 5.3.0 Stage 6 + evaluation gates (`--mode local-stage6`)
  - Optional turnaround sheet splitting (`--sheet <path>`)

Enforces `.agents/rules/colab-cli.md` safeguards:
  - Explicit `colab.exe`, `python3.15.exe`, and `colab_exec_safe.py` paths
  - Detached remote GPU execution (`start_new_session=True`)
  - REST `colab download` artifact retrieval
  - Mandatory `colab stop` + `colab usage` verification in `finally`
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = Path(__file__).resolve().parent
DEFAULT_CANONICAL_GLB = REPO_ROOT / "assets" / "base_bodies" / "anime_vroid_base_v1.glb"
DEFAULT_BLENDER_EXE = Path(
    os.environ.get("BLENDER_EXE", r"C:\Program Files (x86)\Steam\steamapps\common\Blender\blender.exe")
)
DEFAULT_COLAB_EXE = Path(os.environ.get("COLAB_EXE", str(Path.home() / ".local" / "bin" / "colab.exe")))
DEFAULT_COLAB_PY = Path(os.environ.get("COLAB_PYTHON", str(Path.home() / ".local" / "bin" / "python3.15.exe")))
DEFAULT_COLAB_EXEC_SAFE = Path(
    os.environ.get(
        "COLAB_EXEC_SAFE",
        str(Path.home() / ".gemini" / "config" / "skills" / "colab-operator" / "scripts" / "colab_exec_safe.py"),
    )
)

sys.path.insert(0, str(SCRIPTS_DIR))
from bake_flat_cel_albedo import bake_flat_cel_albedo  # noqa: E402
from eval_greenfield_pipeline import evaluate_stage_a_glb  # noqa: E402
from split_turnaround_sheet import split_turnaround_sheet  # noqa: E402
from stage6_blender_retopo_uv_bake import run_stage6_retopo_uv  # noqa: E402


def _colab_cli(colab_exe: Path, args: list[str]) -> str:
    cmd = [str(colab_exe), *args]
    print(f"$ {' '.join(cmd)}", flush=True)
    proc = subprocess.run(cmd, text=True, capture_output=True)
    if proc.returncode != 0:
        raise RuntimeError(f"colab command failed (rc={proc.returncode}):\n{proc.stdout}\n{proc.stderr}")
    return proc.stdout


def _colab_exec(colab_py: Path, exec_safe: Path, session: str, code: str, proc_timeout: int = 60) -> str:
    cmd = [
        str(colab_py),
        str(exec_safe),
        "-s",
        session,
        "--proc-timeout",
        str(proc_timeout),
        "--colab-timeout",
        str(max(20, proc_timeout - 15)),
        "--stdin",
    ]
    env = os.environ.copy()
    env["REQUEST_TIMEOUT"] = "60"
    proc = subprocess.run(cmd, input=code, text=True, capture_output=True, env=env)
    if proc.returncode != 0:
        raise RuntimeError(f"colab_exec_safe failed (rc={proc.returncode}):\n{proc.stdout}\n{proc.stderr}")
    return proc.stdout


def launch_colab_stages(
    *,
    char_id: str,
    input_dir: Path,
    canonical_glb: Path,
    session: str,
    gpu: str,
    reuse_session: bool,
    complete_occluded_shoes: str,
    colab_exe: Path,
    colab_py: Path,
    exec_safe: Path,
) -> dict[str, Any]:
    remote_root = f"/content/greenfield_jobs/{char_id}"
    if not reuse_session:
        _colab_cli(colab_exe, ["new", "-s", session, "--gpu", gpu])

    _colab_exec(
        colab_py,
        exec_safe,
        session,
        (
            "import pathlib; "
            f"[pathlib.Path(p).mkdir(parents=True, exist_ok=True) for p in ("
            f"'{remote_root}/inputs', '{remote_root}/assets', '{remote_root}/out', "
            "'/content/greenfield_workspace/wheel_cache')]"
        ),
    )

    _colab_cli(colab_exe, ["upload", "-s", session, str(SCRIPTS_DIR / "greenfield_colab_worker.py"), "/content/greenfield_colab_worker.py"])
    _colab_cli(colab_exe, ["upload", "-s", session, str(canonical_glb), f"{remote_root}/assets/anime_vroid_base_v1.glb"])
    hf_tok = Path.home() / ".cache" / "huggingface" / "token"
    if hf_tok.is_file():
        _colab_cli(colab_exe, ["upload", "-s", session, str(hf_tok), "/content/greenfield_workspace/hf_token"])
    for role in ("front", "right", "back", "left"):
        cands = sorted(input_dir.glob(f"*{role}*.png"))
        if not cands:
            raise FileNotFoundError(f"Missing {role} PNG in {input_dir}")
        _colab_cli(colab_exe, ["upload", "-s", session, str(cands[0]), f"{remote_root}/inputs/{cands[0].name}"])

    remote_cmd = [
        "python3",
        "-u",
        "/content/greenfield_colab_worker.py",
        "--input-dir",
        f"{remote_root}/inputs",
        "--canonical-glb",
        f"{remote_root}/assets/anime_vroid_base_v1.glb",
        "--out-dir",
        f"{remote_root}/out",
        "--complete-occluded-shoes",
        complete_occluded_shoes,
    ]
    launcher = (
        "import json, os, subprocess, pathlib\n"
        f"pathlib.Path('{remote_root}/out').mkdir(parents=True, exist_ok=True)\n"
        f"log_f = open('{remote_root}/out/worker.log', 'w', buffering=1)\n"
        "env = os.environ.copy()\n"
        "env['MPLBACKEND'] = 'Agg'\n"
        "env['PYTHONUNBUFFERED'] = '1'\n"
        f"p = subprocess.Popen({remote_cmd!r}, start_new_session=True, stdin=subprocess.DEVNULL, stdout=log_f, stderr=subprocess.STDOUT, close_fds=True, env=env)\n"
        "print(json.dumps({'launched_pid': p.pid}))\n"
    )
    launch_out = _colab_exec(colab_py, exec_safe, session, launcher, proc_timeout=60)
    print(launch_out, flush=True)
    return {"session": session, "remote_root": remote_root, "launch_out": launch_out.strip()}


def download_colab_outputs(
    *,
    char_id: str,
    out_dir: Path,
    session: str,
    colab_exe: Path,
) -> Path:
    remote_out = f"/content/greenfield_jobs/{char_id}/out"
    local_colab_dir = out_dir / "colab_stages"
    local_colab_dir.mkdir(parents=True, exist_ok=True)

    rel_files = (
        "status.json",
        "worker.log",
        "stage1/normalized_sq/sq_meta.json",
        "stage1/cutouts/view_00_front_rgba.png",
        "stage1/cutouts/view_01_right_rgba.png",
        "stage1/cutouts/view_02_back_rgba.png",
        "stage1/cutouts/view_03_left_rgba.png",
        "stage1/normalized_sq/view_00_front_sq.png",
        "stage1/normalized_sq/view_01_right_sq.png",
        "stage1/normalized_sq/view_02_back_sq.png",
        "stage1/normalized_sq/view_03_left_sq.png",
        "stage1/normals/normals_meta.json",
        "stage1/normals/normal_00_front.png",
        "stage1/normals/normal_01_right.png",
        "stage1/normals/normal_02_back.png",
        "stage1/normals/normal_03_left.png",
        "stage2/stage2_trellis2_raw.glb",
        "stage3/stage3_refined.glb",
        "stage3/stage3_silhouette_metrics.json",
        "stage3/stage3_silhouette_grid.png",
        "stage4/p3sam_part_ids.npy",
        "stage4/p3sam_semantic_face_ids.npy",
        "stage4/p3sam_clusters_report.json",
        "stage4/stage4_p3sam_clusters.png",
        "stage5/stage5_unirig_raw.glb",
        "stage5/stage5_unirig_vroid.glb",
    )
    for rel in rel_files:
        dst = local_colab_dir / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        try:
            _colab_cli(colab_exe, ["download", "-s", session, f"{remote_out}/{rel}", str(dst)])
        except Exception as exc:
            print(f"[download skip] {rel}: {exc}", flush=True)
    return local_colab_dir


def run_local_stage6_and_eval(
    *,
    char_id: str,
    out_dir: Path,
    fixture_dir: Path,
    blender_exe: Path,
    target_faces: int = 36500,
) -> dict[str, Any]:
    colab_dir = out_dir / "colab_stages"
    stage5_glb = colab_dir / "stage5" / "stage5_unirig_vroid.glb"
    if not stage5_glb.is_file():
        raise FileNotFoundError(f"Missing Stage 5 GLB: {stage5_glb}")

    # Prefer Stage 1 BiRefNet RGBA cutouts for crisp projection (falling back to fixture dir)
    cutouts_dir = colab_dir / "stage1" / "cutouts"
    view_paths = {}
    for role, fn in (
        ("front", "view_00_front"),
        ("right", "view_01_right"),
        ("back", "view_02_back"),
        ("left", "view_03_left"),
    ):
        cand_cut = cutouts_dir / f"{fn}_rgba.png"
        cand_fix = fixture_dir / f"{fn}.png"
        view_paths[role] = cand_cut if cand_cut.is_file() else cand_fix

    s6_dir = out_dir / "stage6"
    s6_geom_glb = s6_dir / f"{char_id}_stage6_geom.glb"
    retopo_rep = run_stage6_retopo_uv(
        stage5_glb,
        s6_geom_glb,
        out_report=s6_dir / "stage6_retopo_report.json",
        target_faces=target_faces,
        blender_exe=blender_exe,
    )

    final_glb = out_dir / f"{char_id}_stage_a_raw.glb"
    stage2_glb = colab_dir / "stage2" / "stage2_trellis2_raw.glb"
    bake_rep = bake_flat_cel_albedo(
        glb_path=s6_geom_glb,
        out_glb=final_glb,
        view_paths=view_paths,
        tex_dir=s6_dir / "textures",
        base_glb=stage2_glb if stage2_glb.is_file() else None,
        delight_shadows=True,
        body_res=2048,
        part_res=1024,
        blender_exe=blender_exe,
    )
    (s6_dir / "stage6_bake_report.json").write_text(json.dumps(bake_rep, indent=2), encoding="utf-8")

    eval_rep = evaluate_stage_a_glb(final_glb, fixture_dir, out_dir, char_id=char_id, blender_exe=blender_exe)
    return {"retopo": retopo_rep, "bake": bake_rep, "eval": eval_rep}


def main() -> int:
    ap = argparse.ArgumentParser(description="Run Greenfield Best-Practice Pipeline (Colab L4 + Local Blender 5.3.0)")
    ap.add_argument("--char-id", default="valkyrie")
    ap.add_argument("--sheet", type=Path, default=None, help="Optional 3-view or 4-view turnaround sheet to split")
    ap.add_argument("--input-dir", type=Path, default=None)
    ap.add_argument("--fixture-dir", type=Path, default=None)
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--canonical-glb", type=Path, default=DEFAULT_CANONICAL_GLB)
    ap.add_argument("--mode", choices=("full", "launch-colab", "status", "download-colab", "local-stage6"), default="full")
    ap.add_argument("--gpu", default="L4")
    ap.add_argument("--session", default=None)
    ap.add_argument("--reuse-session", action="store_true")
    ap.add_argument("--keep-session", action="store_true")
    ap.add_argument("--complete-occluded-shoes", choices=("auto", "always", "never"), default="auto")
    ap.add_argument("--target-faces", type=int, default=36500)
    ap.add_argument("--blender-exe", type=Path, default=DEFAULT_BLENDER_EXE)
    ap.add_argument("--colab-exe", type=Path, default=DEFAULT_COLAB_EXE)
    ap.add_argument("--colab-py", type=Path, default=DEFAULT_COLAB_PY)
    ap.add_argument("--colab-exec-safe", type=Path, default=DEFAULT_COLAB_EXEC_SAFE)
    args = ap.parse_args()

    cid = args.char_id
    input_dir = args.input_dir or (REPO_ROOT / "inputs" / f"{cid}_4view")
    fixture_dir = args.fixture_dir or (REPO_ROOT / "eval_fixtures" / f"{cid}_4view")
    out_dir = args.out_dir or (REPO_ROOT / "exports" / cid)
    session = args.session or f"greenfield-{cid}"

    if args.sheet is not None:
        split_turnaround_sheet(args.sheet, out_inputs_dir=input_dir, out_fixture_dir=fixture_dir, char_id=cid)

    if args.mode == "launch-colab":
        info = launch_colab_stages(
            char_id=cid,
            input_dir=input_dir,
            canonical_glb=args.canonical_glb,
            session=session,
            gpu=args.gpu,
            reuse_session=args.reuse_session,
            complete_occluded_shoes=args.complete_occluded_shoes,
            colab_exe=args.colab_exe,
            colab_py=args.colab_py,
            exec_safe=args.colab_exec_safe,
        )
        print(json.dumps(info, indent=2))
        return 0

    if args.mode == "status":
        tmp_s = out_dir / "colab_stages" / "status.json"
        tmp_l = out_dir / "colab_stages" / "worker.log"
        tmp_s.parent.mkdir(parents=True, exist_ok=True)
        for rf, lf in ((f"/content/greenfield_jobs/{cid}/out/status.json", tmp_s), (f"/content/greenfield_jobs/{cid}/out/worker.log", tmp_l)):
            try:
                _colab_cli(args.colab_exe, ["download", "-s", session, rf, str(lf)])
            except Exception as exc:
                print(f"[status warn] {rf}: {exc}", flush=True)
        if tmp_s.is_file():
            print(tmp_s.read_text(encoding="utf-8"))
        return 0

    if args.mode == "download-colab":
        d = download_colab_outputs(char_id=cid, out_dir=out_dir, session=session, colab_exe=args.colab_exe)
        if not args.keep_session:
            _colab_cli(args.colab_exe, ["stop", "-s", session])
            print(_colab_cli(args.colab_exe, ["usage"]))
        print(json.dumps({"downloaded_to": str(d)}, indent=2))
        return 0

    if args.mode == "local-stage6":
        res = run_local_stage6_and_eval(
            char_id=cid,
            out_dir=out_dir,
            fixture_dir=fixture_dir,
            blender_exe=args.blender_exe,
            target_faces=args.target_faces,
        )
        print(json.dumps(res, indent=2))
        return 0

    # Full synchronous mode (for standalone terminal invocation)
    started = False
    try:
        launch_colab_stages(
            char_id=cid,
            input_dir=input_dir,
            canonical_glb=args.canonical_glb,
            session=session,
            gpu=args.gpu,
            reuse_session=args.reuse_session,
            complete_occluded_shoes=args.complete_occluded_shoes,
            colab_exe=args.colab_exe,
            colab_py=args.colab_py,
            exec_safe=args.colab_exec_safe,
        )
        started = not args.reuse_session

        tmp_s = out_dir / "colab_stages" / "status.json"
        tmp_s.parent.mkdir(parents=True, exist_ok=True)
        t0, last_hb = time.time(), time.time()
        while time.time() - t0 < 3600:
            try:
                if tmp_s.exists():
                    tmp_s.unlink()
                _colab_cli(args.colab_exe, ["download", "-s", session, f"/content/greenfield_jobs/{cid}/out/status.json", str(tmp_s)])
                st = json.loads(tmp_s.read_text(encoding="utf-8"))
                print(f"[Colab status] {st.get('state')} / {st.get('step')}", flush=True)
                if st.get("state") == "completed":
                    break
                if st.get("state") in ("failed", "error"):
                    raise RuntimeError(f"Colab worker failed: {st}")
            except RuntimeError:
                raise
            except Exception:
                pass
            if time.time() - last_hb >= 180:
                try:
                    _colab_exec(args.colab_py, args.colab_exec_safe, session, "import time; print('hb', int(time.time()))")
                except Exception:
                    pass
                last_hb = time.time()
            time.sleep(45)

        download_colab_outputs(char_id=cid, out_dir=out_dir, session=session, colab_exe=args.colab_exe)
    finally:
        if started and not args.keep_session:
            try:
                _colab_cli(args.colab_exe, ["stop", "-s", session])
                print(_colab_cli(args.colab_exe, ["usage"]))
            except Exception as stop_exc:
                print(f"[WARN] Failed stopping Colab session {session}: {stop_exc}", flush=True)

    res = run_local_stage6_and_eval(
        char_id=cid,
        out_dir=out_dir,
        fixture_dir=fixture_dir,
        blender_exe=args.blender_exe,
        target_faces=args.target_faces,
    )
    print(json.dumps(res, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
