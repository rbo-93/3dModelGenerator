---
trigger: always_on
---

# Google Colab CLI Rules (`research`)

When running commands or pipelines against Google Colab in this project, follow the global `colab-operator` skill (`~/.gemini/config/skills/colab-operator/SKILL.md`) and enforce these mandatory safeguards:

1. **Tool Selection & Explicit Paths:**
   - Use `C:\Users\rbo\.local\bin\colab.exe` (`google-colab-cli`) for headless/automated GPU runs (`colab-mcp` only bridges an open browser tab).
   - `%USERPROFILE%\.local\bin` is not on default Windows `PATH`; invoke `C:\Users\rbo\.local\bin\colab.exe` and `C:\Users\rbo\.local\bin\python3.15.exe` explicitly.

2. **Always Use the Safe `colab exec` Wrapper (Never Bare `colab exec`):**
   - Execute remote Python via `C:\Users\rbo\.gemini\config\skills\colab-operator\scripts\colab_exec_safe.py` (sets `REQUEST_TIMEOUT=60`, rejects UTF-8 BOMs, and force-kills wedged `colab.exe` WebSocket ping threads after `--proc-timeout`).
   - In Windows PowerShell 5.1, use `-f <script.py>` or `--stdin` (piping a single-quoted here-string) rather than `-c` whenever code spans multiple lines or contains double quotes.
   - Always write local `.py` files executed on Colab in UTF-8 **without BOM** (never PowerShell 5.1 `Set-Content -Encoding UTF8`).

3. **Headless Colab Environment Constraints:**
   - Never run TTY-interactive commands from an agent (`colab auth`, `colab drivemount`, `colab repl`, or bare `colab ssh` without `-- CMD`).
   - Do not depend on `google.colab.drive.mount` or `google.colab.userdata` under `colab exec`; pre-create remote parent directories on the VM before `colab upload`.
   - Set `MPLBACKEND=Agg` and `PYTHONUNBUFFERED=1` before running isolated `uv`/Python subprocesses on Colab.

4. **Long GPU Jobs, Monitoring, & Billing Cleanup:**
   - Always pass `-s <name>` explicitly on `colab new`.
   - Launch multi-minute GPU runs detached on the VM (`start_new_session=True`, writing a status JSON + log file), inspect files/logs over REST (`colab download` / `colab ls`), and prevent VM idle reclamation with periodic lightweight `colab_exec_safe.py` kernel heartbeats (~3–5 min via `schedule`).
   - Immediately after downloading outputs, run `colab stop -s <name>` and `colab usage` to verify `Usage rate: 0.00/hr` and `Active assignments: 0`.
