#!/usr/bin/env python3
"""Start the osipy-qc web console, with QEI-Net wired in if it is installed.

    python3 scripts/run_ui.py                # macOS / Linux
    python  scripts/run_ui.py                # Windows
    python3 scripts/run_ui.py --port 8123
    python3 scripts/run_ui.py --no-browser

If `setup_qei_net.py` has been run, this finds the model and sets the two
environment variables for you. If it has not, everything else still runs and
`1.1.qei_net` reports N/A, which does not count against the report's coverage.
"""

from __future__ import annotations

import argparse
import os
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
DROP = ROOT / "qei_net_model"
DEPS = ["torch", "torchio", "nibabel", "numpy", "pandas"]


def say(msg: str = "") -> None:
    # Flushed, always. Python block-buffers stdout when it is not a terminal and
    # this process then hands stdout to the server, so without this the
    # "QEI-Net: ON" line - the one thing the reader is here to see - disappears
    # whenever the output is piped or redirected.
    print(msg, flush=True)


def py_cmd() -> str:
    """macOS ships no `python`, only `python3`; Windows is the other way round."""
    return "python" if os.name == "nt" else "python3"


def venv_python(venv: pathlib.Path) -> pathlib.Path:
    if os.name == "nt":
        return venv / "Scripts" / "python.exe"
    return venv / "bin" / "python"


def _runnable(py: pathlib.Path) -> bool:
    """Can that interpreter actually import the model's dependencies?

    Asked by importing. Reporting "QEI-Net: ON" because two filenames exist was
    worse than reporting nothing: a half-built environment passed that test, the
    banner said ON, and then every single scan came back UNKNOWN "No module
    named 'torch'".
    """
    if not py.is_file():
        return False
    return subprocess.run([str(py), "-c", "import " + ", ".join(DEPS)],
                          capture_output=True).returncode == 0


def find_model() -> tuple[str, str] | None:
    """(interpreter, script) for QEI-Net, or None if it is not usable.

    An environment the operator set explicitly wins: someone who exported these
    knows where their model is, and silently preferring a copy in the drop
    folder would be worse than not looking.
    """
    env_py, env_sc = (os.environ.get("OSIPY_QEI_NET_PYTHON"),
                      os.environ.get("OSIPY_QEI_NET_SCRIPT"))
    if env_py and env_sc and os.path.isfile(env_py) and os.path.isfile(env_sc):
        return env_py, env_sc

    if not DROP.is_dir():
        return None
    py = venv_python(DROP / "qei_env")
    scripts = sorted(s for s in DROP.rglob("run_qei.py")
                     # the staging folder of an interrupted unpack is not a model.
                     # Tested per path SEGMENT: "_unpacking" never equals the
                     # segment string when the check is `not in parts` and the
                     # segment is e.g. "package.unpacking".
                     if not any(seg.endswith("_unpacking") for seg in s.parts))
    if not scripts:
        return None
    # weights, not just the script: a package whose weights were left out
    # unpacks perfectly and then fails on every scan.
    if not any(scripts[0].parent.parent.rglob("*.pth")):
        return None
    return (str(py), str(scripts[0])) if _runnable(py) else None


def half_built() -> bool:
    """Is there a model folder that is present but not usable?

    Worth a different message from "not set up": one means do the setup, the
    other means the setup did not finish.
    """
    return (DROP / "qei_env").is_dir() and not _runnable(venv_python(DROP / "qei_env"))


def interpreter_for_osipy() -> str:
    """Prefer the project's own venv, so a reader who made one does not have to
    remember to activate it first."""
    local = venv_python(ROOT / ".venv")
    return str(local) if local.is_file() else sys.executable


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--no-browser", action="store_true")
    args = ap.parse_args()

    env = dict(os.environ)
    found = find_model()
    say()
    if found:
        env["OSIPY_QEI_NET_PYTHON"], env["OSIPY_QEI_NET_SCRIPT"] = found
        say("  QEI-Net: ON")
    elif half_built():
        say("  QEI-Net: found, but its environment is incomplete - 1.1.qei_net will")
        say("           read N/A. Finish it with:")
        say(f"               {py_cmd()} scripts/setup_qei_net.py --rebuild")
    else:
        say("  QEI-Net: off - not set up, so 1.1.qei_net will read N/A.")
        say("           Everything else runs normally. To enable it, put the")
        say("           package zip in qei_net_model/ and run:")
        say(f"               {py_cmd()} scripts/setup_qei_net.py")

    py = interpreter_for_osipy()
    # Checked BEFORE announcing a URL. Printing "Opening http://..." and then
    # dying on "No module named osipy_qc" sends the reader to a browser tab that
    # will never load, with the real error scrolled off above it.
    probe = subprocess.run([py, "-c", "import osipy_qc"], capture_output=True)
    if probe.returncode != 0:
        say()
        say("  STOPPED: osipy-qc is not installed in this Python.")
        say("           From the repository root, run:")
        say(f"               {py_cmd()} -m pip install -e .")
        say()
        return 1

    cmd = [py, "-m", "osipy_qc", "--serve", "--host", args.host, "--port", str(args.port)]
    if args.no_browser:
        cmd.append("--no-browser")
    say(f"  Opening http://{args.host}:{args.port}")
    say("  Press Ctrl-C to stop.")
    say()
    try:
        return subprocess.call(cmd, env=env)
    except KeyboardInterrupt:
        return 0


if __name__ == "__main__":
    sys.exit(main())
