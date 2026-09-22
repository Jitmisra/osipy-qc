#!/usr/bin/env python3
"""Set QEI-Net up from the zip its authors sent you. macOS, Windows or Linux.

Run it from the repository root, with the zip sitting in `qei_net_model/`:

    python3 scripts/setup_qei_net.py        # macOS / Linux
    python  scripts/setup_qei_net.py        # Windows

It unpacks the zip, builds a SEPARATE Python environment for the model, proves
the model actually runs by scoring a volume, and prints the one command that
starts the tool with it wired in.

Why a separate environment: osipy-qc depends on numpy and nibabel and nothing
else, and that is a deliberate constraint of the project it belongs to. QEI-Net
needs torch, torchio and SimpleITK, roughly 700 MB. Keeping them apart is why
`1.1.qei_net` runs the model as a subprocess instead of importing it.

Nothing here uploads anything. The weights stay in the folder you put them in.

Safe to re-run. It checks the environment it finds rather than trusting that a
folder exists, and rebuilds when that check fails; `--rebuild` forces it.
"""

from __future__ import annotations

import argparse
import os
import pathlib
import shutil
import subprocess
import sys
import zipfile

ROOT = pathlib.Path(__file__).resolve().parent.parent
DROP = ROOT / "qei_net_model"
PKG = DROP / "package"
VENV = DROP / "qei_env"

#: Written only after an install that finished. Its absence means "not finished",
#: which is not the same question as "does the folder exist".
MARKER = ".osipy-setup-complete"

#: What the package's own code imports. Used both as the fallback install list
#: (the shipped requirements.txt pins an exact torch that has no wheel on every
#: platform and Python) and as the probe that decides whether an existing
#: environment is usable.
DEPS = ["torch", "torchio", "nibabel", "numpy", "pandas"]

#: The published model ensembles this many folds. Fewer is not "still fine":
#: run_qei.py defaults to --folds 0 1 2 3 4 and exits when one is missing.
N_FOLDS = 5


def say(msg: str = "") -> None:
    print(msg, flush=True)


def fail(msg: str, *hints: str) -> None:
    say()
    say(f"  STOPPED: {msg}")
    for h in hints:
        say(f"           {h}")
    say()
    sys.exit(1)


def py_cmd() -> str:
    """How to spell the interpreter in an instruction for THIS platform.

    macOS ships no `python`, only `python3`; the Windows installer is the other
    way round. Printing one of them to both audiences sends half the readers to
    a "command not found".
    """
    return "python" if os.name == "nt" else "python3"


def venv_python(venv: pathlib.Path) -> pathlib.Path:
    if os.name == "nt":
        return venv / "Scripts" / "python.exe"
    return venv / "bin" / "python"


def run(cmd: list[str], what: str, *hints: str, quiet: bool = False) -> None:
    """A subprocess whose failure is a sentence, not a traceback."""
    try:
        r = subprocess.run(cmd, capture_output=quiet, text=True)
    except OSError as exc:
        fail(f"could not {what}: {exc}", *hints)
    if r.returncode != 0:
        tail = []
        if quiet:
            tail = (r.stderr or r.stdout or "").strip().splitlines()[-4:]
        fail(f"could not {what}", *tail, *hints)


def find_zip(explicit: str | None) -> pathlib.Path:
    if explicit:
        p = pathlib.Path(explicit).expanduser().resolve()
        if not p.is_file():
            fail(f"no file at {p}")
        return p
    if not DROP.is_dir():
        fail(f"{DROP} does not exist",
             f"run this from the repository root: {py_cmd()} scripts/setup_qei_net.py")
    zips = sorted(p for p in DROP.glob("*.zip") if not p.name.startswith("."))
    if not zips:
        fail(f"no .zip in {DROP.name}/",
             "put the QEI-Net package zip in that folder and run this again, or",
             f"pass its path:  {py_cmd()} scripts/setup_qei_net.py /path/to/the.zip")
    if len(zips) > 1:
        fail(f"{len(zips)} zips in {DROP.name}/: " + ", ".join(z.name for z in zips),
             "leave only the one you want, or pass its path as an argument")
    return zips[0]


def stage_zip(zip_path: pathlib.Path) -> pathlib.Path:
    """Unpack to a staging folder. Nothing already installed is touched yet.

    Deliberately not extracted over the real folder. Doing that deleted a
    working install BEFORE discovering the new zip was truncated, leaving no
    model at all and a BadZipFile traceback to explain it.
    """
    stage = DROP / "_unpacking"
    if stage.exists():
        shutil.rmtree(stage, ignore_errors=True)
    stage.mkdir(parents=True)

    try:
        zf = zipfile.ZipFile(zip_path)
    except (zipfile.BadZipFile, OSError) as exc:
        shutil.rmtree(stage, ignore_errors=True)
        fail(f"{zip_path.name} is not a readable zip ({exc})",
             "it may have been truncated in transit - ask for it again")
    with zf:
        bad = zf.testzip()
        if bad is not None:
            shutil.rmtree(stage, ignore_errors=True)
            fail(f"{zip_path.name} is damaged (first bad entry: {bad})",
                 "download or copy it again")
        for info in zf.infolist():
            name = info.filename
            parts = pathlib.PurePosixPath(name).parts
            # `__MACOSX/`, `._x` and `.DS_Store` are what a Mac adds when zipping.
            # Unpacked they sit beside the real files and confuse every glob.
            if any(p == "__MACOSX" or p.startswith("._") or p == ".DS_Store"
                   for p in parts):
                continue
            # A zip entry can name ../ and be written outside the target: zip-slip.
            # Compared as PATHS, because a string prefix test accepts /a/bc for a
            # destination of /a/b.
            target = (stage / name).resolve()
            try:
                target.relative_to(stage.resolve())
            except ValueError:
                shutil.rmtree(stage, ignore_errors=True)
                fail(f"the zip contains an entry that escapes the folder: {name!r}",
                     "this is not a normal package; do not use it")
            zf.extract(info, stage)

    kids = [p for p in stage.iterdir() if not p.name.startswith(".")]
    return kids[0] if (len(kids) == 1 and kids[0].is_dir()) else stage


def check_layout(pkg: pathlib.Path) -> tuple[pathlib.Path, list[str]]:
    """Find run_qei.py and the folds, or say plainly what is missing."""
    script = pkg / "src" / "run_qei.py"
    if not script.is_file():
        # tolerate one extra level of nesting rather than reject a good package
        found = sorted(pkg.rglob("src/run_qei.py"))
        if not found:
            fail("this zip does not contain src/run_qei.py",
                 "it does not look like the QEI-Net inference package")
        script = found[0]
    root = script.parent.parent
    weights = sorted(root.rglob("*.pth"))
    if not weights:
        fail(f"the package unpacked but carries no .pth weight files",
             "ask the authors whether the weights were included -",
             "the package cannot do anything without them")
    folds = sorted({w.parent.name for w in weights})
    return script, folds


def fold_args(folds: list[str]) -> list[str]:
    """`--folds K ...` when the package is not the full published ensemble.

    run_qei.py defaults to folds 0 1 2 3 4 and exits if one is absent, so a
    partial package does NOT "still run" - it fails on the next line. Passing
    what is actually there lets it run and makes the difference explicit.
    """
    nums = sorted(int(f[4:]) for f in folds if f.startswith("fold") and f[4:].isdigit())
    if len(nums) == N_FOLDS or not nums:
        return []
    return ["--folds", *(str(n) for n in nums)]


def env_is_usable(py: pathlib.Path) -> bool:
    """Can this environment actually run the model?

    Asked by importing, not by looking for a file. `python -m venv` succeeds in
    seconds and creates the interpreter; every pip call after it can fail on its
    own. Testing only that the interpreter exists meant one failed install
    poisoned the folder permanently: every later run printed "reusing the
    environment", installed nothing, and died in a traceback with no hint that
    the cure was to delete a folder by hand.
    """
    if not py.is_file():
        return False
    probe = subprocess.run([str(py), "-c", "import " + ", ".join(DEPS)],
                           capture_output=True, text=True)
    return probe.returncode == 0


def build_env(pkg: pathlib.Path, rebuild: bool) -> pathlib.Path:
    py = venv_python(VENV)
    if not rebuild and env_is_usable(py):
        say(f"  the environment at {VENV.name}/ already works - reusing it")
        return py
    if VENV.exists():
        say(f"  the environment at {VENV.name}/ is incomplete - rebuilding it")
        shutil.rmtree(VENV, ignore_errors=True)

    say(f"  creating a Python environment at {DROP.name}/{VENV.name}")
    say("  this downloads torch, about 700 MB - expect several minutes")
    say()
    run([sys.executable, "-m", "venv", str(VENV)], "create the environment")
    py = venv_python(VENV)
    run([str(py), "-m", "pip", "install", "--upgrade", "pip"],
        "update pip", quiet=True)

    req = pkg / "requirements.txt"
    ok = False
    if req.is_file():
        say("  installing the package's pinned requirements")
        # progress deliberately NOT suppressed: this is the long step, and a
        # console that sits silent for twenty minutes is one a reader interrupts
        ok = subprocess.run([str(py), "-m", "pip", "install", "-r", str(req)]).returncode == 0
        if not ok:
            say()
            say("  those exact versions do not exist for this Python - that is")
            say("  expected on some platforms. Installing what the code needs instead.")
    if not ok:
        say(f"  installing {', '.join(DEPS)}")
        run([str(py), "-m", "pip", "install", *DEPS], "install the model's dependencies",
            "check the internet connection and run this again")

    if not env_is_usable(py):
        fail("the dependencies installed but cannot be imported",
             f"delete {DROP.name}/{VENV.name} and run this again")
    (VENV / MARKER).write_text("ok\n", encoding="utf-8")
    return py


def smoke_test(py: pathlib.Path, script: pathlib.Path, folds: list[str]) -> str:
    """Actually score a volume. Anything less is a guess that it works.

    The volume is deliberately OFF the model's own 2.5 mm 96x96x64 grid, so the
    resampling path runs and torchio is exercised. An on-grid volume takes a
    shortcut past both, and "it works" would then be true only of the one case
    no real CBF map is in.
    """
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        # An explicit mask, because --derive_mask_from_cbf is only legal for a
        # volume already on the model's own grid - the crop to 96x96x64 is
        # centred on the mask, so without one an off-grid input cannot be
        # positioned at all. Supplying the mask is what lets the test be
        # off-grid, which is the whole point: that is the path every real CBF
        # map takes.
        make = (
            "import numpy as np, nibabel as nib, sys\n"
            "rng = np.random.RandomState(0)\n"
            "shape, aff = (64, 64, 40), np.diag([3.0, 3.0, 3.0, 1.0])\n"
            "a = np.zeros(shape, np.float32)\n"
            "a[12:52, 12:52, 6:34] = np.abs(rng.normal(50, 8, (40, 40, 28)))\n"
            "nib.save(nib.Nifti1Image(a, aff), sys.argv[1])\n"
            "m = np.zeros(shape, np.uint8); m[12:52, 12:52, 6:34] = 1\n"
            "nib.save(nib.Nifti1Image(m, aff), sys.argv[2])\n"
        )
        cbf = os.path.join(tmp, "smoke.nii.gz")
        mask = os.path.join(tmp, "smoke_mask.nii.gz")
        r = subprocess.run([str(py), "-c", make, cbf, mask],
                           capture_output=True, text=True)
        if r.returncode != 0:
            fail("could not build the test volume",
                 *(r.stderr or "").strip().splitlines()[-3:])
        try:
            out = subprocess.run([str(py), str(script), "--single", cbf,
                                  "--mask", mask, *fold_args(folds)],
                                 capture_output=True, text=True, timeout=1800)
        except subprocess.TimeoutExpired:
            fail("the model did not finish within 30 minutes",
                 "this machine may be too slow, or the model is stuck")
        if out.returncode != 0:
            tail = (out.stderr or out.stdout or "").strip().splitlines()[-6:]
            fail("the model did not run", *(tail or ["no output"]))
        try:
            return f"{float(out.stdout.strip().splitlines()[-1]):.3f}"
        except (ValueError, IndexError):
            fail("the model ran but printed no score", (out.stdout or "")[-200:])
    return ""


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("zip", nargs="?",
                    help=f"path to the zip (default: look in {DROP.name}/)")
    ap.add_argument("--rebuild", action="store_true",
                    help="discard the existing model environment and build it again")
    ap.add_argument("--skip-test", action="store_true",
                    help="do not run the model afterwards (not recommended)")
    args = ap.parse_args()

    say()
    say("  QEI-Net setup")
    say("  " + "-" * 54)

    zip_path = find_zip(args.zip)
    say(f"  found  {zip_path.name}  ({zip_path.stat().st_size / 1e6:.0f} MB)")

    staged = stage_zip(zip_path)
    script, folds = check_layout(staged)          # validated BEFORE anything is replaced

    # Only now, with the new package validated, is the old one replaced.
    if PKG.exists():
        shutil.rmtree(PKG, ignore_errors=True)
    shutil.move(str(staged), str(PKG))
    shutil.rmtree(DROP / "_unpacking", ignore_errors=True)
    script = next(iter(sorted(PKG.rglob("src/run_qei.py"))), None)
    if script is None:                       # cannot happen: check_layout just found it
        fail("the package moved but run_qei.py is no longer where it was")

    weights = sorted(PKG.rglob("*.pth"))
    say(f"  unpacked to {DROP.name}/{PKG.name}")
    say(f"  weights: {len(weights)} file(s) across {len(folds)} fold(s) "
        f"({sum(w.stat().st_size for w in weights) / 1e6:.0f} MB)")
    if len(folds) < N_FOLDS:
        say(f"  NOTE: the published model ensembles {N_FOLDS} folds; this package has "
            f"{len(folds)} ({', '.join(folds)}).")
        say("        It will be run with those, so the score is NOT the published one.")
        say("        Ask the authors for the complete set if you need comparable numbers.")

    py = build_env(PKG, args.rebuild)

    if args.skip_test:
        say("  skipping the run test, as asked - nothing has proved the model works")
    else:
        say("  running the model on a test volume")
        say(f"  it works - scored the test volume {smoke_test(py, script, folds)}")

    say()
    say("  " + "-" * 54)
    say("  Done. Start the tool with:")
    say()
    say(f"      {py_cmd()} scripts/run_ui.py")
    say()
    say("  If you would rather set it up by hand, these are the two variables:")
    say()
    if os.name == "nt":
        say(f'      $env:OSIPY_QEI_NET_PYTHON = "{py}"')
        say(f'      $env:OSIPY_QEI_NET_SCRIPT = "{script}"')
    else:
        say(f'      export OSIPY_QEI_NET_PYTHON="{py}"')
        say(f'      export OSIPY_QEI_NET_SCRIPT="{script}"')
    say()
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        say()
        say("  Interrupted. Nothing is broken - run this again to carry on.")
        sys.exit(130)
    except Exception as exc:                     # noqa: BLE001 - the whole point
        # A traceback is not an answer for the audience this is written for.
        fail(f"{type(exc).__name__}: {exc}",
             f"if this keeps happening, delete {DROP.name}/qei_env and run it again")
