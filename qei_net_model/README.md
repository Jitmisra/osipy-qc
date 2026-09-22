# Put the QEI-Net package here

This folder is deliberately empty. **QEI-Net is not part of osipy-qc.** Its
weights are unpublished work by the model's authors and are not ours to
distribute, so nothing about them is in this repository — you supply the package
yourself and it stays on your own machine.

Everything below takes two commands.

---

## Before you start

You need the repository installed. If you have not done that yet, from the
folder **above this one** (the repository root):

```bash
python3 -m venv .venv                 # Windows: python -m venv .venv
source .venv/bin/activate             # Windows: .venv\Scripts\activate
pip install -e .
```

macOS and Linux have `python3` and usually no `python` at all; the Windows
installer gives you `python`. Use whichever your system has — the scripts below
print the right one back to you.

---

## Setting up the model

1. Get `qei_inference_package.zip` from the model's authors.

2. Put it **in this folder**. Do not unzip it; the script does that.

3. From the repository root:

   ```bash
   python3 scripts/setup_qei_net.py   # Windows: python scripts\setup_qei_net.py
   ```

   It unpacks the zip, builds a separate Python environment for the model, and
   then **runs the model on a test volume** to prove it works. The environment
   downloads PyTorch, about 700 MB, so the first run takes several minutes and
   prints progress while it does.

   You should see this at the end:

   ```
     weights: 5 file(s) across 5 fold(s) (72 MB)
     running the model on a test volume
     it works - scored the test volume 0.731
   ```

   The score itself is meaningless — it is a synthetic volume. What matters is
   that a number came back at all.

4. Start the tool:

   ```bash
   python3 scripts/run_ui.py          # Windows: python scripts\run_ui.py
   ```

   It should say **`QEI-Net: ON`** and open <http://127.0.0.1:8000>.

Drop a scan in, and `1.1.qei_net` now reports a score.

---

## If something goes wrong

The scripts stop with a sentence beginning `STOPPED:` rather than a Python
traceback, and the sentence says what to do. Two cases worth knowing:

**"the environment is incomplete"** — an install was interrupted or lost its
connection. Re-running the setup fixes it by itself; it checks whether the
environment actually works rather than whether the folder is there. To force it:

```bash
python3 scripts/setup_qei_net.py --rebuild
```

**`QEI-Net: off`** when you expected ON — the setup has not been run, or it did
not finish. `run_ui.py` tells you which, and what to run.

---

## Without the model

Everything else works unchanged. `1.1.qei_net` reports **N/A**, which is not a
failure and does not count against the report's coverage figure — it means "not
applicable here", the same as a check that does not apply to your organ. The
classical QEI (`1.qei`) is computed by osipy-qc itself and never needs this.

Run the tool the same way (`python3 scripts/run_ui.py`) or use `osipy-qc`
directly; the only difference is that one row reads N/A.

---

## Why it is a drop folder and not a dependency

osipy-qc depends on numpy and nibabel and nothing else, deliberately. QEI-Net
needs torch, torchio and SimpleITK — roughly 700 MB — so it lives in its own
environment and is reached by running it as a separate process. That is why this
is a folder you put a file into rather than a line in `pyproject.toml`.

Everything you put in this folder is ignored by git, and that is enforced rather
than merely intended: `test_the_weights_are_ignored_by_git` in
`tests/test_qei_net.py` shells out to `git check-ignore` and fails if the rules
ever stop covering `*.pth`, `weights/` or this folder's contents.
