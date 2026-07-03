# osipy-qc — Usage & Function Reference

Rough, working reference — not the final documentation, just an accurate map of
every function and its options right now. The basic shape of the API won't
change; this just makes it visible.

For a narrative "try it in 30 seconds" walkthrough, see [README.md](README.md).
This doc is the exhaustive one: every function, every parameter.

---

## 1. Command line

```bash
osipy-qc --demo                 # grade a built-in synthetic CBF map (full Stream B)
osipy-qc <folder>                # QC a folder of raw NIfTIs (Stream A)
osipy-qc <folder> --json         # same, machine-readable JSON
python -m osipy_qc <folder>      # identical, module form
```

| flag | type | meaning |
|---|---|---|
| `folder` | positional, optional | path to a folder of raw `.nii`/`.nii.gz` files. Recurses into subfolders. Not needed with `--demo`. |
| `--json` | flag | print the full JSON report instead of the human-readable table |
| `--demo` | flag | ignore `folder`; grade a synthetic "clean" CBF map instead |

---

## 2. Python API — the two entry points you'll actually use

### `run_qc(inputs, cfg=None, checks=None) -> QCReport`

The core function. Every other helper below just builds `inputs` for you.

```python
from osipy_qc import run_qc

report = run_qc(inputs)                       # run everything registered
report = run_qc(inputs, checks=["1.qei"])     # run only named checks
report = run_qc(inputs, cfg=my_config)        # override thresholds (see §4)
```

- **`inputs`** — a flat `dict`. You only put in what you have; every check reads
  the keys it needs and returns `UNKNOWN` for anything missing. See the table in
  §3 for exactly which keys each check reads.
- **`cfg`** — an optional `QCConfig` instance (§4). Defaults to `QCConfig()`.
- **`checks`** — an optional list of check names to run instead of all 17.
- **Returns** a `QCReport`: `.overall` (a `Verdict`), `.results` (list of
  `CheckResult`), `.to_dict()`, `.to_json()`.

### `grade_cbf(cbf, gm=None, wm=None, csf=None, cfg=None) -> QCReport`

The one-liner for "I have a CBF map (and maybe tissue maps), just grade it."

```python
from osipy_qc import grade_cbf

report = grade_cbf("sub01_cbf.nii.gz")                     # level/noise checks only
report = grade_cbf("sub01_cbf.nii.gz", gm="gm.nii.gz",
                    wm="wm.nii.gz")                          # + QEI, ratio, etc.
report = grade_cbf("sub01_cbf.nii.gz", gm="gm.nii.gz",
                    wm="wm.nii.gz", csf="csf.nii.gz")        # csf given explicitly
```

If `csf` is omitted but `gm`/`wm` are given, it's derived as
`clip(1 - gm - wm, 0, 1)`. All maps must be on the **same voxel grid** as the
CBF map — a shape mismatch raises `ValueError` immediately, rather than
grading silently-wrong data.

---

## 3. Every check — what it reads, stream, and what it's for

| Check | Stream | Reads (`inputs` keys) | What it's grading |
|---|---|---|---|
| `1.qei` | B | `cbf, gm, wm, csf`, optional `voxel_mm` | Quality Evaluation Index (Dolui 2024) |
| `2.1.spatial_cov` | B | `cbf, gm` | spatial CoV — ExploreASL 3-tier |
| `2.2.snr` | B | `cbf, gm`, optional `asl_4d, brain` | spatial SNR (+ tSNR if a 4D series is given) |
| `2.3.histogram` | B | `cbf, gm` | GM CBF skewness + negative fraction |
| `3.1.cbf_level` | B | `cbf, gm, wm` | mean/median GM & WM CBF in range |
| `3.2.gm_wm_ratio` | B | `cbf, gm, wm` | GM brighter than WM (scale-free) |
| `3.3.negative_gm` | B | `cbf, gm` | fraction of negative GM voxels |
| `4.1.coregistration` | B | `asl_mask, struct_mask` | Dice overlap, ASL vs T1 brain mask |
| `5.1.schema` | A | `sidecar` (BIDS JSON dict), `detected` | BIDS field validation, degrades gracefully |
| `5.2.volume_integrity` | A | `asl_4d` or `n_volumes`, `structure` | even control/label volume count |
| `5.3.swap` | A | `asl_4d`, `background_suppression`, `structure` | control brighter than label (N/A under BS) |
| `6.1.m0_present` | A | `m0_type` (`"separate"`/`"included"`/`None`) | is there an M0, what kind |
| `6.2.m0_tr` | A | `m0_tr_s` | M0 TR ≥ 5s, else WARN + correction factor |
| `6.3.m0_no_bs` | A | `m0_background_suppression` (bool) | M0 acquired WITHOUT background suppression |
| `6.5.m0_geometry` | A | `m0_shape, asl_shape` | M0 on the same grid as the ASL |
| `7.1.motion` | A | `motion_params` (6-col array) and/or `asl_4d`, optional `brain` | FWD (Power 2012) + DVARS |
| `8.2.data_type` | A | `files` (list of `{name, shape, voxel_mm}`), `context` | vendor / 2D-3D / structure inference — routing, always `INFO` |

Every check also silently accepts (and ignores) a `cfg: QCConfig` kwarg, and any
extra keys in `inputs` via `**_` — that's what lets one flat dict feed all 17
checks safely.

---

## 4. Configuration — every threshold lives in one place

```python
from osipy_qc import run_qc, QCConfig

cfg = QCConfig(
    qei_pass=0.60,        # raise the QEI PASS bar from the default 0.55
    gm_cbf_lo=35.0,       # widen the GM CBF PASS band
    scov_macro=0.70,      # loosen the macrovascular sCoV cutoff
)
report = run_qc(inputs, cfg=cfg)
```

All ~30 tunable values (QEI curve constants, sCoV/SNR bands, CBF level bands,
M0 TR minimum, motion thresholds, Dice cutoffs, ...) are fields on
`QCConfig` — see `osipy_qc/core/config.py` for the full list with their defaults
and the source each one comes from.

---

## 5. Pipeline adapters — for oxford_asl / ASLPrep output folders

```python
from osipy_qc.io import find_oxford_asl, find_aslprep, load_cbf_inputs
from osipy_qc import run_qc

paths = find_oxford_asl("my_subject/oxford_out")   # {'cbf':..., 'gm':..., 'wm':...}
report = run_qc(load_cbf_inputs(**paths))

paths = find_aslprep("derivatives/aslprep/sub-01/perf")
report = run_qc(load_cbf_inputs(**paths))
```

Both return a dict of file paths (any entry may come back `None` — filenames
vary by tool version) — pass explicit paths to `load_cbf_inputs` yourself if a
lookup misses. ASLPrep's tissue maps sometimes live in `anat/` space and need
resampling to the ASL grid first.

`load_cbf_inputs(cbf, gm=None, wm=None, csf=None) -> dict` is the lower-level
function these wrap — use it directly if you want the `inputs` dict itself
(e.g. to add extra keys before calling `run_qc`).

`load_folder(folder, load_arrays=True) -> dict` is the Stream-A equivalent —
what the CLI calls internally when you run `osipy-qc <folder>`.

---

## 6. Synthetic test data (no real scan needed)

```python
from osipy_qc.synth import synthetic_case
from osipy_qc import run_qc

c = synthetic_case(quality="clean")        # or "borderline" / "garbage"
report = run_qc({"cbf": c.cbf, "gm": c.gm, "wm": c.wm, "csf": c.csf,
                  "brain": c.brain, "voxel_mm": c.voxel_mm})
```

`quality` controls how strongly the CBF map correlates with the tissue
template — `"clean"` should PASS, `"garbage"` should FAIL, `"borderline"` sits
in between. Useful for testing without needing a real dataset.

---

## 7. Reading the result

```python
report.overall          # Verdict.PASS / WARN / FAIL / UNKNOWN
report.results          # list[CheckResult] — one per check that ran
report.to_dict()        # plain dict
report.to_json()        # JSON string
```

Each `CheckResult` has `.check` (name), `.verdict`, `.metric` (dict of the
actual numbers), `.reason` (one-line human explanation).

---

## 8. Quick demo — the mid-term walk-through, exactly as run

```bash
cd osipy-qc
source .venv/bin/activate
```

**(a) proof it works — synthetic data, 10 seconds**
```bash
osipy-qc --demo
```
→ all Stream B checks PASS, `QEI 0.9694`

**(b) raw-data QC — Stream A, on the 3 real datasets**
```bash
osipy-qc data/GE_PCASL_Product_Sequence/
osipy-qc data/Siemens2DPCASL_No_M0/
osipy-qc data/Siemens_BS3DPCASL/
```
→ auto-detects vendor / 2D-3D / M0 / background suppression from raw files, with
zero metadata (no BIDS sidecar, no `_aslcontext.tsv`).

**(c) CBF-map QC — Stream B, real QEI on all 3 (with real tissue maps)**
```bash
python examples/grade_cbf_map.py output/oxford_ge/perfusion_calib.nii.gz \
    --gm output/oxford_ge/pvgm_inasl.nii.gz --wm output/oxford_ge/pvwm_inasl.nii.gz

python examples/grade_cbf_map.py output/oxford_s2d/perfusion.nii.gz \
    --gm output/oxford_s2d/pvgm_inasl.nii.gz --wm output/oxford_s2d/pvwm_inasl.nii.gz

python examples/grade_cbf_map.py output/oxford_bs3d/perfusion_calib.nii.gz \
    --gm output/oxford_bs3d/pvgm_inasl.nii.gz --wm output/oxford_bs3d/pvwm_inasl.nii.gz
```
→ real QEI: **GE 0.1335 FAIL · Siemens 2D 0.4638 FAIL · BS-3D 0.2937 FAIL** — all
below the 0.5 cutoff. Provisional acquisition parameters were used (labeling
duration, PLD, etc. aren't in the NIfTI headers); the real GE/BS-3D parameters
are one of the open asks for the mentors.

> These commands use `data/` and `output/oxford_*` — the raw datasets the
> mentors provided and the tissue maps generated from them via `oxford_asl
> --pvcorr`. Both folders are git-ignored (the data isn't redistributable), so
> a fresh clone won't have them — this section documents the exact real run,
> not a reproducible fresh-clone command.
