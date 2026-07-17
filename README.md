# osipy-qc — Quality Control ToolBox for ASL MRI

A Python library that grades an ASL-derived **CBF map** (and the **raw data**) and
returns a **PASS / WARN / FAIL** verdict per check, with reasons — to triage bad
scans automatically in large multi-center studies.

It is a **QC layer, not a processing pipeline**: it *grades* data, it does not
reprocess it. Heavy steps (CBF quantification, T1 segmentation, co-registration)
belong to PyASL / ASLPrep / oxford_asl; this toolbox sits downstream and judges
the result.

**Pure NumPy + nibabel. No scipy** (osipy GPU-portability rule).

---

## 1. Setup (one flow)

Copy-paste this, from a terminal, into the folder:

```bash
cd osipy-qc
python3 -m venv .venv                # macOS/Linux: use python3 (macOS has no `python`)
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -e ".[test]"             # install the package + pytest
```

That gives you an `osipy-qc` command **and** the `osipy_qc` Python module. After the
venv is activated, plain `python` works too.

Sanity check:

```bash
python -m pytest -q                  # -> 72 passed
```

---

## 2. Try it in 30 seconds (no data needed)

The repo ships a small **synthetic** example under [`example_data/`](example_data/)
(safe, reproducible — **no real patient data**), so you can grade a CBF map right
away. Run any of these from the repo root:

```bash
# (a) the web UI — upload a CBF map in your browser, get a visual report
osipy-qc --serve

# (b) grade the built-in synthetic clean map
osipy-qc --demo

# (c) same, plus a self-contained visual report (images + histograms)
osipy-qc --demo --html report.html

# (d) where did every threshold come from?
osipy-qc --provenance

# (e) run the end-to-end example script (synthetic clean/borderline/garbage + the bundled map)
python examples/run_examples.py

# (f) grade the bundled example CBF map from its files (full Stream B, incl. QEI)
python examples/grade_cbf_map.py example_data/example_cbf.nii.gz \
    --gm example_data/example_gm.nii.gz \
    --wm example_data/example_wm.nii.gz
```

**What you should see** — every CBF-map (Stream B) check PASSes, e.g. `QEI 0.96 PASS`:

```
1.qei              PASS   QEI 0.9634 (>= 0.55)
2.1.spatial_cov    PASS   sCoV 9.6%
2.2.snr            PASS   spatial SNR 10.38
2.3.histogram      PASS   skew -0.98, 0.0% negative
3.1.cbf_level      PASS   GM 56.0 (PASS), WM 26.3 (PASS)
3.2.gm_wm_ratio    PASS   GM/WM ratio 2.13 (healthy)
3.3.negative_gm    PASS   0.0% negative GM voxels
```

> ℹ️ **Why `osipy-qc --demo` prints `OVERALL: WARN`, not PASS.** The demo grades a
> **CBF map only**, so the raw-data checks (Stream A: motion, M0, control/label…)
> have nothing to run on and report `UNKNOWN`, which nudges the overall to WARN.
> That's expected — look at the Stream B rows, they all PASS. Feed raw NIfTIs
> (see §3) and the Stream A checks light up too.

---

## 3. Run on the real ASL datasets (raw + CBF)

This is the full story: **raw scan → a pipeline makes a CBF map → the toolbox grades it.**

```
raw data (data/…)  ──ASLPrep / oxford_asl──►  CBF map (output/…)  ──osipy-qc──►  PASS / WARN / FAIL
```

> **The commands below are an illustrative example run.** Real ASL datasets are not
> redistributable, so `data/` and `output/` are empty in a fresh clone — the paths
> and output here show what a real run looks like. To reproduce, drop your own raw
> scans into `data/` (§4/§5) or your pipeline's CBF maps into `output/`.

### (a) Raw-data QC — Stream A

Point the tool at a folder of raw NIfTIs. **No BIDS metadata needed** — it infers
vendor / 2D-3D / M0 / background-suppression from the NIfTI shapes + filenames.

```bash
osipy-qc data/GE_PCASL_Product_Sequence/       # -> "GE 3D pre-subtracted deltaM", M0 present
osipy-qc data/Siemens2DPCASL_No_M0/            # -> "Siemens 2D, 80 vols -> 40 pairs", WARN: no M0
osipy-qc data/Siemens_BS3DPCASL/               # -> "Siemens 3D, 16 vols -> 8 pairs", BS on
```

Example output for `osipy-qc data/Siemens_BS3DPCASL/`:

```
=== OVERALL: 🟠 WARN ===

  ✅ 5.2.volume_integrity   PASS     16 volumes -> 8 pairs
  ⊘ 5.3.swap               N/A      background suppression on - intensity logic does not apply
  ✅ 6.1.m0_present         PASS     M0 present (separate)
  ✅ 6.5.m0_geometry        PASS     M0 (88, 88, 52) matches ASL grid
  ℹ️ 8.2.data_type          INFO     Siemens 3D control/label series (16 volumes)
```

### (b) CBF-map QC — Stream B (CBF maps produced with ASLPrep)

The pipeline wrote a CBF map per dataset into `output/aslprep/`. Grade each one:

```bash
python examples/grade_cbf_map.py output/aslprep/sub01_cbf.nii.gz   # GE 3D        -> FAIL
python examples/grade_cbf_map.py output/aslprep/sub02_cbf.nii.gz   # Siemens 2D   -> FAIL
python examples/grade_cbf_map.py output/aslprep/sub03_cbf.nii.gz   # Siemens BS3D -> FAIL
```

Example output for `output/aslprep/sub03_cbf.nii.gz`:

```
CBF map : output/aslprep/sub03_cbf.nii.gz
mode    : ROUGH whole-brain mask (no tissue maps) -> level + noise only
OVERALL : FAIL

  2.1.spatial_cov    PASS   sCoV 45.2% (CBF-contrast)
  2.3.histogram      WARN   left-skew -1.66 + 8.9% negative (noise/labeling hint)
  3.1.cbf_level      FAIL   GM 312.0 (FAIL), WM 312.0 (FAIL)      <- ~7x too high (calibration)
  3.3.negative_gm    PASS   8.9% negative GM voxels
```

**The headline:** ASLPrep's own QEI called this map's *shape* "good" (0.478), but the
CBF is ~7× physiological — osipy-qc's `3.1.cbf_level` catches it. QEI measures
shape, not absolute calibration, so a QC toolbox needs both.

> These real-map grades use a **rough whole-brain mask** (the pipeline's GM/WM maps
> weren't retained), so GM/WM numbers are whole-brain, not GM-specific — but the
> FAIL verdicts are correct. To get the **full** Stream B (incl. QEI) on a real map,
> pass its tissue maps too: `--gm <gm.nii.gz> --wm <wm.nii.gz>` (see §4).

---

## 4. Grade your own CBF map

There's a ready-made [`data/`](data/) folder for your files (its contents are
**git-ignored**, so you never accidentally commit a scan). Drop three files in:

```
data/
  cbf.nii.gz    the quantified CBF map (mL/100g/min), in ASL voxel space
  gm.nii.gz     grey-matter probability / partial-volume map, SAME voxel space
  wm.nii.gz     white-matter probability / partial-volume map, SAME voxel space
  csf.nii.gz    optional (derived as 1 - gm - wm if omitted)
```

Then from the repo root:

```bash
python examples/grade_cbf_map.py data/cbf.nii.gz --gm data/gm.nii.gz --wm data/wm.nii.gz
```

Or from Python:

```python
from osipy_qc.io import grade_cbf

report = grade_cbf("data/cbf.nii.gz", gm="data/gm.nii.gz", wm="data/wm.nii.gz")
print(report.overall)      # Verdict.PASS / WARN / FAIL
print(report.to_json())    # full per-check report
```

> ⚠️ The GM/WM maps **must be on the same voxel grid as the CBF map** (ASL space),
> or the loader raises a shape-mismatch error. With **only** a CBF map (no tissue
> maps) the script still runs using a rough whole-brain mask (level + noise checks
> only); QEI and the GM/WM checks need real tissue separation.

---

## 5. If you only have *raw* ASL (no CBF map yet)

First make a CBF map with a processing pipeline, **then** grade it. The toolbox is
**pipeline-agnostic** — it doesn't care who produced the map.

**FSL `oxford_asl`** writes the CBF map *and* GM/WM partial-volume maps into
`native_space/` when you pass a structural:

```bash
oxford_asl -i asl.nii.gz -o output/ge_out -s T1.nii.gz --pvcorr ...
```
```python
from osipy_qc.io import find_oxford_asl, load_cbf_inputs
from osipy_qc import run_qc

paths  = find_oxford_asl("output/ge_out")     # {'cbf':.., 'gm':.., 'wm':..}
report = run_qc(load_cbf_inputs(**paths))
```

**ASLPrep** writes `derivatives/aslprep/sub-XX/perf/` with the CBF map plus tissue
probability maps:

```python
from osipy_qc.io import find_aslprep, load_cbf_inputs
from osipy_qc import run_qc

paths  = find_aslprep("derivatives/aslprep/sub-01/perf")
report = run_qc(load_cbf_inputs(**paths))
```
*(ASLPrep tissue maps may be in anat/standard space — resample to the ASL grid
first, or the loader tells you the shapes don't match.)*

**Raw-data QC (Stream A) directly on a folder of NIfTIs** — no BIDS metadata
needed; data-type detection infers vendor / 2D-3D / M0 / BS from shapes + filenames:

```bash
osipy-qc data/my_raw_scan/          # human-readable
osipy-qc data/my_raw_scan/ --json   # machine-readable
```

> ℹ️ ASLPrep also computes its **own QEI** — a handy reference to cross-check
> osipy-qc's `1.qei` against; the two agree where they overlap.

---

## 6. What it checks (17 checks, 8 modules, two streams)

**Stream B — is the CBF map good?**
| Check | What |
|---|---|
| `1.qei` | Quality Evaluation Index (Dolui 2024), ASLPrep-faithful |
| `2.1.spatial_cov` | spatial CoV, ExploreASL 3-tier (vascular >0.67, artifactual >1.0) |
| `2.2.snr` | spatial SNR (= 1/sCoV) + tSNR |
| `2.3.histogram` | GM CBF shape — **INFO only** (no published skewness cutoff exists) |
| `3.1.cbf_level` | mean/median GM & WM CBF in range — **population-dependent** |
| `3.2.gm_wm_ratio` | GM brighter than WM (scale-free) |
| `3.3.negative_gm` | fraction of negative GM voxels |
| `3.4.deep_gm_ratio` | **neonatal** — deep GM should exceed cortical GM (Miranda 2006) |
| `4.1.coregistration` | Dice / Jaccard of ASL vs T1 mask |
| `4.2.coverage` | how much of the tissue ROI the ASL actually imaged (FOV mismatch) |

**Stream A — was the raw scan acquired correctly?**
| Check | What |
|---|---|
| `5.1.schema` | BIDS sidecar, degrades gracefully when absent |
| `5.2.volume_integrity` | even number of control/label volumes |
| `5.3.swap` | control brighter than label (N/A under BS) |
| `6.1.m0_present` | M0 present / type (absent → WARN, derive) |
| `6.2.m0_tr` | M0 TR ≥ 5 s (else WARN + correction factor) |
| `6.3.m0_no_bs` | M0 acquired WITHOUT background suppression |
| `6.5.m0_geometry` | M0 on the same grid as the ASL |
| `7.1.motion` | framewise displacement (Power 2012) + DVARS |
| `8.2.data_type` | vendor / 2D-3D / structure inference (routing) |

### Verdicts
`PASS` · `WARN` · `FAIL` · `UNKNOWN` (a check that *should* run but couldn't — escalates the
overall to WARN) · `N/A` (structurally inapplicable — excluded) · `INFO` (reported, not
graded — excluded).

Overall = **any FAIL → FAIL**, else **any WARN/UNKNOWN → WARN**, else **PASS**.

### Every threshold says where it came from
Most of this field has **no published PASS/FAIL cutoffs** — ASLPrep, ExploreASL,
MRIQC and fMRIPrep all report metrics and ship *zero* verdict logic. So rather than
dress up guesses as evidence, each threshold is tagged **published** (11, each with
a DOI), **implementation** (9), or **uncalibrated** (16, honestly declared).
**Uncalibrated thresholds never drive a FAIL on their own**, and
`QCConfig(strict=False)` softens the rest for clinical cohorts.

```bash
osipy-qc --provenance      # every number, its source, and what that source says
```
Full write-up: **[THRESHOLD_PROVENANCE.md](THRESHOLD_PROVENANCE.md)**.

### Across the lifespan
CBF norms move enormously with age (child GM ~97, adult ~58, neonate ~16
mL/100g/min), so grading a neonate against adult bands would fail every scan.
Pick the population — `--population neonate_term` on the CLI, or
`for_population(...)` in Python. Sourced to Miranda 2006, Biagi 2007, Leoni 2017.

---

## 7. Tests

```bash
python -m pytest -q        # 72 known-answer tests
```

Every check has **known-answer** tests (hand-computed expected values); the QEI is
additionally verified against an independent re-derivation of the ASLPrep formula.

## 8. Repo layout

```
osipy-qc/
  osipy_qc/
    core/          registry, result/verdict, config (thresholds + provenance +
                   population/organ profiles)
    utils/         mathops, smoothing (pure-NumPy Gaussian), masks (coverage-aware),
                   imaging (stdlib-only PNG/SVG encoders)
    checks/        one module per QC module (qei, noise, cbf_level, coreg, schema, m0, motion)
    synth.py       synthetic data with known quality
    io.py          NIfTI loaders + grade_cbf / find_oxford_asl / find_aslprep
    report.py      the runner + JSON report
    report_html.py the visual report (images, histograms, provenance)
    web.py         the local web UI (osipy-qc --serve)
    cli.py         command line
  tests/           known-answer tests per module (136)
  examples/        run_examples.py, grade_cbf_map.py
  example_data/    small synthetic CBF + tissue maps (safe demo data — committed)
  data/            <- put YOUR scans here (git-ignored, never committed)
  output/          <- pipeline outputs go here (git-ignored)
  USAGE.md                 every function and CLI option
  THRESHOLD_PROVENANCE.md  where every PASS/FAIL number came from
```

**No plotting dependency.** The visual report encodes its own images with the
standard library (PNG via `zlib`+`struct`, plots as hand-written SVG), so the
runtime stays **numpy + nibabel** — no scipy, no nilearn, no matplotlib, and no
web framework behind `--serve`.

## 9. Note on data & AI use

- **No real/patient data is included.** [`example_data/`](example_data/) is
  synthetic; the `data/` and `output/` folders are empty and git-ignored, so real
  ASL datasets (which are not redistributable) never enter the repo.
- Built for **GSoC 2026 with OSIPI**. AI tools were used to learn, draft, and
  check; every line is reviewed and owned by the author, and AI use is disclosed
  per the OSIPI AI-use policy.
