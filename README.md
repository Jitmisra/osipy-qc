# osipy-qc — Quality Control ToolBox for ASL MRI

A Python library that grades an ASL-derived **CBF map** (and the **raw data**) and
returns a **PASS / WARN / FAIL** verdict per check, with reasons — to triage bad
scans automatically in large multi-center studies.

**Three organs.** Brain is the v1.0 target and the only one with a published
quality index behind it. **Kidney** and **placenta** ship too, because ASL is used
in both and neither has a QC tool at all — but they are built to a stricter honesty
rule than brain: where the consensus literature states no number, the check reports
and refuses to grade rather than inventing a cut-off. See §6.

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

## 6. What it checks — 54 checks, three organs, two streams

Two streams run against every organ: **Stream B** asks *is the perfusion map good?*,
**Stream A** asks *was the raw scan acquired correctly?* Pick the organ with
`--organ kidney` on the CLI, or in Python:

```python
from osipy_qc import run_qc
from osipy_qc.core.config import QCConfig
from osipy_qc.io import load_organ_folder

report = run_qc(load_organ_folder("data/my_kidney_scan", "kidney"),
                cfg=QCConfig(organ="kidney"))
```

Only that organ's checks run. `load_organ_folder` is the loader that routes masks —
the brain loader has no concept of them and would silently drop every one.

| organ | checks | what backs the numbers |
|---|---:|---|
| [brain](#brain--20-checks) | 20 | QEI (Dolui 2024), ASL White Paper, ASLPrep, ExploreASL |
| [kidney](#kidney--19-checks) | 19 | Nery 2020 renal consensus — 59 statements, **zero numeric thresholds** |
| [placenta](#placenta--15-checks) | 15 | Taso 2023 — neither recommendations nor summarised practice |

### brain — 20 checks

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
| `3.5.brain_cbf` | whole-brain CBF over a self-derived mask — the only magnitude check that runs with **no tissue maps** |
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

### kidney — 19 checks

The renal consensus (Nery 2020, MAGMA/PARENCHIMA) is 59 statements and **not one
numeric quality threshold**. So most of these checks report a number and mark it
INFO; the ones that grade, grade on structure (is the mask sane, is the geometry
one grid, was the M0 clean) rather than on a perfusion value we cannot cite.

**Stream B — is the RBF map good?**
| Check | What |
|---|---|
| `k1.1.renal_qei` | **always N/A** — no renal QEI exists, and inventing one is the single worst thing this tool could do |
| `k2.1.tsnr` | temporal SNR per kidney — **INFO only** |
| `k2.2.pws_pct` | perfusion-weighted signal as a % of M0 |
| `k2.3.implausible_values` | fraction of within-kidney voxels that are physically impossible |
| `k3.1.cortical_rbf` | cortical RBF **per kidney separately** (Nery R10.1, 100% agreement), against a sanity bound only |
| `k3.2.cmr` | cortico-medullary ratio — a **segmentation-integrity** flag, not a perfusion verdict (R10.2: medullary values are unreliable) |
| `k3.3.left_right` | left-vs-right consistency — a review flag, never a rejection |
| `k4.1.mask_integrity` | is each supplied mask one sane, unclipped object? |
| `k4.2.registration` | one geometry, registration scope, centroid residual |
| `k4.3.slice_coverage` | share of kidney-bearing slices that carry usable data |

**Stream A — was the raw scan acquired correctly?**
| Check | What |
|---|---|
| `k5.1.metadata` | how many of the nine consensus-reportable acquisition facts are present |
| `k5.2.data_type` | routing — what kind of renal dataset is this (INFO) |
| `k5.3.swap` | control/label ordering, judged **per pair** not on the pooled mean |
| `k6.1.m0_present` · `k6.2.m0_clean` · `k6.3.m0_tr` | M0 exists, carries no labelling or BS, and relaxed fully |
| `k7.1.kidney_displacement` | per-kidney respiratory displacement, per anatomical axis |
| `k7.2.outlier_rate` | subtraction-outlier rejection — the one genuinely implementable published renal number |
| `k7.3.breathing_strategy` | which breathing strategy, and how efficient the gating was |

### placenta — 15 checks

Even thinner ground: Taso 2023 offers neither recommendations nor summarised
practice. The design consequence is that **P2.1 is a gate** — until the map's units
are declared, nothing numeric downstream is allowed to grade.

**Stream B — is the perfusion map good?**
| Check | What |
|---|---|
| `p1.1.placental_qei` | **always N/A** — no placental quality index exists |
| `p2.1.units_declaration` | **the gate**: what do this map's numbers mean? Everything numeric waits on it |
| `p2.2.implausible_values` | negative / non-finite / upper-outlier fractions inside the placenta |
| `p2.3.segment_cov` | within-placenta heterogeneity — **INFO only** |
| `p3.1.mask_integrity` | one sane object, right grid, and where the mask came from |
| `p3.2.slab_coverage` | does the imaging slab actually contain the whole placenta? |

**Stream A — was the raw scan acquired correctly?**
| Check | What |
|---|---|
| `p4.1.labelling_scheme` | which scheme — and therefore **which circulation** was measured (FAIR labels both) |
| `p4.2.ga_context` | gestational age and maternal/scanner context |
| `p5.1.m0_state` | present, unlabelled, and not background-suppressed (a BS'd M0 never PASSes) |
| `p5.2.m0_heterogeneity` | how structured the M0 is inside the placenta |
| `p5.3.quant_constants` | are the quantification constants recorded, T1-blood consistent |
| `p6.1.pair_outliers` | per-pair subtraction outlier rejection |
| `p6.2.temporal_sd` | temporal stability after motion correction |
| `p6.3.registration_residual` | was a deformable registration used, how much residual deformation |
| `p6.4.contraction_events` | candidate uterine contraction events — **INFO only** |

### Verdicts
`PASS` · `WARN` · `FAIL` · `UNKNOWN` (a check that *should* run but had no input to look
at) · `N/A` (structurally inapplicable) · `INFO` (reported, not graded).

Overall = **any FAIL → FAIL**, else **any WARN → WARN**, else **PASS**, else **UNKNOWN**.
UNKNOWN, N/A and INFO are excluded from that: they report an absence, not a finding.

So the overall verdict says *"of what could be measured, this is the worst of it"* — and
nothing about **how much** could be measured. That is what `coverage()` carries, and every
report prints it beside the verdict:

```python
from osipy_qc.core.result import coverage
coverage(report.results)   # {'graded': 14, 'total': 18, 'unknown': 4, 'complete': False, ...}
```

A PASS over 4 graded checks and a PASS over 18 are different claims. Showing one without
the other misrepresents it, so the CLI, the JSON and the HTML report all carry both.

### Every threshold says where it came from
Most of this field has **no published PASS/FAIL cutoffs** — ASLPrep, ExploreASL,
MRIQC and fMRIPrep all report metrics and ship *zero* verdict logic. So rather than
dress up guesses as evidence, each threshold is tagged **published** (12, each with
a DOI), **implementation** (34), or **uncalibrated** (44, honestly declared).

A FAIL decided by an uncalibrated cut-off is marked **provisional** in the report.
`--no-strict` (or `QCConfig(strict=False)`) demotes every provisional FAIL to a WARN,
leaving only failures backed by a published threshold. On a deliberately broken scan
that is the difference between four failures and two:

| check | threshold | strict (default) | `--no-strict` |
|---|---|---|---|
| `1.qei` | published (Dolui 0.5) | FAIL | **FAIL** |
| `3.2.gm_wm_ratio` | published (ASLPrep) | FAIL | **FAIL** |
| `3.1.cbf_level` | uncalibrated | FAIL | WARN |
| `3.3.negative_gm` | uncalibrated | FAIL | WARN |

Strict grading is the default, so an uncalibrated number *can* reach a FAIL — what it
cannot do is reach one **silently**. Earlier versions of this file claimed uncalibrated
thresholds never fail at all; that was never true of the code.

```bash
osipy-qc --provenance      # every number, its source, and what that source says
```
Full write-up: **[THRESHOLD_PROVENANCE.md](THRESHOLD_PROVENANCE.md)**.

### Adult and newborn
Newborn CBF is far lower than adult (neonate GM ~16 vs adult ~58 mL/100g/min), so
grading a neonate against adult bands would fail every scan. v1.0 ships two
calibrated profiles — **adult** (the brain target) and **neonate** (the mentor's
neonatal domain). Pick one with `--population neonate` on the CLI, or
`for_population("neonate")` in Python. Adult = White Paper (Alsop 2015) + Wu 2013;
neonate = Miranda 2006. Other age groups are planned once their bands are
calibrated with the mentors (see the population table in [THRESHOLD_PROVENANCE.md](THRESHOLD_PROVENANCE.md)).

---

## 7. Tests

```bash
python -m pytest -q        # 465 known-answer tests
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
                   roi (mask-first ROI toolkit: components, SSIM, asymmetry),
                   imaging (stdlib-only PNG/SVG encoders)
    checks/        brain: qei, noise, cbf_level, coreg, schema, m0, motion
                   other organs: kidney, placenta
    synth.py       synthetic data with known quality
    io.py          NIfTI loaders + grade_cbf / find_oxford_asl / find_aslprep
    report.py      the runner + JSON report
    report_html.py the visual report (images, histograms, provenance)
    web.py         the local web UI (osipy-qc --serve)
    cli.py         command line
  tests/           known-answer tests per module (27 files, 465 tests)
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
