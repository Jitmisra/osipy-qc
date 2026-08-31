# Real-data review — commands, outputs, and one finding that matters

Requested in review: *"can you send me a PDF which has — if I run this module or
this command, these are the outputs I'm getting?"* and *"how are you determining
what is pass and what is fail?"*

This is that document. Every number below was produced by running the command
shown, on the three datasets you sent. Nothing here is illustrative.

- **How PASS/FAIL is decided, with references:** [THRESHOLD_PROVENANCE.md](THRESHOLD_PROVENANCE.md)
- **Every function and flag:** [USAGE.md](USAGE.md)

---

## 🔴 Read this first: two of the three CBF maps are nearly empty

The new coverage check (`4.2.coverage`) surfaced something that invalidates my
earlier real-data numbers, including the ones I sent by email. **The oxford_asl
runs for the two Siemens datasets produced CBF maps that are almost entirely
zero**:

| dataset | CBF non-zero voxels | share of the volume | GM ROI actually covered |
|---|---|---|---|
| GE 3D pCASL | 90,754 | **13.8%** — a plausible brain | 70.1% |
| Siemens 2D | 5,098 | **4.0%** | 16.1% |
| Siemens BS 3D | 3,906 | **1.0%** | **3.8%** |

A brain should occupy roughly 15% of the volume. So for BS 3D, **97% of the grey
matter ROI has no CBF data at all**, and any statistic computed over it is being
taken from ~1,000 voxels.

**What this means, honestly:**
- The QEI values I reported previously for Siemens 2D and BS 3D were computed
  over nearly-empty maps. **They are not meaningful and should be disregarded.**
- Those two `oxford_asl` runs need to be redone before any QC verdict on them is
  worth anything. Most likely the BASIL fit or the brain mask failed — plausibly
  the same root cause as the acquisition parameters I had to guess.
- **Only the GE map is populated enough to grade**, and even there the tissue ROI
  extends 30% beyond the ASL field of view.

This is the toolbox working: it refused to report a confident number over data
that isn't there. Previously it would have quietly averaged the zeros and given a
plausible-looking answer.

---

## 1. The bug you predicted — measured

Your concern, verbatim in substance: *"the cerebellum is probably not covered in
the ASL scan. And that's still in the ROI. So there is zero values in the CBF, but
it's actually in the ROI. So the gray matter CBF artificially looks low."*

**You were right, and it was in the data.** On the GE scan:

| | |
|---|---|
| GM ROI voxels (from the T1) | 24,123 |
| …actually imaged by the ASL | 16,920 |
| **…structural zeros inside the ROI** | **7,203 (29.9%)** |
| naive mean GM CBF (zeros averaged in) | **2435.9** |
| true mean GM CBF (coverage-masked) | **3472.8** |

And the mechanism is exactly the one you described — `naive == coverage × true`
holds **to the decimal** (2435.9 = 0.701 × 3472.8), which is not a coincidence but
the arithmetic of averaging in zeros.

Worth noting: **ASLPrep does not guard against this either** — its
`average_cbf_by_tissue()` (`aslprep/utils/confounds.py`) averages the ROI
directly. ExploreASL does guard, via a coverage-aware mask.

**Fixed** in `utils/masks.covered_tissue_mask()`: every tissue mask is intersected
with `cbf != 0`. And so the exclusion is never silent, `4.2.coverage` reports the
covered fraction as a finding in its own right.

---

## 2. The commands, and exactly what they print

### Setup
```bash
cd osipy-qc
python3 -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[test]"
python -m pytest -q                                   # 136 passed
```

### The web UI — no CLI needed
Answering *"how are they uploading the data right now? Just writing the Python
command with flags?"*:
```bash
osipy-qc --serve          # -> http://127.0.0.1:8000
```
Upload a CBF map (+ tissue maps), pick the population, get the visual report.
Local only; nothing is stored or sent anywhere.

### Where every threshold came from
```bash
osipy-qc --provenance
```
```
=== PUBLISHED - a paper states this number for this purpose (11) ===
  gm_cbf_lo              = 40.0
    source: Alsop DC, et al. MRM 2015;73(1):102-116, p.17. doi:10.1002/mrm.25197
    says  : Verbatim: "As a general rule, gray matter CBF values from 40-100 ml/min/100ml can be normal."
...
=== UNCALIBRATED - our engineering default, NOT calibrated. These never FAIL alone. (16) ===
  ratio_pass             = 1.5
    source: NONE
    says  : No source states 1.5. ...
```

### Raw-data QC (Stream A) — no metadata needed
```bash
osipy-qc data/Siemens_BS3DPCASL/
```
```
=== OVERALL: WARN ===
  ✅ 5.2.volume_integrity   PASS   16 volumes -> 8 pairs
  ⊘ 5.3.swap               N/A    background suppression on - intensity logic does not apply
  ✅ 6.1.m0_present         PASS   M0 present (separate)
  ✅ 6.5.m0_geometry        PASS   M0 (88, 88, 52) matches ASL grid
  ℹ️ 8.2.data_type          INFO   Siemens 3D control/label series (16 volumes)
```

### CBF-map QC (Stream B) — with the real tissue maps
```bash
python examples/grade_cbf_map.py output/oxford_ge/perfusion_calib.nii.gz \
    --gm output/oxford_ge/pvgm_inasl.nii.gz \
    --wm output/oxford_ge/pvwm_inasl.nii.gz
```
```
OVERALL : FAIL
  1.qei             FAIL   QEI 0.0901 (< 0.5)
  2.1.spatial_cov   PASS   sCoV 22.1% (CBF-contrast)
  2.2.snr           PASS   spatial SNR 4.53
  2.3.histogram     INFO   skew 0.06, 0.0% negative (informational)
  3.1.cbf_level     FAIL   GM 3472.8 (FAIL), WM 2384.7 (FAIL) [adult bands]
  3.2.gm_wm_ratio   WARN   GM/WM ratio 1.46 (weak contrast - inspect, may be benign)
  3.3.negative_gm   PASS   0.0% negative GM voxels
  4.2.coverage      FAIL   only 70.1% of the GM ROI is covered - large FOV mismatch
```
GE CBF is still ~50× too high in absolute terms — consistent with the GE-product
M0 scaling issue, and with what both pipelines showed. That is a **data /
calibration** problem, not a toolbox one.

### A visual report
```bash
osipy-qc --demo --html report.html
```
One self-contained file: CBF slice mosaic (negative voxels in blue), GM and WM
histograms with the population's expected band shaded, a GM-masked view so
coverage is judgeable by eye, and every check with the **provenance of the
thresholds behind it**. No matplotlib — the images are encoded with the standard
library, so the package stays `numpy + nibabel`.

### Grading across the lifespan
```bash
osipy-qc --demo --population neonate
```

---

## 3. What changed since the review

| Raised by | Change |
|---|---|
| Sudipto | **Provenance on every threshold** — 11 published (each with a DOI), 9 implementation, **16 declared uncalibrated**. Uncalibrated ones never FAIL alone. |
| Sudipto | **The skewness rule is retracted, not re-worded.** −0.5 was a statistics textbook heuristic (Bulmer 1979) with no ASL basis, and "ExploreASL-style histogram QC" **does not exist** — `grep -i skew` over ExploreASL returns nothing, and ASLPrep computes no skewness either. Now INFO. |
| Sudipto | **Uncovered-FOV bug fixed** + `4.2.coverage` (§1). |
| Sudipto | **Motion rewired to the right statistics**: mean FWD > 1 mm is Adebimpe 2022's exclusion rule; 0.5 mm is Power 2012's *per-frame* censoring line, now counted rather than applied to the mean. An undeclared `0.2` magic number removed. |
| Sudipto | **Quantification parameters are user-supplied** (`labeling_efficiency`, `label_duration_s`, `post_labeling_delay_s`, `t1_blood_s`), defaulting to `None` — never a silent guess. |
| Sudipto | **Upload UI** — `osipy-qc --serve`. |
| Maria | **Population profiles** across the lifespan, sourced to Miranda 2006 / Biagi 2007 / Leoni 2017. A neonate now passes neonatal bands and correctly fails adult ones. |
| Maria | **New `3.4.deep_gm_ratio`** — in newborns CBF is higher in deep GM than cortical GM (Miranda 2006: 30 vs 16, p<0.0001). N/A outside neonates. |
| Maria | **Visual report** with the histograms. |
| Maria | **Organ profiles**; the kidney stub explicitly skips QEI, whose spatial template is a brain tissue model. |
| Zhiliang | **sCoV tiers were inverted.** ExploreASL *keeps* 0.67–1.0 and excludes only >1.0; we were failing scans it keeps. Fixed. |
| Zhiliang | **Pathology ≠ artifact.** The GM/WM ratio informs rather than condemns (only ratio<1, the published ASLPrep rule, can FAIL), and `strict=False` demotes uncalibrated FAILs for clinical cohorts. |

136 tests pass (was 72).

---

## 4. Three things I need from you

1. **The acquisition parameters** — still the blocker for trustworthy absolute CBF:
   - **GE**: labeling duration, PLD, M0 TR, control/label order
   - **BS 3D**: PLD, labeling duration, background-suppression efficiency

2. **Two questions only you can settle** (you authored both sources):
   - `tissue_thresh`: your **paper says 0.9**, your **ASLPrep code says 0.7**.
   - `qei_c`: ASLPrep uses **0.054**; the paper's Fig. 2 prints **0.1**. Every other
     QEI constant rounds cleanly — this one is a ~1.85× gap.
   - (And: QEI cut-off **0.50**, or the paper's ROC-derived **0.53**?)

3. **A calibration path.** QEI ≈0.5 is the only threshold in this corpus derived by
   ROC against rater labels — that is the model to copy for the other 16. It needs
   **rated data**. Is AURA (or any expert-labelled set) available for that?

### One more for the lifespan work
The QEI's spatial template `spCBF = 2.5·GM + 1·WM` encodes an **adult** 2.5:1
GM:WM ratio. Measured cortical-GM:WM by age: **neonate 1.3–1.6, child 3.7,
adolescent 3.6, adult 2.9**. So the template is mis-specified at both ends of the
lifespan. Should it be re-weighted per age band before QEI is applied to
neonatal or paediatric data?
