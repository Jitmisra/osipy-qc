# How every PASS / WARN / FAIL is decided — and where each number came from

This document answers the question asked in review: *"How did you get this number?
Do you have a reference?"* — for every threshold in the toolbox, including the
ones where the honest answer is **"I didn't."**

Machine-readable source of truth: `osipy_qc/core/config.py` →
`THRESHOLD_PROVENANCE`. Every number below is tagged in code, and the report
prints the tag next to the value. Run `python -m osipy_qc --provenance` to dump it.

---

## The headline finding

**The field has almost no published PASS/FAIL thresholds.** ASLPrep, ExploreASL,
MRIQC and fMRIPrep all compute QC metrics and ship **zero verdict logic**:

- ExploreASL's paper says so outright: *"While these QC parameters can be helpful
  in detecting artifacts and/or protocol deviations, **their use has not yet been
  validated, and the normal and abnormal range for each of the parameters still
  need to be determined**."*
- The nipreps QC protocol paper (Provins et al. 2023, *Front Neuroimaging*
  1:1073734): *"Our exclusion criteria are all based on the visual inspection …
  so they are all qualitative."*
- MRIQC's own paper warns its thresholds are *"unlikely to generalize beyond the
  ADNI dataset."*

**The QEI cut-off (~0.5) is essentially the only threshold in this entire corpus
with a real published derivation.** That is not this project's embarrassment — it
is its justification. The gap this toolbox fills is exactly the verdict layer
nobody ships.

So rather than dress up guesses as evidence, every threshold is tagged:

| tag | meaning |
|---|---|
| **published** | a peer-reviewed paper states this number, for this purpose |
| **implementation** | a reference implementation uses it (code, not a paper) |
| **uncalibrated** | our engineering default. Not calibrated against rater labels |

**Rule: an `uncalibrated` threshold never drives a FAIL on its own**, and
`QCConfig(strict=False)` demotes all of them to WARN for clinical cohorts.

---

## 1. PUBLISHED — a paper states this number

| threshold | value | source | what it actually says |
|---|---|---|---|
| `qei_warn` | 0.50 | Dolui S, et al. *JMRI* 2024;60(6):2497-2508. doi:10.1002/jmri.29308 | *"a cut-off value of 0.5 has worked reliably for a wide variety ASL protocols in multiple studies."* The **ROC-derived value in the paper is 0.53** — see open questions. |
| `smooth_fwhm_mm` | 5.0 | Dolui 2024 | *"our recommendation is to smooth the CBF maps by a 5 mm isotropic Gaussian kernel … as that was used to derive the QEI parameters and the cut-off value."* |
| `gm_cbf_lo` / `gm_cbf_hi` | 40 / 100 | Alsop DC, et al. *MRM* 2015;73(1):102-116, **p.17**. doi:10.1002/mrm.25197 | Verbatim: *"As a general rule, gray matter CBF values from 40–100 ml/min/100ml can be normal."* |
| `wm_cbf_lo` / `wm_cbf_hi` | 15.8 / 27.5 | Wu W-C, et al. *PLoS One* 2013;8(12):e82679. doi:10.1371/journal.pone.0082679 | *"The measured white matter perfusion and perfusion ratio of gray matter to white matter were 15.8-27.5 ml/100ml/min and 1.8-4.0."* Corroborated by **Clement P, … Dolui S, et al.** *Front Radiol* 2022;2:929533: *"about 20 mL/100g/min in white matter."* |
| `ratio_min` | 1.0 | Adebimpe A, et al. *Nat Methods* 2022;19(6):683-686. doi:10.1038/s41592-022-01458-7 | Verbatim, twice: *"this ratio is expected to be greater than 1"* and *"we excluded participants with … a CBF GM to WM ratio of less than 1."* |
| `m0_tr_min_s` | 5.0 | Alsop 2015, **p.15** | *"If TR is less than 5s, the PD image should be multiplied by the factor (1/(1 − e^−TR/T1,tissue))."* |
| `fd_mean_fail_mm` | 1.0 | Adebimpe 2022 | *"we excluded participants with **mean** frame-wise displacement greater than 1 mm."* |
| `fd_frame_censor_mm` | 0.5 | Power JD, et al. *NeuroImage* 2012;59(3):2142-2154. doi:10.1016/j.neuroimage.2011.10.018 | *"values of 0.5 for framewise displacement … were chosen to represent values well above the norm."* A **per-frame censoring** line — Power chose it by studying plots. |
| `head_radius_mm` | 50 | Power 2012 | *"displacement on the surface of a sphere of radius 50 mm."* A **definition**, not a threshold. |

### A correction made during this audit
The two motion numbers were **wired to the wrong statistics**: 1.0 mm was applied
to *max* FWD and 0.5 mm to *mean* FWD. Both are now applied to the statistic
their source defines — mean FWD > 1 mm is the citable FAIL; 0.5 mm counts
censorable frames. An undeclared magic number (`0.2`) was removed.

---

## 2. IMPLEMENTATION — code uses it, no paper states it

| threshold | value | source | caveat |
|---|---|---|---|
| `qei_a/b/d/e/f` | 3.0126, 2.4419, 0.9272, 2.8478, 0.5196 | `aslprep/utils/confounds.py` `compute_qei` | Verified byte-exact. ASLPrep's own comment: *"The constants used here differ slightly from those in the paper, but match the actual values used in the original QEI implementation."* All round cleanly to the paper's Fig. 2 values. |
| `qei_c` | 0.054 | same | ⚠️ **Does not round cleanly.** Paper Fig. 2 prints `exp(-0.1·x^0.9)`; 0.054 → 0.05, **not 0.1** — a ~1.85× gap. See open questions. |
| `tissue_thresh` | 0.7 | ASLPrep code (*"binarized after thresholding at 70% probability"*) | ⚠️ **Conflicts with the paper**: Dolui 2024 p.4 says *"thresholding the tissue probabilistic maps to 0.9."* 0.7 appears nowhere in the paper. |
| `scov_vascular` | 0.67 | ExploreASL `xASL_qc_SortBySpatialCoV.m` | ⚠️ **Broken citation chain — see below.** |
| `scov_artifact` | 1.0 | ExploreASL, same file | ExploreASL's tiers: CBF-contrast < 0.67 < vascular < 1.0 < artifactual. |

### ⚠️ The sCoV 0.67 citation chain does not hold up
This is worth flagging explicitly, because 0.67 is widely repeated:

1. ExploreASL 2020 states *"spatial CoV above 0.67 **(Mutsaerts et al., 2018)**."*
2. **Mutsaerts 2018 contains no spatial CoV and no 0.67** (checked twice, independently).
3. **Mutsaerts 2017** — the sCoV paper everyone cites — contains **neither 0.67 nor
   any cutoff**. Its actual content: *"The overall mean GM spatial CoV was
   56.9 ± 13.2% (range 39.3%–113.6%)."*
4. 0.67 is literally **2/3** — an ExploreASL implementation default.

So 0.67 is defensible **as an ExploreASL convention**, not as a Mutsaerts-derived
cutoff, and the code now says so.

**We had also inverted its meaning.** ExploreASL *keeps* the 0.67–1.0 band
(excluding it only from CBF statistics); **>1.0 is its exclusion line**. Our code
FAILed at 0.67 — i.e. it failed scans ExploreASL keeps. Now fixed: >0.67 → WARN,
>1.0 → FAIL.

---

## 3. UNCALIBRATED — our defaults. No source. Please supply.

Ranked by how exposed they are. **These are the numbers to challenge.**

| # | threshold | value | the honest status |
|---|---|---|---|
| 1 | ~~`skew_lo` / `skew_hi`~~ | ~~-0.5 / 1.0~~ | **RETRACTED — see below.** |
| 2 | `ratio_pass` | 1.5 | No source states 1.5. |
| 3 | `gm_cbf_fail_lo` / `_hi` | 10 / 150 | A literature sweep found **zero hits** for a `<10` GM cutoff. **150 is a repurposed reporting number** — ASLPrep's `CBF_THRESH_DEFAULTS = (100,150,200)` are configurable bins for *"% voxels above X"*, with no verdict attached. |
| 4 | `wm_cbf_fail_lo` / `_hi` | 5 / 50 | Provisional engineering bounds. |
| 5 | `neg_gm_warn` / `_fail` | 0.10 / 0.20 | Dolui 2024 uses negative-GM fraction as a **continuous QEI term, never a cutoff**. |
| 6 | `dice_pass` / `dice_warn` | 0.9 / 0.7 | No published Dice cutoff for **registration** QC. 0.7 exists only for **segmentation** (Zou KH, et al. *Acad Radiol* 2004;11(2):178-189). Birn 2023 further shows Dice is **non-monotonic** with registration quality. |
| 7 | `qei_pass` | 0.55 | 0.55 appears nowhere in the QEI paper — our margin above 0.50. |
| 8 | `t1_tissue_s` | 1.4 s | Matches **no** published or implemented value. The White Paper's Table 3 gives no GM T1; ExploreASL uses 1240 ms (GM) / 800 ms (WM); Wansapura 1999 reports ~1331 ms. |
| 9 | `coverage_warn` / `_fail` | 0.90 / 0.75 | Our own defence against the FOV bug (below). |
| 10 | `deep_gm_ratio_lo` / `_hi` | 1.3 / 2.6 | Extrapolated from Miranda 2006's two measured ratios (1.88, 2.05) — the *measurements* are published, the *band* is ours. |
| 11 | `fd_censor_frac_warn` | 0.20 | Our default for "how many censored frames is too many". |

### The skewness threshold — retracted, not re-worded

Asked *"how did you get minus 0.5?"*, the honest answer is:

1. **-0.5 / 1.0 are a generic statistics textbook heuristic** (Bulmer,
   *Principles of Statistics*, 1979: |skew| < 0.5 ≈ "approximately symmetric").
   **Never validated on CBF, ASL, or any medical image.**
2. **The justification previously given was factually wrong.** "ExploreASL-style
   histogram QC" does not exist: `grep -i skew` across all of ExploreASL returns
   **zero hits**, and ASLPrep computes no skewness either. **Neither reference
   pipeline does histogram-skewness QC at all.**
3. `skew_hi` was **dead code** — declared, never referenced.

So the rule was not re-worded — it was **removed**. `2.3.histogram` now reports
skewness and negative fraction as **INFO** (excluded from the overall verdict).
The negative-voxel *verdict* lives in `3.3.negative_gm`, which at least rests on
a physical argument: negative perfusion is impossible.

---

## 4. The GM CBF extraction bug — found, reproduced, fixed

The concern raised in review, verbatim in substance: *"you do it from the T1 space
and warp it to the ASL space … sometimes the cerebellum is not covered in the ASL
scan, and that's still in the ROI. So there is zero values in the CBF, but it's
actually in the ROI. So the gray matter CBF artificially looks low."*

**This was a real defect, and it was worse here than in ASLPrep.** ASLPrep's
`average_cbf_by_tissue()` does not guard against it either; ExploreASL does.

Reproduced with a synthetic case where the ASL covers only 70% of the GM ROI and
true GM CBF is 60:

| | naive (the bug) | fixed |
|---|---|---|
| mean GM CBF | **42.0** — looks hypoperfused | **60.0** — the truth |
| GM/WM ratio | 2.10 | 3.00 |

**Fix:** `utils.masks.covered_tissue_mask()` intersects every tissue mask with
`cbf != 0`, so structural zeros are excluded rather than averaged in. **And so the
exclusion is never silent**, a new check `4.2.coverage` reports the covered
fraction and flags a cropped FOV as a finding in its own right:

> `4.2.coverage  FAIL  only 70.0% of the GM ROI is covered — large FOV mismatch (cerebellum outside the ASL slab?)`

---

## 5. Pathology is not artifact

Raised in review: an altered WM CBF *"could be a pathological finding, not
necessarily the image fail."* Three changes:

1. **`3.2.gm_wm_ratio` informs rather than condemns.** Only `ratio < 1` (the
   published ASLPrep rule) can FAIL. A weak-but->1 ratio WARNs, and the metric
   carries the reason it may be benign. Four innocent explanations for a low ratio:
   - **pathology** — reduced CBF in white-matter hyperintensities / small-vessel disease;
   - **smoothing** — the ratio is smoothing-dependent (Wu 2013: 2.3 unsmoothed → 1.8 at 8 mm; we smooth at 5 mm);
   - **vendor** — smooth GE 3D spiral scans legitimately land at ~1.2–1.3;
   - **age** — it declines ~0.79%/year (Parkes 2004).

2. **sCoV is reported as a transit-time metric, not a quality score.** It
   correlates with GM arterial transit time at **r = 0.85** (Mutsaerts 2017) and
   **rises progressively with cognitive decline** across control → MCI → AD
   (*Sci Rep* 2021;11:23325. doi:10.1038/s41598-021-02313-z). A `sCoV > 0.67 → FAIL`
   rule would therefore **systematically fail dementia patients** — the exact
   population this toolbox is meant to serve.

3. **`QCConfig(strict=False)`** demotes every uncalibrated FAIL to WARN, for
   cohorts where pathology-mimicking-artifact is expected.

---

## 6. Lifespan — adult bands would condemn every neonate

The toolbox was adult-only. It now ships population profiles
(`config.for_population(...)`), because the norms move enormously:

| population | cortical GM CBF | WM CBF | source |
|---|---|---|---|
| neonate (term) | ~16 | ~10 | Miranda MJ, Olofsson K, Sidaros K. *Pediatr Res* 2006;60(3):359-363. doi:10.1203/01.PDR.0000232785.00965.B3 |
| neonate (preterm @ TEA) | ~19 | ~15 | Miranda 2006 |
| infant (~4 mo) | ~38 (whole brain) | — | Kim HG, et al. *AJNR* 2018. doi:10.3174/ajnr.A5774 |
| child | **97 ± 5** | 26 ± 1 | Biagi L, et al. *JMRI* 2007;25(4):696-702. doi:10.1002/jmri.20839 |
| adolescent | 79 ± 3 | 22 ± 1 | Biagi 2007 |
| adult | 58 ± 4 | 20 ± 1 | Biagi 2007; White Paper 40–100 |
| elderly | **46 ± 9** | — | Leoni RF, et al. *Braz J Med Biol Res* 2017;50(4):e5670 |

A child's normal GM CBF (~97) sits at the very top of the adult band; a neonate's
(~16) is **far below** it. `for_population()` **raises** on an unknown name rather
than silently returning adult bands.

### New check `3.4.deep_gm_ratio` (neonatal)
In newborns, CBF is **higher in deep grey matter than cortical GM** — the reverse
of the adult pattern. Miranda 2006 measured basal ganglia + thalami **39 vs
cortical GM 19** (preterm @ TEA) and **30 vs 16** (term), both p<0.0001 → ratios
2.05 and 1.88. A neonatal map *without* that gradient is suspect. N/A outside
neonatal populations.

### ⚠️ The QEI's spatial template is age-locked
`spCBF = 2.5·GM_prob + 1.0·WM_prob` encodes an **adult** 2.5:1 GM:WM ratio.
Measured cortical-GM:WM by age:

| | neonate (preterm) | neonate (term) | child | adolescent | adult |
|---|---|---|---|---|---|
| GM:WM | 1.3 | 1.6 | 3.7 | 3.6 | 2.9 |

So the QEI template is **mis-specified at both ends of the lifespan**. This needs
a decision before QEI is applied to neonatal or paediatric data.

---

## 7. Quantification parameters are now user-supplied, never guessed

Raised in review: *"there are a number of quantification parameters that need to
get into the codes so that you get the right numbers … you have to give the user
option to incorporate those numbers."*

`QCConfig` now carries `labeling_efficiency`, `label_duration_s`,
`post_labeling_delay_s`, `t1_blood_s` — all defaulting to **`None`**, not to a
silent guess. Checks that need them report **UNKNOWN** rather than assuming.

Still needed from you, to make absolute CBF trustworthy on the shared data:
- **GE**: labeling duration, PLD, M0 TR, control/label order
- **Siemens BS-3D**: PLD, labeling duration, background-suppression efficiency

(Note for neonatal work: blood T1 depends on hematocrit, which is highly variable
in newborns, so the adult 1650 ms @3T constant does not transfer —
Varela M, et al. *NMR Biomed* 2011;24(1):80-88. doi:10.1002/nbm.1559.)

---

## 8. Open questions only you can settle

1. **`tissue_thresh`: 0.7 or 0.9?** Your paper says 0.9; your ASLPrep code says
   0.7. You authored both.
2. **`qei_c`: 0.054 or 0.1?** Every other QEI constant rounds cleanly to the
   paper's Fig. 2; this one is a ~1.85× gap.
3. **QEI cut-off: 0.50 or the ROC-derived 0.53?**
4. **The QEI template across the lifespan** — should `2.5·GM + 1·WM` be
   re-weighted per age band (§6)?
5. **Calibration path.** QEI ≈0.5 is the only threshold in the corpus derived by
   ROC against rater labels. That is the model to copy — but it needs **rated
   data**. Can we get expert-labelled scans (or AURA) to ROC-calibrate the 11
   uncalibrated numbers in §3?

---

## Summary

| tag | count | can it FAIL alone? |
|---|---|---|
| published | 11 | yes |
| implementation | 8 | yes, with the caveats noted |
| uncalibrated | 16 | **no** — WARN/INFO only, and `strict=False` softens the rest |

The answer to *"how did you get this number?"* is now recorded in code for every
row — including, where true, **"I didn't, and here is the plan to."**
