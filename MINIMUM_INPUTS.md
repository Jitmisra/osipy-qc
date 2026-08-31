# Minimum inputs for a reliable CBF-map QC

> *"We should define minimum inputs to obtain a reliable QC of the CBF map."*

This document answers that. It is derived from the code, not from the design docs:
every row was read out of `osipy_qc/checks/*.py`, `osipy_qc/io.py`,
`osipy_qc/batch.py` and `osipy_qc/core/`, and every tier in §3 was run through
`run_qc` and its output recorded. Where a docstring and the code disagree, the
code wins and the disagreement is listed in §6.

Nothing here is carried over from `QC_DESIGN.md` or `MODULES_DEEP_DIVE.md`.

**The short answer.** For a *reliable* verdict on a CBF map you need three files:

```
cbf.nii.gz      the quantified CBF map, mL/100g/min
gm.nii.gz       grey-matter probability, on the CBF voxel grid, values in [0, 1]
wm.nii.gz       white-matter probability, same grid, same range
```

That is **tier 3** below. It is the first tier at which the QEI, the GM and WM CBF
levels and the GM/WM ratio all run — i.e. the first tier at which a PASS means
something. A CSF map (tier 4) adds nothing on top for the level checks, but it is
what turns the QEI from `UNKNOWN` into a score, so tier 3 is the practical minimum
and **tier 4 is the exact one** (`load_cbf_inputs` derives CSF for you from
`gm`+`wm`, which is why tier 3 works from the CLI).

A CBF map **on its own** reaches exactly one graded check — `3.5.brain_cbf`, a
plausibility test, not a quality test. §4 explains plainly what that does and does
not license.

---

## 1. How "missing input" is represented

Every check takes one flat `inputs` dict and pulls the keys it needs. A key that
is absent is not an error — the check returns a verdict that says so:

| verdict | meaning | counts toward the overall verdict? |
|---|---|---|
| `PASS` / `WARN` / `FAIL` | the check ran and reached a judgement | yes |
| `UNKNOWN` | the check had **no input to look at** | **no** |
| `N/A` | the check is **structurally inapplicable** to this data | no |
| `INFO` | a description or routing output, never a judgement | no |

`UNKNOWN` used to escalate to `WARN`. It no longer does
(`osipy_qc/core/result.py`), because "I was given nothing to measure" is an
absence, not a finding — and while it did escalate, a reviewer who uploaded only a
CBF map got `WARN` on a flawless map, with no file they could supply short of every
file. A verdict that cannot be earned is not a verdict. The consequence is the
reason this document exists:

> **The overall verdict no longer tells you how much of the report was filled in.**
> `coverage()` does, and it must be shown next to any verdict:
> `{graded, total, unknown, complete, missing}`.
> A `PASS` at `coverage 4/17` and a `PASS` at `coverage 16/16` are very different
> claims.

`total` deliberately excludes `N/A` and `INFO`: a check that cannot apply to this
data was never owed an answer, so counting it as un-covered would understate
coverage.

---

## 2. Per-check input contract

20 checks are registered (`osipy_qc/core/registry.py`; count it with
`len(all_checks())` — if that number is not 20 this table has drifted and the
registry wins). "Required" means the check cannot produce a graded verdict
without it.

### Stream B — the CBF map

| Check | Required | Optional | Returns when a required input is absent | Other non-graded exits |
|---|---|---|---|---|
| `1.qei` | `cbf`, `gm`, `wm`, `csf` | `voxel_mm` (default `(1,1,1)`), `cfg` | `UNKNOWN` — *needs CBF map + GM/WM/CSF tissue maps in ASL space* | `UNKNOWN` if any tissue map's max > 1.001 (a 0–255 or 0–100 segmentation); `UNKNOWN` if ≤1 voxel survives `prob > cfg.tissue_thresh` in **any** of GM/WM/CSF, or if `abs(mean GM CBF) < 1e-6` |
| `2.1.spatial_cov` | `cbf`, `gm` | `cfg` | `UNKNOWN` — *needs CBF + GM mask* | `UNKNOWN` if fewer than 2 **positive** GM voxels |
| `2.2.snr` | `cbf` + `gm`, **or** `asl_4d` | `brain`, `cfg` | `UNKNOWN` — *needs GM CBF map or a 4D series* | `INFO` when only tSNR could be computed (a 4D series but no GM CBF). Never `FAIL` — sCoV owns that verdict |
| `2.3.histogram` | `cbf`, `gm` | `cfg` | `UNKNOWN` — *needs CBF + GM mask* | Always `INFO` when it does run; `UNKNOWN` if fewer than 2 GM voxels |
| `3.1.cbf_level` | `cbf`, `gm`, `wm` | `cfg` | `UNKNOWN` — *needs CBF + GM/WM masks* | `UNKNOWN` if either coverage-masked tissue mask is empty |
| `3.2.gm_wm_ratio` | `cbf`, `gm`, `wm` | `cfg` | `UNKNOWN` — *needs CBF + GM/WM masks* | `UNKNOWN` if either masked tissue set is empty; **`FAIL`** if mean GM ≤ 0 or mean WM ≤ 0 (two negatives would otherwise divide to a plausible positive ratio) |
| `3.3.negative_gm` | `cbf`, `gm` | `is_raw_deltam`, `cfg` | `UNKNOWN` — *needs CBF + GM mask* | `N/A` if `is_raw_deltam` is truthy; `UNKNOWN` if the masked GM set is empty |
| `3.4.deep_gm_ratio` | `cbf`, `deep_gm`, `cortical_gm`, **and** `cfg.population` starting with `neonate` | `cfg` | `N/A` outside neonatal populations; `UNKNOWN` — *needs CBF + separate deep-GM and cortical-GM masks* — inside them | `WARN` if either mean is ≤ 0. Never `FAIL` |
| `3.5.brain_cbf` | `cbf` | `cfg` | `UNKNOWN` — *needs a CBF map* | **`N/A` whenever `gm` is present** — `3.1` then grades GM and WM against published bands, and two magnitude verdicts would double-count one problem. `UNKNOWN` if the map has no non-zero voxels. This is the only check that grades a CBF map with no tissue maps at all |
| `4.1.coregistration` | `asl_mask`, `struct_mask` | `cfg` | `UNKNOWN` — *needs both an ASL mask and a structural (T1) mask* | `UNKNOWN` on differing shapes; `UNKNOWN` if either mask is empty |
| `4.2.coverage` | `cbf`, `gm` | `wm`, `cfg` | `UNKNOWN` — *needs CBF + GM mask* | `UNKNOWN` when the worst coverage is exactly `0.0` |

### Stream A — the raw acquisition

| Check | Required | Optional | Returns when a required input is absent | Other non-graded exits |
|---|---|---|---|---|
| `5.1.schema` | `sidecar` (a BIDS JSON dict) **or** `detected` | — | `UNKNOWN` — *no sidecar and nothing to infer from* | `WARN` if a sidecar is present but missing any of `ArterialSpinLabelingType`, `MRAcquisitionType`, `PostLabelingDelay`; `WARN` if there is no sidecar but `detected` is present. **`PASS` requires a `sidecar` dict** |
| `5.2.volume_integrity` | `asl_4d` **or** `n_volumes` | `structure` | `UNKNOWN` — *volume count unknown* | `N/A` if `structure` contains `"pre-subtracted"`; `N/A` for a single volume |
| `5.3.swap` | `asl_4d` | `background_suppression`, `structure`, `cfg` | `UNKNOWN` — *needs the 4D control/label series* | `N/A` if `background_suppression` is truthy, if pre-subtracted, or if the array is not 4D with ≥2 volumes; `UNKNOWN` if the control or label slab is entirely non-finite |
| `6.1.m0_present` | `m0_type` | — | `UNKNOWN` — *M0 type not determined* | `WARN` for any value other than `"separate"` / `"included"` |
| `6.2.m0_tr` | `m0_tr_s` | `cfg` | `UNKNOWN` — *M0 TR unknown* | `WARN` + a correction factor below 5 s. Never `FAIL` |
| `6.3.m0_no_bs` | `m0_background_suppression` (bool) | — | `UNKNOWN` — *M0 BS status unknown* | `FAIL` when `True` |
| `6.5.m0_geometry` | `m0_shape`, `asl_shape` | — | `UNKNOWN` — *missing M0 or ASL geometry* | `WARN` on a grid mismatch |
| `7.1.motion` | `motion_params` (T×6) and/or `asl_4d` | `brain`, `cfg` | `UNKNOWN` — *needs motion parameters or a 4D series to estimate motion* | `INFO` (DVARS only) when `asl_4d` is given without `motion_params`; `UNKNOWN` if `motion_params` is entirely non-finite |
| `8.2.data_type` | `files` (`[{name, shape, voxel_mm}]`) | `context` | `UNKNOWN` — *no files to inspect* | Always `INFO` when it does run |

### Verdict boundaries worth stating exactly

These are read off the comparisons in the code, not off the config comments:

* `1.qei` — `PASS` at `qei >= qei_pass` (0.55), `WARN` at `>= qei_warn` (0.50),
  else `FAIL`. The `FAIL` is not softened by `strict=False`, because the 0.50
  line it crosses is the published one.
* `3.1.cbf_level` — `FAIL` when `mean <= fail_lo` **or** `mean > fail_hi`. Note
  the asymmetry: the low fail bound is *inclusive*, the high one *exclusive*.
  `PASS` inside `[lo, hi]`, `WARN` between. Overall = the worse of GM and WM.
* `2.2.snr` — `PASS` at `spatial_snr > 1/scov_vascular` (≈ 1.49), else `WARN`.
  `spatial_snr` is algebraically `1/sCoV`, so it carries no information
  `2.1.spatial_cov` does not.
* `4.2.coverage` — `PASS` at `>= coverage_warn` (0.90), `WARN` at
  `>= coverage_fail` (0.75), else `FAIL`/`WARN` by `strict`.
* `3.5.brain_cbf` — `FAIL` at `mean <= brain_cbf_absurd_lo` (0.0) or
  `mean >= brain_cbf_absurd_hi` (300.0), `PASS` in between. These are bounds on
  *gross implausibility*, not on normality: they answer "can this be a quantified
  CBF map at all", and 3.1 answers "is the level right".
* Most `FAIL`s produced by an **uncalibrated** threshold are softened to `WARN`
  when `cfg.strict=False`, and are tagged `provisional=True` on the result.
  `3.5.brain_cbf` is the exception: its two bounds are uncalibrated and its FAIL
  is tagged provisional, but it is **not** gated on `cfg.strict`, so `strict=False`
  does not soften it (see §6.4).

---

## 3. Input tiers

Each row was produced by feeding exactly those keys to `run_qc` with the default
adult config and recording the result. `coverage` is `graded/total`.

| # | You have | Files / keys | Checks that become gradeable | `coverage` | Overall on a clean map | **What a reviewer may legitimately conclude** |
|---|---|---|---|---|---|---|
| 0 | nothing | — | none | `0/19` | `UNKNOWN` | Nothing. |
| 1 | **CBF map alone** | `cbf` | `3.5` only | `1/19` | `PASS` | Only that the magnitude does not rule out a quantified CBF map — the brain mean is positive and below 300. **Nothing about quality**: no QEI, no GM level, no contrast, no noise measure. Read §4 before quoting this PASS. |
| 2 | CBF + GM | `cbf`, `gm` | `2.1`, `2.2`, `3.3`, `4.2` (+ `2.3` as INFO; `3.5` becomes `N/A`) | `4/17` | `PASS` | That GM perfusion is *internally* consistent: its spatial CoV sits in a sensible tier, few voxels are negative, and the map covers the GM ROI. You cannot conclude the map is correctly calibrated, nor that GM/WM contrast exists. |
| 3 | **CBF + GM + WM** ← the practical minimum | `cbf`, `gm`, `wm` | tier 2 **+ `3.1`, `3.2`** (and `1.qei` too, once CSF is derived — see the note) | `6/17` on raw arrays, `7/17` via `load_cbf_inputs` | `PASS` | The full CBF-map judgement: QEI against the published 0.5 cut-off, absolute GM and WM level against population bands, and GM/WM contrast. A `PASS` here is a defensible statement that the map is usable. |
| 4 | CBF + GM + WM + CSF | `cbf`, `gm`, `wm`, `csf` | tier 3 **+ `1.qei` unconditionally** | `7/17` | `PASS` | Same conclusions as tier 3, but the QEI's dispersion term uses **your** CSF map rather than a derived one. Prefer this when your pipeline already produced a CSF map. |
| 5 | **raw ASL alone** | `asl_4d`, `files`, `context`, `structure`, `m0_type` | `5.1` (WARN), `5.2`, `5.3`, `6.1` (WARN) | `4/16` | `WARN` | That the series is structurally sound: an even number of volumes, control brighter than label. Nothing about the CBF map, which does not exist yet. Both WARNs are real findings — no BIDS sidecar, and no M0. |
| 6 | raw + M0 | tier 5 + `m0_shape`, `m0_type="separate"` | tier 5 + `6.5`, and `6.1` becomes PASS | `5/16` | `WARN` | Additionally that a calibration scan exists and sits on the ASL voxel grid. **You still cannot conclude the M0 is valid** — its TR (`6.2`) and its background-suppression status (`6.3`) are not in a NIfTI header and stay `UNKNOWN`. |
| 7 | raw + M0 + acquisition metadata | tier 6 + `sidecar`, `m0_tr_s`, `m0_background_suppression`, `motion_params` | tier 6 + `6.2`, `6.3`, `7.1`, and `5.1` becomes PASS | `8/17` | `PASS` | The complete raw-data judgement: protocol fields present, M0 TR adequate and BS off, head motion within Adebimpe's 1 mm mean-FD exclusion rule. A structural T1 on its own adds nothing — see §6.1. |
| 8 | everything, on one grid | tiers 4 + 7 + `asl_mask`, `struct_mask`, `brain` | 16 graded, 2 `INFO`, 2 `N/A` | `16/16` | `PASS` | Both streams. This is the only configuration in which `coverage.complete` is `True` for an adult, and it can currently only be reached by building the `inputs` dict yourself in Python (§6). |

Notes on the numbers:

* Tier 3's `coverage` depends on **how** you supply the maps.
  `load_cbf_inputs(cbf, gm=…, wm=…)` derives `csf` as
  `clip(1-gm-wm, 0, 1)` restricted to covered voxels, so the QEI runs and you get
  `7/17`. Handing `run_qc` a dict with only `cbf`/`gm`/`wm` — which the Python API
  lets you do — leaves `1.qei` at `UNKNOWN` and `6/17`. Use the loader.
* `3.4.deep_gm_ratio` is `N/A` in every adult row, and `3.5.brain_cbf` is `N/A`
  in every row where `gm` is present — which is why `total` is 16–19 rather than
  20. Under `--population neonate`, `3.4` becomes `UNKNOWN` instead and `total`
  rises by one, because no loader supplies `deep_gm`/`cortical_gm`.
* Tier 8 needs `brain` and `asl_4d` on the **same voxel grid**. They are not
  checked against each other: a mismatch raises inside `2.2.snr` and
  `7.1.motion`, and `run_qc` catches it as
  `UNKNOWN — check error: operands could not be broadcast together …`. So a grid
  mistake presents as "no input" rather than as an actionable message.
* The `coverage` column above is what `run_qc` reports over the **whole registry**,
  which is what the CLI does. Callers that pass `checks=` see a smaller
  denominator for the same inputs: `batch.grade_folder` uses `cbf_map_checks()`
  (10 checks), so tier 4 there reads `7/7` rather than `7/17`, and
  `web._checks_for()` picks Stream A only, Stream B only, or everything-but-`4.1`
  depending on what was uploaded. The **fraction** is what matters, not the
  denominator: `7/7` and `7/17` are both honest, because in the first case the
  missing checks were never asked.

---

## 4. Be honest: what a lone CBF map actually buys you

**One plausibility test, and nothing that deserves the word *quality*.**

Almost every CBF-map check needs a grey-matter mask, because almost every one of
them is a statement about **grey matter**: the QEI correlates the map against a
GM/WM template, the CBF level is a GM and WM mean, the ratio is GM over WM, the
negative fraction is counted inside GM, sCoV is the CoV of GM. Strip the tissue
maps and there is no region left to say those things about.

The one exception is `3.5.brain_cbf`, which derives its own crude brain mask from
the CBF map (`brain_mask_fallback`: magnitude above the 50th percentile of the
non-zero voxels) and asks a single, much weaker question — *is the mean positive
and below 300?* So the report comes back:

```
$ osipy-qc --dashboard <folder with one CBF-only subject>
  sub-06     PASS     QEI=None  coverage 1/9
```

Read that literally. `coverage 1/9` is the whole message: one check of nine
reached a verdict. The `PASS` means *"nothing about the magnitude rules out this
being a quantified CBF map"* — it does **not** mean the map is good, and
`3.5.brain_cbf`'s own docstring is explicit that it cannot even distinguish CBF
from a raw ΔM in scanner units (measured on the three real test datasets, raw ASL
series give brain means of 191, 482 and 905; the first would PASS).

The check deliberately does **not** grade the brain mean against a normal band,
for two reasons stated in its docstring and worth repeating to a reviewer:

1. **No published bound exists.** Every magnitude bound in the ASL literature is
   for grey matter (White Paper p.17: *"gray matter CBF values from 40–100
   ml/min/100ml can be normal"*) or for a ratio (ASLPrep excludes GM/WM < 1).
   Published "global" figures are a different quantity — computed in parenchyma
   masks with CSF excluded — so a brain-mask mean that includes ventricles reads
   systematically lower and is not comparable.
2. **The quantity is not stable.** The mean moves 41 → 60 mL/100g/min as
   `brain_mask_percentile` goes 25 → 75. A threshold on a number an internal knob
   moves by a third of any plausible band's width is not measuring the scan.

Note also what the aggregation rule does here. Before `3.5` existed, a lone CBF
map returned `UNKNOWN` — the honest answer when `coverage.graded == 0`, since
`aggregate()` only returns `PASS` once at least one check has actually graded.
Now that `3.5` grades, the same upload returns `PASS`. **The verdict got more
optimistic without the evidence improving**, which is exactly the situation the
coverage rule exists for: a `PASS` at `1/9` must never be rendered without its
coverage beside it.

**The underlying limitation is real, and is not a feature.** A reviewer with a
folder of CBF maps and no segmentations still gets no *quality* verdict, and the
tool cannot fix that for them: it is a QC layer, not a pipeline, so it will not
run a T1 segmentation to manufacture the masks (that is `PyASL` / `ASLPrep` /
`oxford_asl` territory, and a QC tool that silently invents its own reference
frame is worse than one that declines). Two honest ways out:

1. Supply the maps your pipeline already wrote. `oxford_asl --pvcorr` leaves
   `pvgm_inasl` / `pvwm_inasl` next to `perfusion_calib`; ASLPrep writes
   `*_label-GM_probseg` / `*_label-WM_probseg`. `find_oxford_asl()` and
   `find_aslprep()` locate both.
2. Accept a rough, clearly-labelled fallback.
   `examples/grade_cbf_map.py` with no `--gm/--wm` substitutes a whole-brain mask
   from `brain_mask_fallback()` for *both* GM and WM and runs only
   `3.1`, `2.1`, `2.3`, `3.3`. Read those numbers as whole-brain, not GM: the
   GM/WM ratio is meaningless (it is 1.0 by construction) and the QEI is not
   attempted. It is enough to catch a gross calibration failure — a map that is
   7× physiological — and nothing subtler. Note what the substitution costs: it
   makes `gm` non-`None`, so `3.1.cbf_level` grades a **whole-brain** mean against
   the **grey-matter** band — a stricter and less defensible claim than
   `3.5.brain_cbf`'s plausibility bound, and the reason the bundled example map
   comes back `WARN` on `WM 48.3` when the "WM" values are really whole-brain
   ones. The script also hard-codes its four checks, so `3.5` never runs there at
   all. Verified:

   ```
   $ python examples/grade_cbf_map.py example_data/example_cbf.nii.gz
   mode    : ROUGH whole-brain mask (no tissue maps) -> level + noise only
   OVERALL : WARN
     2.1.spatial_cov    PASS   sCoV 26.9% (CBF-contrast)
     2.3.histogram      INFO   skew -0.75, 0.0% negative (informational)
     3.1.cbf_level      WARN   GM 48.3 (PASS), WM 48.3 (WARN) [adult bands]
     3.3.negative_gm    PASS   0.0% negative GM voxels
   ```

---

## 5. Dependency table: input → checks unlocked

Read downward: each key is listed once, against the checks that cannot grade
without it.

| Input key | Type | Unlocks | Supplied by |
|---|---|---|---|
| `cbf` | 3-D float array | every Stream-B check; on its own, `3.5` | `load_cbf_inputs`, `_find_cbf_inputs`, `--demo`, web upload |
| `gm` | 3-D prob array, ≤1 | `1.qei`, `2.1`, `2.2`, `2.3`, `3.1`, `3.2`, `3.3`, `4.2` — and **switches `3.5` off** (`N/A`) | as `cbf` |
| `wm` | 3-D prob array, ≤1 | `1.qei`, `3.1`, `3.2`; refines `4.2` | as `cbf` |
| `csf` | 3-D prob array, ≤1 | `1.qei` | as `cbf`; **derived** from `gm`+`wm` when omitted |
| `voxel_mm` | 3-tuple mm | the QEI's 5 mm FWHM smoothing kernel | both loaders (from the NIfTI zooms) |
| `brain` | 3-D bool array | narrows `2.2` tSNR and `7.1` DVARS | `--demo` and `demo_cohort()` only |
| `deep_gm`, `cortical_gm` | 3-D prob arrays | `3.4` | **nothing** |
| `asl_mask`, `struct_mask` | 3-D bool arrays | `4.1` | **nothing** |
| `asl_4d` | 4-D array | `5.2`, `5.3`, `7.1` (DVARS), `2.2` (tSNR) | `load_folder` |
| `motion_params` | (T, 6) array, mm + rad | `7.1` FWD — the only graded motion path | **nothing** |
| `files` | `[{name, shape, voxel_mm}]` | `8.2` | `load_folder` |
| `context` | str (folder name) | vendor / BS inference inside `8.2` | `load_folder` |
| `detected` | dict from `detect_dataset` | `5.1` fallback (→ `WARN`) | `load_folder` |
| `structure` | str | gates `5.2`, `5.3` to `N/A` on pre-subtracted data | `load_folder` |
| `background_suppression` | bool or `None` | gates `5.3` to `N/A` | `load_folder` (`True` or `None`, never `False`) |
| `sidecar` | BIDS JSON dict | `5.1` → `PASS` | **nothing** |
| `m0_type` | `"separate"` / `"included"` / `"absent"` | `6.1` | `load_folder` (`"separate"`/`"absent"` only) |
| `m0_shape`, `asl_shape` | shape tuples | `6.5` | `load_folder` |
| `m0_tr_s` | float seconds | `6.2` | **nothing** |
| `m0_background_suppression` | bool | `6.3` | **nothing** |
| `is_raw_deltam` | bool | gates `3.3` to `N/A` | **nothing** |
| `n_volumes` | int | `5.2` without loading the array | **nothing** (`detect_dataset` computes it but `load_folder` does not pass it on) |

---

## 6. Known gaps

Verified against the code, one by one. These are the honest edges of v1.0.

### 6.1 `4.1.coregistration` is unreachable

`coregistration_check(asl_mask=None, struct_mask=None, …)` is the only consumer
of those two keys, and **no loader produces either**:

```
$ grep -rn "asl_mask" osipy_qc/
osipy_qc/checks/coreg.py:25:def coregistration_check(asl_mask=None, struct_mask=None,
osipy_qc/checks/coreg.py:28:    if asl_mask is None or struct_mask is None:
osipy_qc/checks/coreg.py:31:    a = np.asarray(asl_mask, dtype=bool)
```

Three hits, all inside the check. `load_folder` classifies a T1 file
(`detected["t1_structural"] = True`) but never skull-strips it, and
`load_cbf_inputs` has no structural argument at all. So the check returns `UNKNOWN`
for every real invocation, and two call sites exclude it by name for exactly that
reason (`batch.cbf_map_checks()` and `web._checks_for()`). Its Dice/Jaccard maths is
unit-tested (`tests/test_coreg.py`) by passing masks in directly, which is the only
way to exercise it.

**To close it**, the toolbox needs a brain-mask source it does not currently have.
A QC-grade option that stays inside the "no heavy processing" scope: threshold the
CBF map for the ASL mask, threshold the T1 for the structural mask, and require
the caller to have already resampled them onto one grid — which the check already
enforces (`a.shape != b.shape → UNKNOWN`).

### 6.2 Other unreachable inputs

Same test — a `grep` for the key across `osipy_qc/` returns only the check that
consumes it:

| Key | Check left permanently `UNKNOWN` | Why it matters |
|---|---|---|
| `motion_params` | `7.1.motion` | Motion **never reaches a graded verdict** through any shipped entry point. With a 4D series it reports DVARS as `INFO`; the published FD rule (mean FD > 1 mm, Adebimpe 2022) is wired up but never fed. Motion parameters come from MCFLIRT / SPM realign, i.e. from the pipeline — reading a `*_motion.tsv` or `.par` alongside the NIfTIs would close this. |
| `m0_tr_s` | `6.2.m0_tr` | The White Paper's 5 s M0 TR rule can never fire. TR is in the DICOM and in a BIDS sidecar, not in a NIfTI header. |
| `m0_background_suppression` | `6.3.m0_no_bs` | The "M0 must be acquired without BS" flag can never fire — even though `detect_dataset` *does* infer a dataset-wide `background_suppression` from the folder name. That inference is wired to `5.3.swap` only; it is never passed as `m0_background_suppression`. This is the closest gap to a fix, and the one a mentor is most likely to ask about. |
| `sidecar` | `5.1.schema` can never `PASS` | Nothing in the package reads a `.json` file: `io._find_niftis` filters on `.nii`/`.nii.gz` only. So from a folder, `5.1` is `WARN` at best ("no BIDS sidecar — fields inferred"), which is honest but means a properly-BIDS-organised dataset gets no credit for it. |
| `deep_gm`, `cortical_gm` | `3.4.deep_gm_ratio` | The neonatal check is `N/A` for adults and `UNKNOWN` for neonates. Verified: `osipy-qc --demo --population neonate` reports *needs CBF + separate deep-GM and cortical-GM masks*. |
| `is_raw_deltam` | the `N/A` branch of `3.3.negative_gm` | Negative voxels are legitimate in a pre-subtracted ΔM, and the check knows it, but nothing ever sets the flag — so grading a raw ΔM as if it were CBF would flag its negatives as a failure. `load_folder` *does* detect `structure == "pre-subtracted deltaM"`; it just is not forwarded. |
| `n_volumes` | nothing (redundant) | `detect_dataset` computes it and `load_folder` drops it, so `5.2` re-derives it from `asl_4d`. Harmless today, but it means `load_folder(folder, load_arrays=False)` silently loses `5.2`. |
| `brain` | nothing (optional) | Only `--demo` and `demo_cohort()` supply it. On real data `2.2` falls back to `tmean != 0` and `7.1` computes DVARS over the whole volume. |

### 6.3 The organ switch is inert

`cfg.organ` exists, defaults to `"brain"`, and is **read in two places, both
purely for display**:

```
osipy_qc/report_html.py    "population: … · organ: {cfg.organ}"   # a report caption
osipy_qc/api.py            "organ": cfg.organ                     # a JSON field, twice
```

`for_organ()` and `skipped_for_organ()` are defined in `core/config.py` and are
called from **nothing but `tests/test_meeting_followups.py`**. No CLI flag, no web
control, and no code path applies `ORGANS["kidney"]["skip_checks"]`. So selecting
kidney today changes a label and nothing else — `1.qei` would still run and still
correlate a renal CBF map against a `2.5*GM + 1*WM` brain template.

This is honest as a documented stub and dangerous as a silent one. Until the skip
list is actually applied by the runner, the switch should not be exposed in any
UI, and the kidney profile should be described as *planned*, not *supported*.

### 6.4 `3.5.brain_cbf` is not wired into the surrounding machinery

The check itself is careful and heavily documented. Three things around it are
not yet connected:

* **No human label.** `batch.CHECK_LABELS` has no `"3.5.brain_cbf"` entry, so
  `check_label()` falls through to the raw id. The cohort ledger's flag column and
  `BatchSummary.artifact_breakdown` will show `3.5.brain_cbf` where every other
  check shows a name like *CBF level*.
* **No provenance record.** `brain_cbf_absurd_lo`, `brain_cbf_absurd_hi` and
  `brain_mask_percentile` are absent from `THRESHOLD_PROVENANCE`, so
  `osipy-qc --provenance` does not list them (it still reports 11 / 9 / 16) and
  `provenance_of()` returns the fallback *"No provenance recorded."* They are
  uncalibrated and the code says so in a comment; the machine-readable table that
  the report prints from does not know it.
* **Not tunable.** They are not in `batch.TUNABLE`, so neither the dashboard
  threshold panel nor the upload form's `thr_*` fields can move them — although
  the upload form's `thr_*` handler accepts any `QCConfig` attribute, so
  `thr_brain_cbf_absurd_hi` would in fact work there by accident.
* **Not gated on `strict`.** Every other uncalibrated `FAIL` in the package is
  written `Verdict.FAIL if cfg.strict else Verdict.WARN`. `3.5`'s two `FAIL`s are
  unconditional (they are tagged `provisional=True`, but `strict=False` does not
  soften them), so this is the one uncalibrated threshold that can hard-FAIL a
  clinical cohort.

### 6.5 Config and registry fields that nothing acts on

* **`registry` `required=` flag.** `register_qc_check(..., required=True/False)`
  is stored and then surfaced only in `api.checks_catalogue()`'s JSON. `run_qc`
  never reads it, so a "required" check that returns `UNKNOWN` is treated exactly
  like an optional one. The five checks marked `required=False`
  (`2.3.histogram`, `3.4.deep_gm_ratio`, `3.5.brain_cbf`, `6.5.m0_geometry`,
  `8.2.data_type`) behave no differently from the fifteen marked `required=True`.
* **Quantification parameters.** `labeling_efficiency`, `label_duration_s`,
  `post_labeling_delay_s`, `t1_blood_s` all default to `None` and are read by
  **no check**. They are documented as "acquisition facts the QC layer must be
  told", and today nothing tells them and nothing asks.
* **`t1_tissue_s`** is read only by `6.2.m0_tr`, which is itself unreachable
  (§6.2) — so the one uncalibrated constant in the M0 correction factor is never
  exercised on real data.
* **`cfg.strict`** is honoured by seven checks (`2.1`, `3.1`, `3.2`, `3.3`, `4.1`,
  `4.2`, `7.1`) and is exposed on the web form, but there is **no CLI flag** for
  it. From the command line you cannot grade a clinical cohort leniently.
* **`--population` is silently ignored in dashboard and serve modes.** `cli.py`
  validates it only after the `--dashboard` / `--dashboard-demo` / `--serve`
  branches have returned, so `osipy-qc --dashboard X --population toddler`
  starts happily and grades against **adult** bands. Verified:
  `/api/cohort` reports `"population": "adult"`.

### 6.6 Code / docstring disagreements found while writing this

| Where | Says | Code does |
|---|---|---|
| `core/config.py` `ORGANS["brain"]["note"]` | "All **17** checks apply." | 20 are registered. `README.md` and `USAGE.md` repeat the 17. |
| `core/result.py` module docstring | "the JSON, HTML and web reports all show it [coverage]" | Only the JSON shows it: `QCReport.to_dict()["coverage"]` and `api.subject_payload()["coverage"]`. `report_html.py` renders the per-verdict count chips (which do include the UNKNOWN tally) but not the `coverage` dict, and `api.ledger_row()` — the cohort table — carries no coverage at all. `grep -n coverage osipy_qc/report_html.py` returns only `4.2.coverage`, the FOV check, which is a **different quantity with a colliding name**. |
| `batch.Subject.primary_artifact` → `api.ledger_row()["incomplete"]` | flags a subject whose report was only partial | Dead in practice. The branch is gated on `self.overall == "WARN"`, and since UNKNOWN stopped escalating, UNKNOWNs can no longer *produce* a WARN — so a partial report reaches the ledger unflagged. Measured on a cohort: `sub-06` graded `1/9` and `sub-08` graded `0/9` both come back `'flag': '-', 'incomplete': False`. **The cohort view therefore renders a verdict with no completeness beside it, and its one completeness signal now reads False when it should read True.** This is the single most important consequence of the aggregation change still to be wired up. |
| `batch.grade_folder` docstring | "Defaults to the CBF-map stream (`checks=stream_b_checks()`)" | Defaults to `cbf_map_checks()`, which is `stream_b_checks()` **minus** `4.1.coregistration`. |
| `batch.stream_b_checks` and `batch.cbf_map_checks` docstrings | justify excluding checks because their UNKNOWNs would "drag every subject to WARN" | true before the aggregation change, not now. The exclusions are still right — they keep `coverage` honest rather than the verdict — but the stated reason is stale. `web._checks_for()` already carries the corrected wording, so the two files now disagree with each other. |
| `cli.py` `--demo` help | "run the full **Stream B** on synthetic data" | runs the **whole registry** with `checks=None`, so the 9 Stream-A checks plus `4.1.coregistration` return `UNKNOWN`. That is why `--demo` reports `coverage 7/17` and `{'PASS': 7, 'INFO': 1, 'N/A': 2, 'UNKNOWN': 10}` on a flawless map. |
| `cli.py` module docstring | lists 3 invocations | 10 flags exist, including `--serve`, `--dashboard`, `--dashboard-demo`, `--html`, `--provenance`, `--population`, `--host`. |
| `cli.py` × 3 | `ap.error(msg); return 2` | `ap.error` raises `SystemExit(2)`; the `return 2` is dead. |
| `checks/noise.py` `spatial_cov_check` | `if mean == 0: return UNKNOWN` | unreachable — `_positive_gm` filters to `vals > 0`, so a non-empty set always has a positive mean. |
| `cli.py:40` `_print_provenance`, and `report_html.py:535` | "Full write-up: `THRESHOLD_PROVENANCE.md`" | that file does not exist in the repo. `USAGE.md` links it too, and both link the equally absent `POPULATION_BANDS.md`. (`README.md` no longer does — it points at `core/config.py` instead.) |

---

## 7. The input contract, as a checklist

Before trusting a CBF-map verdict, confirm all of these:

- [ ] `cbf` is **quantified CBF in mL/100g/min**, not a raw ΔM and not an
      arbitrary-unit perfusion-weighted image. Nothing in the tool can tell the
      difference — a NIfTI does not state its units — so `3.1.cbf_level` will
      simply report the wrong level, and `3.5.brain_cbf` says so in its own
      docstring. `8.2.data_type` is the check that establishes what a file is, and
      it needs the raw files, not the map.
- [ ] `gm` and `wm` are **probability / partial-volume maps in `[0, 1]`**. A
      0–255 or 0–100 segmentation is rejected with an actionable message by
      `load_cbf_inputs`, and by `1.qei` if you bypass the loader — because
      thresholding a 0–255 map at 0.7 makes every tissue mask the whole brain and
      the score comes out plausible and wrong.
- [ ] `gm`, `wm`, `csf` are on **exactly the CBF voxel grid**. `load_cbf_inputs`
      raises on a shape mismatch and tells you to resample; it does not resample
      for you.
- [ ] The population is right. `--population neonate` for newborns — the adult
      40–100 GM band fails every neonatal scan, and the neonatal band `WARN`s
      every adult one (verified: `--demo --population neonate` →
      `3.1.cbf_level WARN GM 56.0`).
- [ ] The verdict is being read **next to its coverage**. If your consumer only
      reads `overall_verdict`, it is reading half the answer.
