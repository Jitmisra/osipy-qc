# osipy-qc — Usage & Function Reference

Rough, working reference — not the final documentation, just an accurate map of
every function and its options right now. The basic shape of the API won't
change; this just makes it visible.

For a narrative "try it in 30 seconds" walkthrough, see [README.md](README.md).
This doc is the exhaustive one: every function, every parameter.

---

## 1. Command line

```bash
osipy-qc --serve                 # web UI: upload a CBF map in the browser  <- easiest
osipy-qc --demo                  # grade a built-in synthetic CBF map (full Stream B)
osipy-qc <folder>                # QC a folder of raw NIfTIs (Stream A)
osipy-qc <folder> --json         # same, machine-readable JSON
osipy-qc --demo --html r.html    # write a self-contained visual report
osipy-qc --provenance            # where every threshold came from
osipy-qc --organ kidney <folder> # grade a kidney dataset (19 checks) instead of a brain one
osipy-qc --organ-demo placenta   # see an organ's report without owning data of that organ
osipy-qc --dashboard output/     # grade a whole cohort into one sortable page
osipy-qc --version               # which build produced this report
python -m osipy_qc <folder>      # identical, module form
```

| flag | type | meaning |
|---|---|---|
| `folder` | positional, optional | path to a folder of raw `.nii`/`.nii.gz` files. Recurses into subfolders. Not needed with `--demo`/`--serve`. |
| `--json` | flag | print the full JSON report instead of the human-readable table |
| `--demo` | flag | ignore `folder`; grade a synthetic "clean" CBF map instead |
| `--html PATH` | path | also write a self-contained visual HTML report (images + histograms) |
| `--population NAME` | str | CBF bands to grade against: `adult` (default) or `neonate` |
| `--provenance` | flag | print the source of every threshold, then exit |
| `--serve` | flag | start the local web UI (upload → report) |
| `--port N` | int | port for `--serve` (default 8000) |
| `--no-browser` | flag | with `--serve`, don't auto-open a browser |
| `--host ADDR` | str | bind address for `--serve` (default `127.0.0.1`). Anything other than loopback is a **public** bind and disables the DNS-rebinding guard, so it must be typed deliberately — it is not read from the environment. |
| `--organ NAME` | str | which organ's check set to run: `brain` (default, 20 checks), `kidney` (19), `placenta` (15). Uses the mask-aware loader, so masks are routed rather than dropped. |
| `--organ-demo NAME` | str | grade a synthetic phantom of known quality for that organ. Lets you see a kidney or placenta report without owning data of either. |
| `--dashboard DIR` | path | grade every subject folder under `DIR` and write one sortable cohort page |
| `--dashboard-demo` | flag | the same cohort page, built from synthetic subjects |
| `--no-strict` | flag | demote every *provisional* FAIL (one decided by an uncalibrated threshold) to a WARN, leaving only published-threshold failures |
| `--version` | flag | print the version and exit |

### The web UI
```bash
osipy-qc --serve            # -> http://127.0.0.1:8000
```
Upload a CBF map (+ optional GM/WM/CSF), pick the population, get the full visual
report. The uploaded files are written to a temp folder and deleted as soon as
grading finishes.

**A folder of subject folders is graded as a cohort.** Drop `my_cohort/` where
each `sub-XX/` holds that subject's files, and one page comes back with the
ledger — worst subject first — plus every subject's full report underneath it.
Same self-contained HTML as a single scan, so it can be emailed.

A folder counts as a subject when it holds a CBF map or an ASL series. That rule
is what stops a single BIDS subject (`sub-01/anat` + `sub-01/perf`, two folders,
one person) being read as a two-subject cohort. Each subject is then graded on
the checks *its own* files justify, so a cohort of bare CBF maps is not dragged
to WARN by ten Stream-A checks with nothing to look at, and a subject that
shipped its raw series gets those checks for real — two subjects in one cohort
can legitimately be graded on different sets, and each report says how many.

Capped at 12 subjects per upload (`OSIPY_MAX_COHORT`), because every subject's
arrays are held at once so its figures can be drawn — that is a memory ceiling,
not a politeness limit. Above 8 subjects the per-subject figures are dropped and
the page says so; four mosaics each would run it into the tens of megabytes.

For a cohort already sitting on the machine running the server, `--dashboard
FOLDER` grades it in place and serves the interactive React dashboard instead. The graded arrays of the **last few uploads** are then held in
memory so their figures can be drawn on demand; they are evicted as new uploads
arrive and are gone when the process stops. Nothing is written to disk and
nothing is sent anywhere.

> ⚠️ **Local by default, and that default matters.** The server binds `127.0.0.1`
> and refuses requests whose `Host` header is not loopback (a DNS-rebinding guard).
> There is no auth, no rate limiting, and no sandbox around the NIfTI parser.
>
> `--host 0.0.0.0` makes it public and turns that guard off, which is why the flag
> exists but is never read from the environment — a stray `HOST` variable used to be
> enough to move the bind off loopback silently. The public demo at
> [osipy-qc.onrender.com](https://osipy-qc.onrender.com) runs exactly that way,
> deliberately, on synthetic and user-supplied uploads only. **Do not point a public
> instance at real participant data.**

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
| `2.3.histogram` | B | `cbf, gm` | GM CBF shape — **INFO only, never graded** (no published skewness cutoff exists) |
| `3.1.cbf_level` | B | `cbf, gm, wm` | mean/median GM & WM CBF in range (population-dependent) |
| `3.2.gm_wm_ratio` | B | `cbf, gm, wm` | GM brighter than WM (scale-free) |
| `3.3.negative_gm` | B | `cbf, gm` | fraction of negative GM voxels |
| `3.4.deep_gm_ratio` | B | `cbf, deep_gm, cortical_gm` | **neonatal only** — deep GM should exceed cortical GM |
| `4.1.coregistration` | B | `asl_mask, struct_mask` | Dice overlap, ASL vs T1 brain mask |
| `4.2.coverage` | B | `cbf, gm`, optional `wm` | how much of the tissue ROI the ASL actually imaged |
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

## 4. Configuration — every threshold lives in one place, with its provenance

```python
from osipy_qc import run_qc, QCConfig

cfg = QCConfig(
    qei_pass=0.60,        # raise the QEI PASS bar from the default 0.55
    gm_cbf_lo=35.0,       # widen the GM CBF PASS band
    scov_artifact=1.10,   # loosen the artifactual sCoV cutoff
    strict=False,         # demote UNCALIBRATED FAILs to WARN (clinical cohorts)
)
report = run_qc(inputs, cfg=cfg)
```

### Population profiles — CBF norms move across the lifespan
```python
from osipy_qc.core.config import for_population

cfg = for_population("neonate")   # v1.0 ships "adult" (default) and "neonate"
```
A newborn's normal GM CBF (~16) is far below the adult 40–100 band, so grading a
neonate against adult bands would fail every scan. `for_population()` **raises**
on an unknown name rather than silently defaulting. Other age groups are planned
once their bands are calibrated with the mentors (see the population table in [THRESHOLD_PROVENANCE.md](THRESHOLD_PROVENANCE.md)).

### Provenance — "how did you get this number?"
```python
from osipy_qc.core.config import provenance_of, uncalibrated_fields

provenance_of("gm_cbf_lo")   # (PUBLISHED, 'Alsop 2015 ... doi:10.1002/mrm.25197', 'verbatim: "40-100 ..."')
uncalibrated_fields()        # the 44 numbers we cannot cite
```
Every threshold is tagged **published** (a paper states it), **implementation**
(reference code uses it), or **uncalibrated** (our engineering default).
A FAIL decided by an uncalibrated cut-off is marked **provisional**, and
`--no-strict` demotes every provisional FAIL to a WARN. Strict is the default, so an
uncalibrated number *can* reach a FAIL — it simply cannot do so silently. Run
`osipy-qc --provenance` for the full dump, or see
[THRESHOLD_PROVENANCE.md](THRESHOLD_PROVENANCE.md).

### Organ profiles
```python
from osipy_qc.core.config import for_organ, skipped_for_organ

skipped_for_organ("kidney")   # ('1.qei', '3.2.gm_wm_ratio', '3.4.deep_gm_ratio')
report = run_qc(inputs, cfg=for_organ("kidney"),
                checks=[c for c in all_checks() if c not in skipped_for_organ("kidney")])
```
QEI's spatial template (`2.5·GM + 1·WM`) is a **brain** tissue model, so it is
explicitly skipped for other organs rather than silently producing a number.

### Quantification parameters
Not thresholds — acquisition facts the QC layer must be *told*, because they are
absent from NIfTI headers. They default to `None` (unknown), never to a guess:
```python
cfg = QCConfig(labeling_efficiency=0.85, label_duration_s=1.8,
               post_labeling_delay_s=2.0, t1_blood_s=1.65)
```

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

`load_folder(folder, load_arrays=True) -> dict` is what the CLI calls internally
when you run `osipy-qc <folder>`. It recurses, and it reads **both** streams: raw
acquisitions by filename (ASL / M0 / T1) and pipeline output as well — a
quantified CBF map and GM/WM/CSF maps already in ASL space. So for a folder that
holds a subject's derivatives plus the raw series, `osipy-qc <folder>` needs no
adapter at all; the adapters above are for pointing at one specific directory.

What it recognises as pipeline output is `classify_derivative(name, shape=None)`
in `checks/schema.py` — `*cbf*`/`*perfusion*`/`*rbf*` for the map, and
`*GM*`/`*WM*`/`*CSF*` with a `probseg`/`pv` style name for the tissue maps,
3-D images only. That one function is also what the adapters and the upload page
apply, so the three cannot drift apart.

Three behaviours worth knowing:

* The **M0 TR** is read from the NIfTI header when no sidecar states one — but
  only from a 4-D image, only when the header declares its time unit, and it is
  reported as `[NIfTI header (pixdim[4])]` so you can see where it came from.
  `pixdim[4]` on a 3-D image is leftover data, not a repetition time.
* Where **several CBF maps** sit in one folder, a calibrated map beats an
  uncalibrated one (oxford_asl's `perfusion_calib` over `perfusion`) and an
  unqualified name beats a `desc-` variant (ASLPrep's `_cbf` over
  `_desc-score_cbf`). `8.2.data_type` names the one that was graded.
* If the tissue maps turn out to be on a **different grid** from the CBF map,
  Stream B is skipped instead of raising, and `8.2.data_type` says why. Stream A
  still grades — one bad resample should not cost you the whole report.

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

---

## QEI-Net, the optional deep-learning quality index

`1.1.qei_net` scores a CBF map with the deep-learning QEI of Beltran Urbano et al.
**The model is not part of this package.** Its weights are unpublished and are not
ours to redistribute, so the check shells out to the author's own inference script
in a separate environment and reads one number back.

Without it configured the check reports N/A and nothing else changes. It is marked
N/A rather than UNKNOWN on purpose: UNKNOWN means the *data* was missing something
and dents the coverage figure, whereas a model you chose not to install says nothing
about the scan in front of you.

### Setting it up

You need the inference package and its weights from the model's authors; they are
not distributed here. Unpack it so `src/` and `weights/` sit side by side:

```
qei_inference_package/
  requirements.txt
  src/     run_qei.py, network.py, preprocess.py
  weights/ fold0/ fold1/ fold2/ fold3/ fold4/   each with best_model.pth
```

All five folds are expected by default — the score is their ensemble mean. Fewer
is possible with `--folds`, but then it is a different number and should not be
compared with a five-fold one.

**1. Give it its own environment.** The model needs torch, torchio and SimpleITK.
This package deliberately depends on none of them, and that separation is the
point: `osipy_qc` stays pure numpy + nibabel, and torch lives somewhere else.

```bash
python3 -m venv ~/qei_env
~/qei_env/bin/pip install -r /path/to/qei_inference_package/requirements.txt
```

If those pins do not resolve on your Python — they are exact, and `torch==2.14.0`
in particular may not have a build for your platform — install what the code
actually imports instead, which is five packages:

```bash
~/qei_env/bin/pip install torch torchio nibabel numpy pandas
```

`torchio` brings SimpleITK and scipy with it. Verified working on Python 3.14
with torch 2.12.0, which is not what the pin asks for.

**2. Check the model runs on its own**, before involving this package. It prints
one number to stdout and nothing else:

```bash
~/qei_env/bin/python /path/to/qei_inference_package/src/run_qei.py \
    --single /path/to/cbf.nii.gz --mask /path/to/brainmask.nii.gz
```

Doing this first means a failure is attributable. If it fails here it is the
model's environment; if it works here and not through `osipy-qc`, it is the
wiring below.

**3. Wire it in.** Two environment variables, and nothing else changes:

```bash
export OSIPY_QEI_NET_PYTHON=~/qei_env/bin/python
export OSIPY_QEI_NET_SCRIPT=/path/to/qei_inference_package/src/run_qei.py

osipy-qc /path/to/subject_folder
```

`1.1.qei_net` should now read `QEI-Net 0.xxx [model <fingerprint>]` instead of
"not configured". If it still says not configured, the variables are not reaching
the process; if it says "is not at the path given", one of the two paths is wrong.

The same two variables work for the web console — set them before `--serve` and
uploads are scored with the model too.

Paths come from the environment rather than from a config file so that a path to
somebody else's unpublished model never lands in a committed file. `.gitignore`
refuses `*.pth`, `*.pt`, `*.ckpt`, `weights/` and `qei_inference_package/`, and
`test_the_weights_are_ignored_by_git` shells out to `git check-ignore` to prove
those rules still bite.

### What it reports

The score is reported as INFO and never decides a verdict, because no validated
cut-off for this model has been published. Reporting a number is honest; inventing
a line to grade it against would not be.

Each result records a fingerprint of the weights that produced it, so a score can
always be traced to the model version behind it.

### Why it refuses some maps

The model normalises with `clip(cbf, -100, 100) / 100`, which is right for a
correctly quantified map and destructive for a mis-scaled one. On a real GE map
whose calibration was about fifty times too high, 74% of the voxels inside the
brain mask pinned to +1.0, only two distinct values survived, and the network still
returned 0.634 while the classical QEI scored the same map 0.0006.

So the check measures the saturated fraction first and reports UNKNOWN instead of a
score when more than 30% of the brain sits at the clip bound, pointing at
`3.1.cbf_level`. A number computed from a flattened volume is not a quality
measurement.

Run both indices rather than choosing between them. They fail differently, which is
the point.
