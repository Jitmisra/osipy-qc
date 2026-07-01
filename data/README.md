# `data/` — put your own scans here

This folder is **empty on purpose**. Drop your own files here to grade them.
Nothing you put in this folder is committed to git (only this README is).

## To grade a CBF map

Put three files here (any filenames — these are just a suggestion):

```
data/
  cbf.nii.gz    <- the quantified CBF map (mL/100g/min), in ASL voxel space
  gm.nii.gz     <- grey-matter probability / partial-volume map, SAME voxel space
  wm.nii.gz     <- white-matter probability / partial-volume map, SAME voxel space
  csf.nii.gz    <- optional; if omitted it is derived as 1 - gm - wm
```

Then, from the repo root:

```bash
python examples/grade_cbf_map.py data/cbf.nii.gz --gm data/gm.nii.gz --wm data/wm.nii.gz
```

> The GM/WM maps **must be on the same voxel grid as the CBF map** (ASL space).
> If you only have a CBF map and no tissue maps, the command still runs with a
> rough whole-brain mask (level + noise checks only). To get the full Stream B
> (incl. QEI) you need the tissue maps — produce them with a pipeline
> (`oxford_asl -s <T1>`, or ASLPrep). See the main README for details.

## To QC a folder of raw ASL NIfTIs (Stream A)

Drop the raw `.nii.gz` files (control/label series, M0, T1) into a subfolder and:

```bash
osipy-qc data/my_raw_scan/
```
