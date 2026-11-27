# Mean Blur Strength and Strength Distribution in Steel Defect Detection

Code, recorded protocols and numerical results for studying motion-blur training
in steel surface defect detection on NEU-DET and GC10-DET.

This repository provides table reconstruction, blur-kernel checks and tools for
evaluating archived predictions. It also includes reports from fixed-checkpoint
inference checks completed in the original lab environment. Numerical inputs
are aligned with manuscript V19 and its V19.1/V19.2 reporting revisions; see [the manuscript mapping](docs/MANUSCRIPT_MAPPING.md).

## What this repository reproduces

The main entry reconstructs Tables 2–5, the scan values behind Figure 2, recorded
mean kernel moments for Table 1, and the high@0.15 versus low@0.20 operating-point
comparison. It checks 36 paired core endpoints, 100 distribution summaries,
1,728 path-count records and 54 adjacent scan comparisons. Absolute performance
SDs and paired-gain SDs are reported separately.

## Quick start

Python 3.9 or newer; table reconstruction needs no third-party packages.

```bash
python scripts/reproduce.py
```

The command verifies the input file hashes and writes results to `outputs/tables/`,
including `CHECKS.json`. The output filename `v17_cross_budget.csv` retains its
historical name; its contents also support V19 Section 4.4.

Optional kernel checks use Python 3.11 or newer and NumPy 2.3.5:

```bash
python -m pip install -r requirements-kernels.txt
python scripts/generate_kernels.py
```

This is the locally checked kernel-audit environment. The historical GPU
environment is recorded separately in `configs/historical_environment.json`.
For measured-trajectory checks, supply the original `500Frames.zip` with
`--trajectories /path/to/500Frames.zip`. Its SHA-256 is checked before use;
the archive is not bundled or downloaded automatically.

## Verification results

- **File integrity:** all 92 referenced assets matched their recorded SHA-256
  hashes; 8,196 dataset files were indexed. See `docs/SERVER_ASSET_AUDIT.json`.
- **Archived prediction counts:** 12 frozen-cutoff validation endpoints matched
  exactly on the lab server. See `docs/SERVER_CPU_ACCEPTANCE.json`.
- **Fixed-checkpoint inference:** DM-YOLO seed 2, low/high arms, both datasets
  and four linear27 validation directions were rerun. All 12 count endpoints
  matched exactly. No ordered box/score differences exceeded atol 1e-4 and
  rtol 1e-5 across 3,280 image-inference views. Returned predictions were also
  recounted locally. See `docs/SERVER_GPU_ACCEPTANCE.json`.
- **Local checks:** table replay, kernel checks and unit tests passed in a clean
  directory. See `LOCAL_VALIDATION.json`. The remote verification workflow passed on 27 November 2026
  at commit `d372565` ([run](https://github.com/luckbobyu/steel-motion-blur-controls/actions/runs/45866367719)).

The inference entry requires the pinned original lab project, datasets and
checkpoints. Instructions are in `docs/GPU_INFERENCE_ACCEPTANCE.md`. These checks
cover the stated validation conditions; they do not establish fresh-install
training, all-model/test-set inference or bootstrap reexecution.

## Repository structure

- `data/`: seed-level results, summary references and per-path counts.
- `configs/`: recorded protocols, frozen cutoffs and data/source fingerprints.
- `reference/`: study-kernel functions and their source fingerprints.
- `scripts/`: numerical reconstruction and acceptance-check tools.
- `tests/`: count rules and acceptance-check regression tests.
- `docs/`: manuscript mapping, provenance, verification reports and instructions.

## Evaluation scope and availability

Validation cutoffs remain frozen during test evaluation. The cross-budget result
compares different validation budgets; it is not a test-refitted iso-FP curve.
FP/view counts extra predictions on defect-bearing benchmark views, rather than
false alarms measured on a defect-free production stream.

The repository was made public on 27 November 2026. Start from this root
folder; the older nested `BlockSys2027_Reproducibility_V19_LocalCandidate/`
folder is a retained historical upload, not a second required installation.
See [source-table and protocol index](SOURCE_TABLES_INDEX.md) for the mapping
between the paper's Source Tables S1-S3 and the CSVs and replay outputs.

Model checkpoints, raw dataset images and third-party detector runtimes are not
bundled. Upstream attribution is recorded in `docs/PROVENANCE.md`. No new license
grant is inferred from public visibility; license selection and formal author
metadata remain pending. Older release-preparation notes in `docs/` describe
pre-release status. This README and `ARTIFACT_STATUS.json` describe current
availability. Numerical inputs and validated scope have not changed.
