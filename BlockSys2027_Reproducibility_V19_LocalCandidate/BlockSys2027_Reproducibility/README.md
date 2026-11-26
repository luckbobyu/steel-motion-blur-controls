# Mean Blur Strength and Strength Distribution in Steel Defect Detection

Local research-artifact candidate aligned to the supplied **English Draft V19**
(26 November 2026). This directory has not been published to GitHub.

## What this candidate reproduces

The standard-library entry reconstructs the numerical content of Tables 2-5,
the scan values behind Figure 2, recorded mean moments for Table 1, and the
V19 high@0.15 versus low@0.20 operating-point comparison from archived exports.
It checks 36 paired core endpoints, 100 distribution summaries, 1,728 path-count
records and 54 adjacent scan comparisons. It also exports absolute performance
and seed SDs so that they are distinct from paired-gain SDs.

The public candidate primarily supports metric/count replay. A limited
fixed-checkpoint inference check has also passed in the original lab environment:
DM seed 2, low/high arms, both datasets and four linear27 validation directions.
All 12 frozen-cutoff endpoints match, and the returned predictions were recounted
locally. This does **not** establish a fresh-install reproduction, all-model/test-set
inference, independent training or image-cluster bootstrap reexecution.
See `ARTIFACT_STATUS.json` for the current boundary.

## Quick start

Python 3.9 or newer, no third-party packages for table reconstruction.

```bash
python scripts/reproduce.py
```

The command first verifies the candidate file hashes. Results are written to
`outputs/tables/`, including `CHECKS.json`, all budget
endpoints and `v17_cross_budget.csv`. The latter filename preserves its historical
analysis name; its contents also support V19 Section 4.4.

Optional kernel-construction checks use NumPy 2.3.5 and Python 3.11 or newer.
This is the locally checked audit environment, not the historical GPU training environment.

```bash
python -m pip install -r requirements-kernels.txt
python scripts/generate_kernels.py
```

For the measured-trajectory check, additionally supply the original official
`500Frames.zip` using `--trajectories /path/to/500Frames.zip`. It is not bundled
and is not downloaded implicitly. Archive SHA-256 is checked before use.

## Structure

- `data/exported/` contains reported seed-level results and summary references.
- `data/operating_points/` contains per-seed per-path TP, FP and denominators.
- `data/scans/` contains seed-level scan exports.
- `configs/` contains recorded protocols and frozen validation cutoffs.
- `reference/` contains small study-kernel functions and original source hashes.
- `scripts/` contains table replay, kernel checks and a cache-count evaluator.
- `docs/MANUSCRIPT_MAPPING.md` maps V19 sections and tables to these inputs.

## Next server step and GitHub publication

The standalone read-only collector and its manifest are packaged separately for
easy transfer. Instructions are in `docs/SERVER_COLLECTION.md`. The completed first pass verified all 92 referenced assets (68 checkpoints and
24 prediction caches), indexed 8,196 dataset files, and matched the five original
kernel-source fingerprints. See `docs/SERVER_ASSET_AUDIT.json`. This establishes
file integrity, not independent training or inference reproduction.

The read-only CPU check is `scripts/verify_server_predictions.py`. It compares
the original DM seed-2 **linear27** validation caches with 12 frozen-cutoff
endpoints. The corrected server check pins 16 exact linear27 job reports and passed
all 12 endpoints exactly (TP, FP, GT, views, recall and FP/view). The first check
selected obsolete linear23 caches; that acceptance-script error was corrected
without changing historical results or frozen cutoffs.
See `docs/CPU_ACCEPTANCE_STATUS.json` and `docs/SERVER_CPU_ACCEPTANCE.json`. Three detector/helper source directories
(622 files) have been privately collected and their returned hashes checked;
these third-party runtimes are not automatically included in the public candidate.

`docs/GITHUB_RELEASE.md` describes the remaining account, license and publication
steps. `.github/workflows/verify.yml` is prepared for Windows/Linux count tests
and a kernel audit after authorized GitHub upload; no remote Actions run has
occurred yet.

## Fixed-checkpoint acceptance

`docs/SERVER_GPU_ACCEPTANCE.json` records the two completed server runs: 16
model/condition passes and 3,280 image-inference views. No ordered box/score
differences exceeded the diagnostic tolerances (atol 1e-4, rtol 1e-5); all TP, FP,
GT and view counts matched exactly at the three frozen validation budgets.
`scripts/verify_checkpoint_inference.py` requires the pinned original project;
see `docs/GPU_INFERENCE_ACCEPTANCE.md`. Checkpoints and detector runtimes are
not automatically redistributed by this metric-replay candidate.

## Scope and release status

Validation cutoffs are kept frozen. The cross-budget analysis compares different
validation budgets; it is not a test-refitted iso-FP curve or a prospective risk
guarantee. FP/view denotes extra predictions on defect-bearing benchmark views.
Three seeds and reused test views are not independent industrial deployments.

No model checkpoints, raw dataset images, unrelated draft manuscripts or peer
review reports are included. Source detector code follows its upstream terms and
is not vendored here. An author-approved license, authorship metadata, stable
public URL and finalized clean-environment checks are still pending. Do not
describe this local candidate as a completed public release.
