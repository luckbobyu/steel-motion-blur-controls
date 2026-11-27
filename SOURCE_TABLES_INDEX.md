# Source-table and protocol index

V19.2 changes reporting, not numerical inputs or training runs.
The paper labels supporting result collections as Source Tables S1-S3. These
are logical collections, not three extra spreadsheets required by the replay.
Run `python scripts/reproduce.py` to write the corresponding output CSVs.

| Paper collection | Repository input | Replay output in outputs/tables |
| --- | --- | --- |
| S1, core mean-strength effects and absolute recall | data/exported/core_by_seed.csv; core_absolute_by_seed.csv | table2_strength.csv (absolute seed-level input remains in core_absolute_by_seed.csv) |
| S2, distribution endpoints, FP/view changes and exported CIs | data/exported/distribution_by_seed.csv; distribution_summary.csv | table3_all_distribution_endpoints.csv |
| S3, frozen-budget pooled and per-path results | data/operating_points/dm_paths.csv; yolov8n_paths.csv | table4_all_budgets_and_maxima.csv; v17_cross_budget.csv; v17_cross_budget_by_seed.csv |

Check the entry's actual output list if a historical filename differs. Exported
image-cluster confidence intervals cannot be reexecuted from summary CSVs alone.

## Exact specifications and evaluation rules

- `reference/kernel_reference.py` preserves the historical `sample_slot` hash
  rule for image/epoch exposure selection, direction and original recipe slot.
  The SHA-256 namespace is `neu-blur-v1:{epoch}:{basename}`, including when the
  project applies this shared rule to GC10 data. Epochs run from 1 to 30.
- `configs/yolo11_*_protocol.json` records per-direction target moments and
  reference exposure counts. Single level means one solved strength within
  each direction, with direction-specific rasterization matching.
- `scripts/generate_kernels.py` uses 13 line samples at positions -6 through 6,
  solves spacing to match realized moments and checks a 2e-5 px^2 tolerance.
  It also regenerates the 12 new paths at generator seed 20261122.
- Measured trajectories are `walk01` through `walk12` from the separately
  checksum-verified official `500Frames.zip`. See kernel reference projection
  code and `--trajectories` instructions; no real production images are implied.
- Cutoff locks and `configs/headroom_protocol.json` record the evaluated
  validation budgets. The headroom comparisons are retrospective benchmark
  diagnostics; new paths reuse the test originals and do not constitute an
  independently sampled new image test set.
- Path exceedance uses unrounded FP/view > 0.25. Pooled pass uses <= 0.25 per
  seed. Max FP is the maximum over all 3 seeds x 12 paths in that table row.
- Native DetectionValidator in each pinned detector runtime supplies the AP
  metrics from confidence-floor candidates before the separate frozen cutoff.
  `scripts/evaluate_predictions.py` recounts IoU-0.50 TP/FP only and does not
  implement AP, NMS, model training or bootstrap reexecution.

## Scope

This index documents existing records and code. Full detector runtimes,
weights, training workflows and per-original prediction caches remain external.
The repository supports result reconstruction and kernel checks, with limited
fixed-checkpoint lab verification. It does not claim independent full retraining.
