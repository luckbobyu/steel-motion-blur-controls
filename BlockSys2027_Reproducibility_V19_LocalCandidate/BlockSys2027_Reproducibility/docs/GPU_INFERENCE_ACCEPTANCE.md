# Limited fixed-checkpoint GPU acceptance

This entry uses the existing lab project, not a standalone freshly installed
training repository. It never calls a training queue. The YOLO training method
is disabled at runtime. It does not fit or change thresholds.

For one chosen dataset, it loads the existing DM seed-2 low/high continuation
weights and reruns all 180 NEU or 230 GC10 validation images in four linear27
directions. This gives eight inference passes and six frozen-cutoff count checks
per dataset (two arms, three validation budgets). Source hashes, weight hashes,
reference job/prediction hashes and target geometry are checked first.

Launch NEU on physical GPU0 and GC10 on GPU1 in separate PowerShell windows:

```powershell
& 'E:\anaconda3\envs\dm_yolo_repro\python.exe' '.\scripts\verify_checkpoint_inference.py' --project-root 'E:\xxy\dm_yolo_repro' --dataset neu --gpu 0
& 'E:\anaconda3\envs\dm_yolo_repro\python.exe' '.\scripts\verify_checkpoint_inference.py' --project-root 'E:\xxy\dm_yolo_repro' --dataset gc10 --gpu 1
```

Use the packaged directory as the working directory. Outputs go to newly named
project `records/repro_inference_*` directories; return each feedback ZIP. Keep
the ZIPs private. `NEEDS_REVIEW` or `ERROR` must be inspected, not fixed by changing
historical counts. Ordered coordinate/score comparisons use atol 1e-4 and rtol
1e-5 as diagnostics; PASS requires exact TP/FP/GT/view counts at all frozen
budgets and matching GT geometry. This checks validation inference only, not the
entire test set, all model families, bootstrap or training reproducibility.

Both CUDA runs passed on the original lab server. NEU completed in 43.31 seconds
and GC10 in 63.65 seconds, including staging and checks. All 12 endpoints match,
all 16 prediction diagnostics are within tolerance, and every endpoint was also
recounted locally from returned inference prediction JSON. See
`SERVER_GPU_ACCEPTANCE.json`. These timings are acceptance-run observations,
not detector-only FPS benchmarks or fresh-install validation.
