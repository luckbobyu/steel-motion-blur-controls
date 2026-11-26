# Server input collection

Run the standalone collector with the supplied manifest in the existing lab
environment. No GPU is reserved and no training or inference is started.

Copy the server-request ZIP to the server and extract it beside the project.
Keep the Python script and manifest together, then run this PowerShell command
from its extracted directory:

```powershell
& 'E:\anaconda3\envs\dm_yolo_repro\python.exe' '.\collect_server_assets.py' --project-root 'E:\xxy\dm_yolo_repro' --manifest '.\server_assets.json'
```

The collector verifies the assets referenced by the currently available reports,
records missing files and hash mismatches, collects the import closure of the
study helpers and the custom detector source, records installed package versions,
and indexes dataset files without copying the raw steel images. The output ZIP
is written under a new `records/repro_assets_*` directory. Return that ZIP for
private inspection; it is **not** intended for direct public upload.

By default large checkpoints and prediction caches are hashed but not copied.
This first pass establishes which original assets are actually available and
which script/runtime versions produced them. If needed after inspection, rerun
with `--include-predictions` to include matching archived prediction JSON, or
`--include-weights` to include matching checkpoints. These can make the ZIP large.
No arbitrary credentials or full home-directory contents are collected.

`NEEDS_REVIEW` means some items are missing, changed or inaccessible; return the
report anyway. Do not rename substitute checkpoints to bypass hashes and do not
rerun experiments merely to make discovery report success.

This is an asset/integrity check. Actual fixed-weight evaluation and bootstrap
reexecution need a later, explicit run once the exact input/runtime closure has
been checked. It does not establish from-scratch training reproducibility.
