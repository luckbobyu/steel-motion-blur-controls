# Source and license provenance

Primary exported metrics originate from the author's archived study runs on
NEU-DET and GC10-DET. Manifest hashes identify the concrete files in this local
candidate. Publication and redistribution approval remain separate from local
verification. No public URL has been fabricated.

`reference/kernel_reference.py` contains small functions extracted unchanged
from the author's historical experiment scripts. `reference/source_hashes.json`
lists those source filenames and SHA-256 fingerprints. The surrounding utilities
make the numerical intervention inspectable without loading a detector runtime.

The upstream baseline is https://github.com/ssjddb/DM-YOLO . YOLO11n experiments
used an isolated official Ultralytics 8.3.0 runtime. These upstream detector
implementations and their licenses are not redistributed by this candidate.
Do not apply a new permissive license blanket to third-party detector code or
datasets. An author-approved license for the material to be released is pending.

The original measured-motion source is
https://webdav.tuebingen.mpg.de/pixel/benchmark4camerashake/src_files/Trajectories6D/500Frames.zip
with SHA-256
`6c8676e1fd63809d42df05b0ec8733d56180adb26795c96427802f6ea963cf14`.
It is not bundled. The optional generator checks the supplied archive checksum.

Dataset raw images are not bundled. The server inventory step records relative
filenames and hashes needed to identify the exact historical splits. Repository
metadata and collected lab records must be reviewed before public release.

The declared Python 3.9 / PyTorch 2.1.0+cu118 environment in the paper describes
historical GPU runs. The local count-replay test runs without PyTorch. Optional
kernel auditing uses NumPy 2.3.5 with Python 3.11 or later; that is an audit
environment, not an instruction to replace the historical training environment.
