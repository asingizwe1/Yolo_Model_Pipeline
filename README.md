# Yolo_Model_Pipeline

Reproducible Ultralytics YOLO detection training for honey bee comb contents.
The dataset is the Roboflow project BeeComb: seven classes, exported in YOLO
format with a train/valid/test split made by source video. That split is part
of the data. This repository does not build another one.

Stage 1 is the locked recipe and the notes for it. There is no `train.py` yet.

| File | Role |
| --- | --- |
| `STRUCTURE.md` | Full tree, including scripts that later stages add. |
| `configs/base.yaml` | Settings that stay identical across the three models. |
| `configs/models.yaml` | The editable list: `yolov8s.pt`, `yolo11s.pt`, `yolov8m.pt`. |
| `configs/data.yaml` | Class names and paths. |
| `docs/STAGE1_NOTES.md` | Defaults, what was changed, VRAM, reproducibility, Windows. |

The recipe was read from **ultralytics 8.4.193** (`requirements.txt`). Do not
upgrade that package between the three runs.

Training hardware is a Windows machine with one NVIDIA GPU (`device: 0`).
The GPU model, VRAM, driver, CUDA version, and Python version are still
**FILL** placeholders in `configs/base.yaml` and `docs/STAGE1_NOTES.md`.
`batch: 16` is the library default, not a batch measured on that GPU. Replace
it with the AutoBatch integer from `yolov8m.pt` before any comparison run.
The procedure is in the notes.

Install, from a virtual environment at the repository root:

```bash
pip install -r requirements.txt
```

On Windows, install a CUDA build of PyTorch that matches the driver first if
`pip` would otherwise pull a CPU wheel. Confirm the GPU with the later
`scripts/env_report.py` rather than with this Linux smoke-test machine.
