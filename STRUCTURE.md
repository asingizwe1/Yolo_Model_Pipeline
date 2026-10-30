# Folder tree

Stage 1 (this commit) writes the recipe, the notes, and the empty folders.
Python training scripts are listed so the later stages have a fixed place to
go. They are not in the repository yet.

```
Yolo_Model_Pipeline/
├── README.md                          # What this repo is, Stage 1 status, and where to read next.
├── STRUCTURE.md                       # This file. One purpose line per planned file.
├── requirements.txt                   # Pins ultralytics==8.4.193. Torch is a dependency, not a Windows CUDA pin.
├── .gitignore                         # Ignores data payloads, runs/, results/, weights, .env, caches.
├── configs/
│   ├── base.yaml                      # Shared hyperparameters for all three models. Inline default and Why comments.
│   ├── models.yaml                    # Editable list of the three detection checkpoints.
│   └── data.yaml                      # Seven class names and the Roboflow folder layout. Relative path, resolved later.
├── docs/
│   └── STAGE1_NOTES.md                # Supervisor notes: defaults, small-object choices, VRAM, reproducibility, Windows.
├── scripts/
│   ├── .gitkeep                       # Keeps scripts/ in git until the Stage 2 scripts exist.
│   ├── env_report.py                  # Later. Prints Python, torch, CUDA, GPU name, and VRAM. Fills the FILL blanks.
│   ├── download_data.py               # Later. Copies the Roboflow YOLO export into data/ without building a new split.
│   ├── check_data.py                  # Later. Counts images, boxes, and class ids per existing split. Does not reshuffle.
│   └── compare_models.py              # Later. Trains each models.yaml entry with the same base.yaml and resolved data.yaml.
├── train.py                           # Later. One model, one call, `if __name__ == "__main__"` guard for Windows spawn.
├── tune.py                            # Later. Hyperparameter search. Not part of the locked three-model comparison.
├── evaluate.py                        # Later. Validation metrics on the existing val split, then the held-out test split.
├── robustness.py                      # Later. Controlled input perturbations scored the same way for every model.
├── predict.py                         # Later. Runs a trained checkpoint on frames or a folder.
├── export.py                          # Later. Exports a chosen checkpoint. Not used to pick the winner.
├── explain.py                         # Later. Qualitative failure examples for the paper. Not a second test set.
├── data/
│   ├── train/images/.gitkeep          # Roboflow train images land here. Payload is gitignored.
│   ├── train/labels/.gitkeep          # Matching YOLO txt labels. The folder must be named labels, sibling of images.
│   ├── valid/images/.gitkeep          # Roboflow validation images. The yaml key that points here is val, not valid.
│   ├── valid/labels/.gitkeep
│   ├── test/images/.gitkeep           # Held-out test images. Not read by training.
│   └── test/labels/.gitkeep
├── runs/
│   └── .gitkeep                       # Ultralytics writes runs/detect/<name>/ here. Checkpoints are gitignored.
└── results/
    └── .gitkeep                       # Later. Tables and figures copied out of runs/ for the paper.
```

Rules that apply to every later script:

- The train/valid/test split is already defined by source video. No script may call a split or shuffle function on it.
- `configs/base.yaml` is the only hyperparameter source for the three-model comparison. `tune.py` does not edit it in place.
- `batch` in `base.yaml` is one integer, chosen from `yolov8m.pt`, then used for all three models.
- Dataset paths are resolved from the repository root to an absolute path before Ultralytics reads them. The contract is in `configs/data.yaml`.
