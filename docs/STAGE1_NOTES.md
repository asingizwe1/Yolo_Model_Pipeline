# Stage 1 notes

These notes are for the methods section and for the supervisor. They describe the locked training recipe in `configs/base.yaml`. They do not report BeeComb accuracy. No BeeComb images were trained on.

## 1. Version and sources

| Item | Value |
| --- | --- |
| ultralytics | **8.4.193** (`pip install -U ultralytics`, then pinned in `requirements.txt`) |
| Where the defaults were read | Installed `ultralytics/cfg/default.yaml`, plus the trainer, loss, augmentation, and AutoBatch modules in that same install |
| Argument reference | https://docs.ultralytics.com/usage/cfg/ |
| Smoke machine | Linux, Python 3.12.3, torch 2.15.0+cu132, CPU. This is not the training machine. |

The training machine is the user's Windows PC with one NVIDIA GPU. These are still blank and must be filled before the comparison is cited:

- GPU model: **FILL**
- Total VRAM: **FILL**
- NVIDIA driver: **FILL**
- CUDA version reported by PyTorch: **FILL**
- Windows version: **FILL**
- Python version: **FILL**

Do not copy the Linux smoke versions into the paper as the training environment.

## 2. What is locked

`configs/base.yaml` is the only hyperparameter source for `yolov8s.pt`, `yolo11s.pt`, and `yolov8m.pt` (`configs/models.yaml`). A later script may pass `model`, `data`, and `name`. It must not change a recipe key for only one of the three.

Keys absent from `base.yaml` stay at the 8.4.193 default. Two of those must stay absent:

- `time`. If it is set, it replaces `epochs`.
- `conf`. If it is left unset, validation sets it to 0.001 for detection (`ultralytics/engine/validator.py`: `self.args.conf = 0.001` when `conf is None` and the task is not obb). Setting `conf` would change the metric.

`fraction` is pinned to 1.0 inside the file so the Roboflow train split cannot be subsampled. The split was made by source video. Nothing in this repository may rebuild it.

## 3. Defaults, and the six changes

`get_cfg(DEFAULT_CFG_DICT, base.yaml)` accepted all 60 keys (section 10). Fifty-four values are the 8.4.193 default. Six are not.

### Changed

| Key | Default | Recipe | Why |
| --- | --- | --- | --- |
| `optimizer` | `auto` | `SGD` | `auto` does not train with the `lr0` written in the file. |
| `patience` | `100` | `0` | `0` disables early stopping, so every model runs the full `epochs`. |
| `device` | null (auto) | `0` | The single NVIDIA GPU. Auto-select can move. |
| `workers` | `8` | `2` | Windows spawn, and worker count is part of the augmentation RNG. |
| `channels_last` | null | `False` | Null turns channels-last on for CUDA outside Windows. Pinning False matches the Windows machine. |
| `flipud` | `0.0` | `0.5` | A comb frame has no reliable image "up". Class identity does not depend on it. |

### `optimizer=auto` (verified)

`BaseTrainer.build_optimizer` in `ultralytics/engine/trainer.py` does this when the name is `auto`:

- It logs that it is ignoring `lr0`.
- It sets `lr_fit = round(0.002 * 5 / (4 + nc), 6)`.
- If the estimated step count is greater than 10000, it selects **MuSGD**, `lr=0.01`, `momentum=0.9`. Otherwise it selects **AdamW**, `lr=lr_fit`, `momentum=0.9`.
- It writes those choices back onto `args.optimizer` and `args.lr0`, and sets `args.warmup_bias_lr = 0.0`.

The step count is `ceil(len(dataset) / max(batch, nbs)) * epochs`. It does not depend on which of the three architectures is loaded, so `auto` would usually pick the same optimizer for all three. It would still make the yaml lie: the printed `lr0` and `momentum` would not be the values that trained, and raising `epochs` across the 10000-step boundary would switch optimizer silently.

SGD is the optimizer the default `lr0: 0.01` comment was written for (`SGD=1e-2`, `Adam/AdamW=1e-3`). With `optimizer: SGD` the trainer builds SGD with `nesterov=True` and uses `lr0` and `momentum` as written. The smoke log confirmed `SGD(lr=0.01, momentum=0.937)`.

MuSGD is what `auto` would pick for a long run. It also multiplies the learning rate by 3 on some head parameters. It is not used here. Switching the comparison to MuSGD later is allowed only if all three models switch together and the notes are updated. It is an open question, not a silent default.

### `patience=0` (verified)

`EarlyStopping.__init__` sets `self.patience = patience or float("inf")`. Zero is falsy, so early stopping is off. The log text in that class documents `patience=0` as the disable switch. `best.pt` is still the epoch with the best validation fitness.

With the default `patience=100` and `epochs=100`, early stopping would also fail to fire: the stopper is called with the 1-based epoch, and at epoch 100 the largest gap from a best epoch of 1 is 99, which is not `>= 100`. Zero is still the right pin. If `epochs` is raised later and patience stays 100, one model can stop while another keeps going.

### `device`, `workers`, `channels_last`

`device: 0` is parsed by `parse_device` into the string `"0"`. The CPU smoke overrode it and did not edit the file.

`workers: 2` is a ceiling. `build_dataloader` sets

```text
nw = min(cpu_count // device_count, workers, 0 if batches <= 1 else batches)
```

On CPU, `BaseTrainer` forces `workers=0` before the loader is built ("faster CPU training"). The smoke banner therefore printed `workers=2` and then `Using 0 dataloader workers`. GPU training keeps the ceiling of 2. The detection trainer asks for `workers * 2` on the validation loader, and the same cap is applied. All three models share the ceiling.

`channels_last is None` is turned on only when the device is CUDA and the OS is not Windows (`trainer.py`: `TORCH_1_11 and not WINDOWS`). The comment in that function says it measured slower on Windows. `False` stops a Linux rerun from taking a different memory format.

### `flipud=0.5`

`v8_transforms` adds `RandomFlip(direction="vertical", p=hyp.flipud)`. The default 0 is the COCO choice: people and vehicles have a gravity direction. Honey, capped honey, bee bread, brood, eggs, and larvae do not change class when a frame is turned upside down. 0.5 is the same probability as the default horizontal flip. This is a property of the scene, not a measured mAP gain. If a class is later shown to be orientation-specific, set `flipud` back to 0.0 for all three models before any comparison run.

### Kept on purpose (small and dense objects)

These stay at the default. The reason is part of the method, not "we did not look."

| Key | Default, kept | Why it stays |
| --- | --- | --- |
| `imgsz` | 640 | Pretrained resolution, and VRAM is unknown. 1024 and 1280 are a later locked experiment, not a silent change. See section 4. |
| `scale` | 0.5 | A float is a gain. `RandomPerspective` draws the factor with `random.uniform(1 - scale, 1 + scale)`. For 0.5 that is the half-open range from 0.5 up to, but not including, 1.5. Do not increase it. A smaller gain keeps object scale closer to the original; that would be a separate three-model experiment after box sizes are known. |
| `mosaic` | 1.0 | 2x2 mosaic (`Mosaic` default `n=4`). `apply_image` builds a canvas of `2*imgsz` by `2*imgsz`. The following affine warps that canvas to `imgsz`, so objects are drawn from a four-image canvas and can be clipped. This is the library's small-object augmentation. |
| `close_mosaic` | 10 | For the last 10 epochs the loader sets mosaic, copy_paste, mixup, and cutmix to 0 and rebuilds. With `epochs=100` that is epochs 90 through 99. The last epochs see one image at `imgsz`. |
| `degrees` | 0.0 | Off. After the affine, `box_candidates` drops a box whose new width or height is `<= 2` pixels, or whose area is below 10% of the original when there are no segment polygons (`area_thr=0.10`; segment labels use 0.01). Eggs and larvae would be dropped first. |
| `shear`, `perspective` | 0.0 | Off. Both distort boxes. Perspective is documented as about 0 to 0.001 when it is used. |
| `translate` | 0.1 | Shift of plus or minus 10% of the image. Boxes that leave the frame are clipped. |
| `mixup`, `cutmix` | 0.0 | Off. MixUp blends images and is a poor fit for dense boxes. |
| `copy_paste` | 0.0 | Off, and a nonzero value would not paste box-only labels. See below. |
| `copy_paste_mode` | `flip` | Unused while `copy_paste` is 0. Legal values are `flip` and `mixup`. |
| `hsv_h`, `hsv_s`, `hsv_v` | 0.015, 0.7, 0.4 | Kept for the first comparison. Hue jitter is already small. Saturation 0.7 and value 0.4 are strong, and honey versus capped honey versus bee bread is partly a color difference. Lowering them is an open question for all three models together, not a number invented here. |
| `bgr` | 0.0 | Channel swap stays off. It would change those colors. |
| `fliplr` | 0.5 | Same orientation argument as `flipud`, and already the default. |
| `rect` | False | `YOLODataset.build_transforms` forces mosaic, mixup, and cutmix to 0 when `rect` is True. Training stays square. The detection trainer still builds the **validation** set with `rect=True` (`DetectionTrainer.build_dataset`: `rect=(mode == "val")`). That is library behavior, identical for all three models. |
| `multi_scale` | 0.0 | A fixed `imgsz` is what the batch probe measures. Multi-scale can also build a one-image batch at a tiny size, which the trainer rejects because BatchNorm would see one value per channel. |
| `cache` | False | RAM cache does not change labels, but the machine's RAM is unknown. |
| `single_cls` | False | Seven classes. True would rename everything to one class, `item`. |
| `fraction` | 1.0 | Every train image. Not a new split. |
| `cos_lr` | False | Linear decay from `lr0` to `lr0 * lrf` (0.01 to 0.0001). |
| `lr0`, `lrf`, `momentum` | 0.01, 0.01, 0.937 | Used, because the optimizer is SGD rather than `auto`. |
| `weight_decay` | 0.0005 | Weights only. BatchNorm and bias groups get decay 0. Scaled as in section 4. |
| `warmup_epochs`, `warmup_momentum`, `warmup_bias_lr` | 3.0, 0.8, 0.1 | Kept. `auto` would have set the bias warmup lr to 0. If the train split is very small, 3 warmup epochs is a large fraction of the run; change it for all models or for none. |
| `nbs` | 64 | Nominal batch size. Section 4. |
| `epochs` | 100 | Fixed budget. |
| `batch` | 16 | Kept only as a placeholder. Section 5. It is not a measured batch. |
| `amp` | True | Same flag for all three. Section 6 for why this is not bitwise reproducibility. |
| `compile` | False | `torch.compile` changes numerics and the memory probe. |
| `pretrained` | True | Each run starts from the checkpoint named in `models.yaml`. |
| `cls_remap` | True | No-op for these names. See below. |
| `box`, `cls`, `dfl` | 7.5, 0.5, 1.5 | Detection loss gains. |
| `cls_pw` | 0.0 | Disables inverse-frequency class weights (`DetectionTrainer.set_class_weights` returns immediately). Turn on only for all three models, and only after class counts exist. |
| `val` | True | Validation uses the dataset `val` key, which points at `valid/images`. The test split is not read during training. |
| `iou` | 0.7 | NMS IoU. Higher keeps more overlapping boxes of the same class. Adjacent cells usually overlap little. Do not change this from a guess. |
| `max_det` | 300 | NMS keeps at most this many boxes per image. A frame with more than 300 objects cannot score the rest. Raise it for all three models if `check_data.py` finds such frames. No count exists yet, so the default stays. |
| `agnostic_nms` | False | Per-class NMS, so a region of one class and cells of another class can both remain. |
| `save`, `save_period` | True, -1 | `last.pt` and `best.pt` only. Intermediate epoch checkpoints are off. |
| `exist_ok` | False | A repeated name increments the folder instead of overwriting. |
| `resume` | False | Comparison runs start from the pretrained checkpoint. |
| `plots`, `verbose` | True | Label plots and the metric csv are the paper trail. |
| `task` | detect | Box detection. |
| `seed`, `deterministic` | 0, True | Section 6. `deterministic=True` is already the 8.4.193 default. It was not switched on from False. |

### `copy_paste` and box labels (verified)

`CopyPaste.__call__` returns the labels unchanged when `len(instances.segments) == 0` or when `p == 0`. `default.yaml` describes the argument as "segment/semantic/obb copy-paste".

For a detection label file, a line longer than 6 numbers is parsed as a polygon (`data/utils.py`: "is segment"). A normal YOLO box line has 5 numbers: class, cx, cy, w, h. Those files produce an empty segment array (`YOLODataset.update_labels_info` builds `zeros((0, ...))` when `len(segments) == 0`). Copy-paste then does nothing even if the probability were raised.

`copy_paste: 0` also short-circuits polygon labels, so the recipe is safe either way. Do not set it above 0 expecting a gain on box labels. `check_data.py` should confirm the export is 5 columns. Paste augmentation is not available for this task unless the project is re-annotated with masks, which is out of scope and would be a different dataset.

### Pretrained head at `nc=7` (verified, not exercised by the smoke)

`BaseModel.load` rebuilds the last convolution of the classification branch when its output count differs from the checkpoint, then calls `bias_init()`. `_remap_cls_by_names` copies a pretrained class-head row only when the destination class name matches a source name (case-insensitive, whitespace stripped). It returns immediately when the number of matches is 0.

None of bee bread, honey, capped honey, worker capped brood, capped drone brood, eggs, or larvae is a COCO class name, so no row is copied. The new class-output convolution keeps its initialization. Matching tensors, including the backbone, still load. The same rule applies to all three checkpoints.

The smoke used coco8, which has 80 classes, and logged `Transferred 355/355 items`. That line does **not** describe the BeeComb head. It only shows that `pretrained: True` loads a checkpoint.

Classification-only keys (`dropout`, `auto_augment`, `erasing`) and segmentation-only keys (`overlap_mask`, `mask_ratio`) are omitted. They stay at the library default and are not part of this detection loss.

## 4. Image size, memory, and `nbs`

### 640 versus 1024 versus 1280

`imgsz` is the square side length. Stride for these detectors is 32, and 640, 1024, and 1280 are all multiples of 32.

Activation maps are stored on a grid proportional to height times width. Relative to 640:

- 1024 is `(1024/640)^2 = 2.56` times as many pixels.
- 1280 is `(1280/640)^2 = 4` times as many pixels.

The activation and convolution-workspace term therefore grows by about 4x from 640 to 1280. Parameter memory, gradient memory, and Adam/SGD optimizer state do **not** grow with `imgsz`; they grow with the model (yolov8m is the largest). Total VRAM is not exactly 4x. Nobody has measured that ratio on the FILL GPU, and this note does not claim a measured number.

A higher `imgsz` is the direct way to give eggs and larvae more than a handful of pixels. It is not free: the batch that fits gets smaller. A smaller micro-batch makes two things noisier, and `nbs` fixes only one of them.

1. **Gradients.** The optimizer steps every `accumulate` micro-batches, not every micro-batch. A smaller batch needs more accumulation to reach the same image count, and each micro-batch gradient is a noisier estimate. The summed step is closer to a large-batch step only in expectation.
2. **BatchNorm.** BatchNorm updates on the forward that just ran. Accumulation does not concatenate those micro-batches before the normalization. Running statistics and the per-step normalized activations still see `batch` images, not `nbs` images. A batch of 4 has noisier BatchNorm statistics than a batch of 16 even when both accumulate toward 64 images.

### What `nbs` actually does (verified)

In `_build_train_pipeline`:

```text
accumulate = max(round(nbs / batch_size), 1)
weight_decay_used = weight_decay * batch_size * accumulate / nbs
```

If `time` is unset, `accumulate` is then capped by `len(train_loader) * epochs`.

`default.yaml` describes `nbs` as the nominal batch size: gradients accumulate over `round(nbs / batch)` batches per optimizer step. The training loop calls `loss.backward()` on every micro-batch and calls `optimizer_step` only when `ni - last_opt_step >= accumulate`. `optimizer_step` clips gradients, steps, and zeroes. It does not divide the loss by `accumulate`.

`v8DetectionLoss.loss` returns the box/cls/dfl vector multiplied by `batch_size` before the backward. Summing `accumulate` of those backwards therefore scales the gradient roughly with `batch_size * accumulate`. When that product equals `nbs` (64), the gradient scale matches a nominal batch of 64, approximately: the assignment normalizer `target_scores_sum` still depends on the objects in each micro-batch, and BatchNorm still sees the micro-batch.

The weight-decay multiplier equals 1 in that same case, so the decay applied to weights stays at the configured 0.0005. The smoke printed `weight(decay=0.0005)` at `batch=16`, `nbs=64`: `round(64/16) = 4`, and `16 * 4 / 64 = 1`.

When `nbs / batch` is not an integer, `round` makes the product differ from 64 and the multiplier is not 1. Example: batch 48 gives `accumulate = round(64/48) = 1` and a multiplier of `48/64 = 0.75`. The gradient is then scaled by about 48 rather than 64, and the decay is 0.75 times the yaml value. Prefer a batch that divides 64 (32, 16, 8, 4) so the multiplier stays 1, unless VRAM forces another integer. Whatever integer is chosen is then used for all three models, including the ones that would have fit a larger batch.

During warmup, `accumulate` itself is interpolated from 1 up to the nominal value, so the first warmup iterations are single micro-batches on purpose.

Learning rate is **not** scaled linearly with batch size anywhere in this path. Matching `nbs` is how Ultralytics keeps the step comparable. It is not the same thing as "we trained at batch 64."

## 5. Choosing `batch` on the GPU

`batch` in the yaml is the library default, 16. Replace it before the comparison. Do not leave `batch=-1` or a fraction in `base.yaml` for the three runs.

`check_train_batch_size` / `autobatch` (`ultralytics/utils/autobatch.py`):

- `batch=-1` uses a memory fraction of **0.60** (`fraction=batch if 0 < batch < 1 else 0.6`).
- `batch=0.70` uses 70% of free GPU memory.
- The probe profiles batch sizes 1, 2, 4, 8, 16, and also 32 and 64 when the GPU has at least 16 GiB total, fits a line to the memory readings, and returns an integer.
- It is intended for CUDA. On CPU or MPS it returns the default batch and does not search. `cudnn.benchmark` must be False or it also refuses.

Procedure:

1. Fill the GPU blanks in section 1 and confirm `device 0` is that GPU.
2. Keep `imgsz: 640` for the first probe.
3. Probe **`yolov8m.pt` only**. It is the largest of the three: the installed `yolov8.yaml` scale `m` summary is 25,902,640 parameters and 79.3 GFLOPs at COCO `nc=80`. Scale `s` is 11,166,560 parameters and 28.8 GFLOPs. YOLO11s is smaller still (9,458,752 parameters, 21.7 GFLOPs, also at `nc=80`). The BeeComb head is smaller (`nc=7`), and the order does not change.
4. A one-epoch probe with `batch=-1` or `batch=0.70` both prints the integer and then trains at it. Discard that run. It is not one of the three comparison runs.
5. Write the integer into `configs/base.yaml` `batch`. Train all three models at that integer, including a fresh yolov8m run.
6. Prefer a divisor of 64 when two integers are close (section 4).

Two caveats from the source, both of which make the probe optimistic for this dataset:

- `Trainer.auto_batch()` is called with the default `max_num_obj=0`. `profile_ops` documents that 0 **disables** the simulated training-loss memory. The probe does not see hundreds of boxes per frame. Real comb images can use more memory than the number it prints.
- On the first epoch, single GPU, an out-of-memory error halves `batch` and retries, up to three times (`trainer.py` OOM branch). If that log line appears, the run is no longer at the yaml batch. Stop, write the reduced integer into `base.yaml`, and restart **all** models. Do not keep a yolov8m run whose batch was halved underneath the other two.

If the integer at `imgsz=640` is comfortable (a divisor of 64, and no OOM on a real yolov8m epoch), a later experiment can repeat the probe at `imgsz=1024` and, only if that still fits, at 1280. That experiment changes `imgsz` for all three models and is reported separately. It is not part of this file until the probe exists. 1024 before 1280 is the smaller memory step (2.56x pixels, then 4x).

## 6. When the GPU run is still not reproducible

`seed: 0` and `deterministic: True` are set. They are not a bitwise guarantee. `init_seeds` in `ultralytics/utils/torch_utils.py`:

- Seeds Python `random`, NumPy, and `torch.manual_seed`. The trainer calls it as `init_seeds(seed + 1 + RANK)`. `RANK` is -1 unless `WORLD_SIZE > 1`, so on one GPU `seed + 1 + (-1)` equals the yaml seed. A DDP rank would get `seed + 1 + rank`. This comparison is one GPU.
- If `deterministic` is True and torch is at least 2.0, it calls `torch.use_deterministic_algorithms(True, warn_only=True)`, sets `torch.backends.cudnn.deterministic = True`, sets `CUBLAS_WORKSPACE_CONFIG=:4096:8`, and sets `PYTHONHASHSEED`.
- `warn_only=True` means an operation that has no deterministic implementation **warns and still runs**. It does not abort the step.
- `cudnn.benchmark` is left off (the line that would set it is commented, because AutoBatch requires it off).

The following still change numeric results between machines or between runs:

- **cuDNN and atomic GPU ops.** Deterministic mode asks for deterministic kernels. It cannot invent one that the installed cuDNN does not have, and `warn_only` lets the nondeterministic kernel run.
- **AMP.** `amp: True` trains CUDA in FP16 (or the AMP check's mixed precision). FP16 rounding depends on the GPU and the kernel. `check_amp` returns False immediately on CPU, and it also returns False for a name match on GeForce GTX 1650/1660/1630 and several T-series Quadro cards, because those GPUs have produced NaN loss under AMP. If the FILL GPU is on that list, all three models train in FP32 together; record the AMP log line, because the saved `args.yaml` can still say `amp: True` while `self.amp` was set False.
- **Different GPU, driver, or CUDA.** The smoke torch was `2.15.0+cu132` on Linux. The Windows wheel will differ. Results are comparable across the three models only when they share one machine and one environment.
- **Worker count.** `seed_worker` seeds NumPy and Python `random` from `torch.initial_seed()`. A different `workers` value changes which worker augments which index. `workers: 2` is part of the recipe, not only a speed setting.
- **The OOM halving path** in section 5, if it changes `batch` mid-run.
- **`close_mosaic` on a short run.** The loader closes mosaic when `epoch >= epochs - close_mosaic`. The smoke used `epochs=1`, so mosaic closed immediately (`Closing dataloader mosaic` before epoch 1). A 100-epoch run does not do that. The smoke does not show mosaic behavior for the real recipe.

Two runs on the same Windows GPU with the same file can still differ in the last bits. The paper should claim a fixed, shared recipe, not bitwise reproduction.

## 7. Windows

- DataLoader workers use **spawn**, not fork. Any script that builds a loader (`train.py` and the others) needs a top-level `if __name__ == "__main__":` guard. Without it, spawn re-imports the file and starts training again inside the worker.
- `workers: 2` is the conservative ceiling. Raise it only by editing `base.yaml` once, after one epoch is stable, and train all three models at the new value. `0` works and is the slowest. The default 8 is what we are not using.
- Paths in yaml use forward slashes. `pathlib` accepts them. `img2label_paths` replaces the last `images` directory using `os.sep`, so on Windows the resolved path must contain `\images\`. `Path.resolve()` produces that native form. Do not write backslash escapes into the yaml.
- `channels_last` is pinned False because the automatic switch is off on Windows and on for other operating systems (section 3).
- Launching a script from a shortcut, another drive, or an editor whose working directory is not the repo root breaks a relative `path: data`. Section 8 is the fix. It is a helper the later scripts call. It is not written yet.

## 8. Dataset paths, and the split

`configs/data.yaml` points `path: data` at the repository root, with:

```text
train: train/images
val: valid/images
test: test/images
```

Roboflow's folder is named `valid`. The key Ultralytics requires is `val`. This version renames a `validation:` key to `val`. It does **not** rename `valid:`. Leaving Roboflow's key unchanged would fail the "train and val are required" check.

`check_det_dataset` resolves paths as follows:

1. `path` is the yaml value, or the yaml file's directory if `path` is missing.
2. If that path is relative **and does not exist**, it is replaced by `<Ultralytics datasets_dir>/<path>`. The missing directory is judged from the process working directory. On Windows that is a different folder whenever the script is not started at the repo root.
3. `train`, `val`, and `test` are joined onto `path` and `Path.resolve()` is called.

The later helper, called before YOLO sees the file:

```python
repo = Path(__file__).resolve().parents[1]   # repository root, not the working directory
cfg = YAML.load(repo / "configs" / "data.yaml")
root = (repo / cfg["path"]).resolve()        # absolute data/
cfg["path"] = str(root)                      # native separators
# write cfg to runs/_resolved_data.yaml (gitignored) and pass that path as data=
```

An absolute `path` that exists skips step 2. Edit `configs/data.yaml` only. Do not hand-edit the resolved copy.

Labels are found by replacing the last `images` path component with `labels`. The export must keep `train/images` next to `train/labels`, and the same for `valid` and `test`. A folder of loose jpg files will not find its txt files.

Class indices, in order: 0 bee bread, 1 honey, 2 capped honey, 3 worker capped brood, 4 capped drone brood, 5 eggs, 6 larvae. Label files store the integer. Confirm the Roboflow export uses this order before the first real train. Reordering the names without rewriting the txt files silently mislabels every class.

`download_data.py` (not written yet) may only copy the export into `data/`. It must not call `ultralytics.data.split` or any shuffle. `fraction` stays 1.0.

Output folders: leave `project` unset and pass `name` as `yolov8s`, `yolo11s`, or `yolov8m`. With the default `runs_dir` of `runs` and the working directory at the repo root, checkpoints go to `runs/detect/<name>/`. The smoke passed `project=runs/smoke` and landed in `runs/detect/runs/smoke/coco8_cpu`, which is the nested path to avoid.

## 9. Models

The editable list is `yolov8s.pt`, `yolo11s.pt`, `yolov8m.pt`. They are detection checkpoints. The parameter counts in `configs/models.yaml` are the COCO `nc=80` summaries from the installed model yaml files, not BeeComb measurements.

`yolov8n.pt` was used for the smoke only. It is not a candidate and does not belong in the comparison table.

## 10. Validation that was run

### Config acceptance

Environment: Python 3.12.3, ultralytics 8.4.193, torch 2.15.0+cu132.

```python
from ultralytics.cfg import get_cfg, DEFAULT_CFG_DICT, check_dict_alignment
from ultralytics.utils import YAML
cfg = YAML.load("configs/base.yaml")
check_dict_alignment(DEFAULT_CFG_DICT, cfg)
ns = get_cfg(DEFAULT_CFG_DICT, cfg)
```

Result: 60 keys loaded, `check_dict_alignment` raised nothing, `get_cfg` returned a namespace. The only keys whose values differ from `DEFAULT_CFG_DICT` are `patience` 0, `optimizer` SGD, `device` 0, `workers` 2, `channels_last` False, and `flipud` 0.5.

### One-epoch CPU smoke

Not a BeeComb run. Not a comparison of the three models. coco8 has 4 train images and 4 val images, and it keeps COCO's 80 classes.

```python
from ultralytics import YOLO
YOLO("yolov8n.pt").train(
    cfg="configs/base.yaml",
    data="coco8.yaml",
    epochs=1,
    imgsz=128,
    device="cpu",
    project="runs/smoke",
    name="coco8_cpu",
    exist_ok=True,
)
```

Overrides relative to `base.yaml`: `data`, `epochs=1`, `imgsz=128`, `device=cpu`, plus `project`, `name`, and `exist_ok` so the smoke did not pretend to be a comparison run. `batch` stayed 16.

Result: the process exited 0. One epoch finished (`1 epochs completed in 0.001 hours`). The startup banner showed `optimizer=SGD`, `lr0=0.01`, `momentum=0.937`, `patience=0`, `flipud=0.5`, `deterministic=True`, `seed=0`, `workers=2`, `channels_last=False`, `fraction=1.0`, `mosaic=1.0`, `copy_paste=0.0`. The optimizer line was `SGD(lr=0.01, momentum=0.937)` with weight decay 0.0005 on the decay group. The loader then printed `Using 0 dataloader workers` (CPU override) and `Closing dataloader mosaic` (because `epochs=1` and `close_mosaic=10`, not because the 100-epoch recipe disables mosaic). Validation ran and wrote `best.pt`. The coco8 fitness number is a 4-image smoke at imgsz 128 and is not a result to cite.

`runs/` and `*.pt` are gitignored. The smoke checkpoint was not committed.

## 11. Open questions

1. GPU model, VRAM, driver, CUDA, Windows version, and Python version (section 1).
2. The AutoBatch integer for `yolov8m.pt` at `imgsz=640`, and whether a real epoch OOMs after that probe (section 5).
3. Whether the first reported comparison stays at imgsz 640 or moves to 1024 after that probe.
4. Image counts per split, boxes per image, and the maximum boxes in one image (`max_det=300` and `warmup_epochs=3` both depend on this).
5. Confirmation that the Roboflow txt files are 5-column boxes and that class index 0 is bee bread through index 6 larvae.
6. Whether any class is orientation-specific. If it is, `flipud` goes back to 0.0 for every model.
7. Whether `hsv_s=0.7` and `hsv_v=0.4` are too strong for classes that differ by color. No new values are proposed.
8. Whether `cls_pw` should leave 0 after class counts exist. Same change for every model, or no change.
9. Whether the supervisor prefers MuSGD (what `auto` selects for a long run, with momentum 0.9 and a 3x head lr on some parameters) instead of SGD. SGD is pinned so that `lr0=0.01` and `momentum=0.937` are the values that train.
10. Ultralytics 8.4.193 is AGPL-3.0. Record that license before the code or a trained weight file is distributed. This note is not legal advice.
