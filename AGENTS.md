# AI Agent Guidelines for the YOLOv5 Node

Learning Loop **Trainer** and **Detector** nodes for YOLOv5 (object detection, point detection and
classification), built on the [Learning Loop Node Library](https://github.com/zauberzeug/learning_loop_node).
The deep-learning part comes from [ultralytics/yolov5](https://github.com/ultralytics/yolov5).
This repository is public.

For coding standards see [CONTRIBUTING.md](CONTRIBUTING.md); it also covers the local development
setup, the release process and the rule for vendored upstream code. [README.md](README.md)
documents the supported hyperparameters and the published docker images.

## Layout

Three independent uv sub-projects, each with its own `pyproject.toml`, `uv.lock`, `Dockerfile`,
`docker.sh` and `docker.conf`; `scripts/node-docker.sh` at the root drives all three. That driver is
shared verbatim with the other node repositories and kept in step by hand, so a fix belongs in all
of them. `.env` is passed whole into the container, so a setting the node declares in `main.py`
needs no change to either file; `CONTAINER_NAME`, `HOST_PORT` and `LINKLL` are read by `docker.sh`
itself and never reach the node.

Sub-projects:

- `trainer/` — the training node. Runs on GPU in an NVIDIA PyTorch image; `app_code/` holds the
  trainer logic, `app_code/yolov5/` **vendored upstream code**, `app_code/tests/` the suite.
- `detector/` — the TensorRT detector. `detector/tensorrtx/` is **vendored upstream code**; see the
  CONTRIBUTING section before changing anything in it.
- `detector_cpu/` — the CPU detector, a second implementation of the same detector contract on top
  of torch/ultralytics. It shares no *yolov5* code with `detector/`; what the two do have in common
  — NMS geometry, clipping, building the loop's detection dataclasses — comes from
  `learning_loop_node.detector.postprocess`. The only sub-project that installs on macOS, which is
  why CONTRIBUTING points the root `.venv` at it.

The model-agnostic half of a detector lives in the node library: `postprocess.to_image_metadata`
turns detections into what the loop expects, `postprocess.bbox_iou` and `geometry.clip_*` are the
shared primitives. Only the yolov5-specific parts — the packed `[cx,cy,w,h,conf,probs...]` output
layout, the letterbox correction in `xywh2xyxy`, the tensorrtx engine build — belong here.

The trainer's auto-detection pass goes through the same module: `_parse_file` reads the label
files into `postprocess.Prediction` values and `postprocess.to_detections` builds the loop's
dataclasses from them. **Nothing in this repository constructs `BoxDetection` or `PointDetection`
itself**, so detector and trainer cannot drift apart in how they resolve a category, clip to the
image, or drop a degenerate prediction. Note what that shared step does: predictions of 2 px or
less are dropped, points included, since a point is carried as the box around it.

`sync.py` live-syncs this repository *and* `../learning_loop_node` onto a robot for on-device
debugging, so the library is expected beside this checkout.

## How the nodes work

Each `main.py` only builds a `TrainerNode`/`DetectorNode` from the library and hands it one class of
ours; everything below is our side of that contract.

**The batch size is measured, not configured.** The node starts `train_det.py` with
`--batch-size -1` and, as `--max-batch-size`, the bound it read with the library's
`requested_batch_size`. Where upstream yolov5 ran its autobatch, right after the model is built and
before the optimizer, `train_det.py` hands that model to `batch_size_calculation.measure`, so the
probe runs in the training's own process and CUDA context, under its seeds, its deterministic
kernels and its `check_amp` verdict. It measures a copy of the model, while the training's own
waits on the CPU, with a step resembling the training's — EMA copy, three-group optimizer, the
real `ComputeLoss`, backward, clipping, optimizer step — and, as `val_step`, the validation
`train_det.py` runs between epochs: the EMA copy at `batch_size // 2`, on the padded shape of the
rectangular validation loader. The step is all this repository supplies, as a `ProbeStep` the
library's `measure_batch_size` builds through a factory and releases itself. `train_det.py` reports
the settled size through `training_run`, and the node adds it to the hyperparameters it syncs;
a probe that finds not even the smallest batch fitting ends the training with a `CriticalError`.
So does a `--img` the model's largest stride does not divide: `train_det.py` rejects it through
`check_resolution` just before the probe, where upstream would round it. So does a missing graphics
card, as `NoGpuError`: the trainer does not fall back to the CPU. `train_det.py` records these
errors in its result folder through `training_run.reporting`, and the node raises them again from
there in `_get_executor_error_from_log`. The node, which has no model, checks
`resolution` against `MIN_STRIDE` before it starts `train_det.py`, which already catches anything
that suits no model. Three of the bounds are ours: `MIN_BATCH_SIZE`, because `train_det.py` validates at
`batch_size // 2`; `DEFAULT_MAX_BATCH_SIZE` for a training that sets no `max_batch_size`; and the
`sample_count` counted off the `train/` folder, which keeps an epoch enough optimizer steps to
mean something. That count is floored at `NOMINAL_BATCH_SIZE` times the library's
`MIN_TRAIN_STEPS_PER_EPOCH`, because `train_det.py` accumulates gradients up to a batch of 64, so a
smaller batch adds no optimizer steps. What the probe has to mirror of the training — that
nominal batch, the validation padding, the loss-weight scaling — lives in `training_recipe`, and
`train_det.py` reads it from there too. `--vram-limit-gb` narrows the card the probe measures
against, through the `limit_cuda_memory` call `train_det.py` makes at its start; because the cap
does not survive a spawn, the node hands the same number to `train_det.py` and `pred_det.py`.

**The trainer never trains in-process.** `Yolov5TrainerLogic` (`trainer/app_code/yolov5_trainer.py`)
implements the library's abstract `TrainerLogic` hooks and shells out through the library's
`Executor` to `trainer/train_det.py` (and `pred_det.py` for `_detect`). What a hook has to return is
therefore recovered *from the subprocess log and from files on disk*, never from a return value:
`_get_progress_from_log` scrapes the `epoch/total` column, `_get_executor_error_from_log` greps for
CUDA messages, `_get_new_best_training_state` picks up the newest weight file through
`model_files.py`. Changing what `train_det.py` prints or where it writes breaks those parsers.

**Category order is the wire format.** `yolov5_format.create_file_structure` converts the loop's
`image_data` into the layout yolov5 expects — `train/` and `test/` folders of image symlinks with one
`class x y w h` text file per image — encoding each category uuid as its *index* in
`training.categories`. Every conversion back (`_parse_file`, the confusion matrix in
`_get_new_best_training_state`, the `--point_sizes_by_id`/`--flip_label_pairs` arguments) depends on
that same ordering.

**Point detection rides on boxes.** A point annotation is written as a box of `point_size` and turned
back into a point by category type. The point hyperparameters are not handed to yolov5 as
hyperparameters; `_save_additional_hyperparameters` parses them out of `hyp.yaml` and re-encodes them
as index-based command line arguments.

**Model handoff between sub-projects.** The trainer publishes `yolov5_pytorch` (`model.pt`) *and*
`yolov5_wts` (built by `generate_wts.py`); the TensorRT detector declares
`model_format = 'yolov5_wts'` and converts that `.wts` into `model.engine` on first load.

**The TensorRT detector compiles itself at model load.** `_build_module_lib` rewrites `kNumClass`,
`kInputH/W` and `USE_FP16` in `tensorrtx/src/config.h` and runs cmake+make *inside the running
container*, guarded by a `build.json` holding the last `_LibConfig`. A changed resolution, category
count or `WEIGHT_TYPE` thus forces a recompile, while an existing `model.engine` is reused until it
is deleted. This is why detector changes need final validation on a device.

## Running and testing

Use the existing environment rather than letting `uv` re-sync — the trainer and detector
environments do not install on macOS. `uv run` resolves the environment per sub-project, so the
root `.venv` from CONTRIBUTING only applies if you point `uv` at it:

```bash
UV_PROJECT_ENVIRONMENT=$(pwd)/.venv uv run --project ./detector_cpu --no-sync python -m pytest -v
cd detector_cpu && uv run --no-sync python -m pytest -v -k test_name   # needs detector_cpu/.venv
```

`./run_tests.sh` runs the `detector_cpu` suite, and passes a single argument on as a `-k` filter; the
trainer and detector lines in it are commented out because they need a GPU and a reachable loop. The
trainer suite additionally generates and deletes a project in the loop from its `conftest.py`
fixtures, so it cannot run offline. **CI is therefore the only real check of the trainer code** —
`.github/workflows/pytest.yml` builds the trainer image on a self-hosted GPU runner and runs
`pytest -vv` inside it against a live loop, `build.yml` compiles the TensorRT detector. When asked,
report the CI result rather than claiming the trainer was verified locally.

There is no `.pre-commit-config.yaml` here. Lint per sub-project, where the ruff config lives:

```bash
cd trainer && uv run --no-sync ruff check .
```

`docker-deploy.yml` publishes the images on a GitHub release tagged `v<MAJOR>.<MINOR>.<PATCH>`.

## Working in this repository

- **Hyperparameters are a cheap way to report a value to the loop.** Anything a trainer writes to
  `training.hyperparameters` lands on the model and shows up in its hyperparameter view — no new
  plumbing in the loop needed. `batch_size` and `trainer_version` already use this. Keep such an
  output apart from any input: the settled size goes to `batch_size`, never back into the
  `max_batch_size` bound, because the node saves `training.hyperparameters` with the training and
  a training resumed after a restart reads them back; it would otherwise take its first run's
  measurement as its bound. (The loop itself never hands reported values to a later training.)
- **Keep upstream mergeable.** Changes in the vendored code, `detector/tensorrtx` and
  `trainer/app_code/yolov5`, must carry a `PATCH (yolov5-node)` comment stating what deviates, so
  `grep -rn "PATCH (yolov5-node)"` lists every deviation. Only `detector/tensorrtx` follows this
  today; `trainer/app_code/yolov5` carries older unmarked deviations (the point-detection support in
  `utils/dataloaders.py`), so do not read a clean grep there as "unmodified". `trainer/train_det.py`
  and `trainer/pred_det.py` are not vendored: they started as upstream's `train.py` and `detect.py`
  but are our own entry scripts now, so changes there carry no marker.
- Each sub-project pins its own `learning_loop_node` version; the image tag `A.B.C-nlvX.Y.Z`
  encodes the node version and the library version it was built against.
