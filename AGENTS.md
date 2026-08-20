# AI Agent Guidelines for the YOLOv5 Node

Learning Loop **Trainer** and **Detector** nodes for YOLOv5 (object detection, point detection and
classification), built on the [Learning Loop Node Library](https://github.com/zauberzeug/learning_loop_node).
The deep-learning part comes from [ultralytics/yolov5](https://github.com/ultralytics/yolov5).
This repository is public.

For coding standards see [CONTRIBUTING.md](CONTRIBUTING.md); it also covers the local development
setup, the release process and the rule for vendored upstream code. [README.md](README.md)
documents the supported hyperparameters and the published docker images.

## Layout

Three independent uv sub-projects, each with its own `pyproject.toml`, `uv.lock`, `Dockerfile` and
`docker.sh`:

- `trainer/` — the training node. Runs on GPU in an NVIDIA PyTorch image; `app_code/` holds the
  trainer logic, `app_code/yolov5/` the upstream code, `app_code/tests/` the suite.
- `detector/` — the TensorRT detector. `detector/tensorrtx/` is **vendored upstream code**; see the
  CONTRIBUTING section before changing anything in it.
- `detector_cpu/` — the CPU detector. The only sub-project that installs on macOS, which is why
  CONTRIBUTING points the root `.venv` at it.

`sync.py` live-syncs this repository *and* `../learning_loop_node` onto a robot for on-device
debugging, so the library is expected beside this checkout.

## Running and testing

Work from **inside** a sub-project, and use the existing environment rather than letting `uv`
re-sync — the trainer and detector environments do not install on macOS:

```bash
cd detector_cpu && uv run --no-sync python -m pytest -v
```

`./run_tests.sh` runs the `detector_cpu` suite; the trainer and detector lines in it are commented
out because they need a GPU and a reachable loop. **CI is therefore the only real check of the
trainer code** — `.github/workflows/pytest.yml` builds the trainer image on a self-hosted GPU
runner and runs `pytest -vv` inside it against a live loop, `build.yml` compiles the TensorRT
detector. Report the CI result rather than claiming the trainer was verified locally.

There is no `.pre-commit-config.yaml` here. Lint per sub-project, where the ruff config lives:

```bash
cd trainer && uv run --no-sync ruff check .
```

`docker-deploy.yml` publishes the images on a GitHub release tagged `v<MAJOR>.<MINOR>.<PATCH>`.

## Working in this repository

- **Hyperparameters are the cheap way to report a value to the loop.** Anything a trainer writes to
  `training.hyperparameters` lands on the model and shows up in its hyperparameter view — no new
  plumbing in the loop needed. `batch_size` and `trainer_version` already use this.
- **Keep upstream mergeable.** Changes in `detector/tensorrtx` (and in the vendored yolov5 code)
  must carry a `PATCH (yolov5-node)` comment stating what deviates, so `grep -rn "PATCH (yolov5-node)"`
  lists every deviation.
- Each sub-project pins its own `learning_loop_node` version; the image tag `A.B.C-nlvX.Y.Z`
  encodes the node version and the library version it was built against.
