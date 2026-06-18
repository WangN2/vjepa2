<!-- AGENTS.md — V-JEPA 2 -->
# AGENTS.md — V-JEPA 2

This file contains project-specific information for AI coding agents working on the V-JEPA 2 codebase.

## Project Overview

V-JEPA 2 is a self-supervised video representation learning framework developed by Meta FAIR. It trains video encoders using masked latent feature prediction on internet-scale video data. The repository contains three main model families:

- **V-JEPA 2**: Core self-supervised video encoder (ViT-L, ViT-H, ViT-g).
- **V-JEPA 2.1**: Improved training recipe focusing on dense, temporally consistent features (ViT-B, ViT-L, ViT-g, ViT-G). Uses dense predictive loss, deep self-supervision, and multi-modal tokenizers.
- **V-JEPA 2-AC**: Action-conditioned world model post-trained from V-JEPA 2 for robot manipulation tasks.

The project is a pure Python package using PyTorch. It is distributed as `vjepa2` (version `0.0.2`) and supports loading pretrained weights via PyTorch Hub (`hubconf.py`) and HuggingFace Transformers.

## Technology Stack

- **Language**: Python >= 3.11 (CI uses 3.12; development commonly uses 3.12).
- **Deep Learning**: PyTorch >= 2.0, torchvision, timm, transformers, einops, peft.
- **Video I/O**: decord (note: macOS users may need eva-decord or decord2 alternatives).
- **Distributed Training**: `torch.distributed` (NCCL backend), `submitit` for SLURM job launching.
- **Data**: webdataset, iopath, pandas, numpy, opencv-python, scikit-image, h5py.
- **Logging**: tensorboard, wandb.
- **Runtime Checking**: beartype.
- **Config**: pyyaml, python-box, fire.
- **Testing**: pytest.
- **Linting/Formatting**: black (26.3.1), flake8 (7.0.0), isort (5.13.2).

## Directory Structure

```
.
├── app/                              # Training entry points and loops
│   ├── vjepa/                        #   V-JEPA 2 pretraining logic
│   ├── vjepa_2_1/                    #   V-JEPA 2.1 pretraining logic (has own models/ subdir)
│   ├── vjepa_droid/                  #   Action-conditioned (robot) training logic
│   ├── main.py                       #   Local multi-GPU launcher (spawn)
│   ├── main_distributed.py           #   SLURM cluster launcher (submitit)
│   └── scaffold.py                   #   Dynamic importer for app.{module}.train
├── src/                              # Core package (imported as src.*)
│   ├── datasets/                     #   Data loaders, video transforms, samplers
│   ├── models/                       #   Vision Transformer, predictor, attentive pooler
│   ├── masks/                        #   Mask collators and masking utilities
│   ├── utils/                        #   Distributed helpers, logging, schedulers, checkpointing
│   └── hub/                          #   PyTorch Hub backbones and preprocessor
├── evals/                            # Frozen evaluation loops (probes on frozen backbones)
│   ├── video_classification_frozen/  #   SSv2, K400, Diving48, COIN, Jester, etc.
│   ├── image_classification_frozen/  #   ImageNet-1K
│   ├── action_anticipation_frozen/   #   EPIC-KITCHENS-100
│   ├── hub/                          #   Evaluation preprocessor for Hub
│   ├── main.py                       #   Local multi-GPU eval launcher
│   ├── main_distributed.py           #   SLURM eval launcher
│   └── scaffold.py                   #   Dynamic importer for evals.{module}.eval
├── configs/                          # YAML experiment configs
│   ├── train/                        #   V-JEPA 2 pretraining & cooldown configs
│   ├── train_2_1/                    #   V-JEPA 2.1 pretraining & cooldown configs
│   ├── eval/                         #   V-JEPA 2 evaluation configs
│   ├── eval_2_1/                     #   V-JEPA 2.1 evaluation configs
│   └── inference/                    #   Inference-only configs (using released probes)
├── tests/                            # Unit tests (pytest)
│   ├── datasets/
│   └── models/
├── notebooks/                        # Demo notebooks and standalone scripts
├── hubconf.py                        # PyTorch Hub entry point
├── setup.py                          # Package setup (setuptools)
├── pyproject.toml                    # black / isort configuration
├── .flake8                           # flake8 configuration
├── requirements.txt                  # Runtime dependencies
└── requirements-test.txt             # Lint/test dependencies
```

## Installation

Install in editable mode for development:

```bash
conda create -n vjepa2-312 python=3.12
conda activate vjepa2-312
pip install -e .
```

Install test/lint tools:

```bash
pip install -r requirements-test.txt
```

## Build and Test Commands

Run the unit test suite:

```bash
pytest tests
```

Run a single test file:

```bash
pytest tests/models/test_vision_transformer.py
```

Run a specific test:

```bash
pytest tests/models/test_vision_transformer.py::TestViTGiant::test_square_inputs
```

Run linters (must pass in CI):

```bash
python -m isort app evals/*.py src tests --check
python -m flake8 --config .flake8 --show-source --statistics app evals/*.py src tests
python -m black --check app evals/*.py src tests
```

Apply auto-formatting:

```bash
python -m isort app evals/*.py src tests
python -m black app evals/*.py src tests
```

## Code Style Guidelines

- **Line length**: 119 characters (configured in `pyproject.toml` and `.flake8`).
- **Formatter**: black.
- **Import sorter**: isort with `profile = "black"`.
- **Linter**: flake8 with `select = E,F,W` and `ignore = E203,E701,W503`.
- **File header**: All Python files should include the Meta copyright header:
  ```python
  # Copyright (c) Meta Platforms, Inc. and affiliates.
  #
  # This source code is licensed under the MIT license found in the
  # LICENSE file in the root directory of this source tree.
  ```
- **Naming**: Follow existing PyTorch conventions. Models use `snake_case` for factory functions (e.g., `vit_large`, `vit_giant_xformers_rope`) and `PascalCase` for classes (e.g., `VisionTransformer`, `AttentiveClassifier`).
- **Indentation**: 4 spaces (no tabs).

## Testing Instructions

- Tests live under `tests/` and are run with `pytest`.
- Tests cover dataset utilities (`ConcatIndices`, memory-efficient sampler, transforms) and model components (Vision Transformer, predictor, attentive pooler).
- Tests requiring CUDA are skipped if no GPU is available (marked with `@pytest.mark.skipif(not torch.cuda.is_available(), reason="...")`).
- When modifying `src/models/` or `src/datasets/`, ensure corresponding tests in `tests/models/` or `tests/datasets/` still pass.
- There is no mocking infrastructure for distributed training; distributed code is typically tested manually or on multi-GPU nodes.
- CI runs on every push: `pytest tests` in `.github/workflows/base_tests.yaml`.
- CI runs linters on push/PR to `master` and `gh/**` branches: `isort`, `flake8`, `black` in `.github/workflows/linters.yaml`.

## Running Training and Evaluation

All experiments are **config-driven** via YAML files under `configs/`.

### Local Training (Multi-GPU on one machine)

```bash
python -m app.main --fname configs/train/vitl16/pretrain-256px-16f.yaml --devices cuda:0 cuda:1
```

Use `--debugmode True` to run in the main process (useful for debugging with breakpoints).

### Distributed Training (SLURM)

```bash
python -m app.main_distributed --fname configs/train/vitl16/pretrain-256px-16f.yaml --time 6000 --account my_account --qos my_qos
```

This uses `submitit.AutoExecutor` to submit SLURM jobs. It copies the code to the experiment folder before launching.

### Local Evaluation

```bash
python -m evals.main --fname configs/eval/vitl/ssv2.yaml --devices cuda:0 cuda:1
```

### Distributed Evaluation (SLURM)

```bash
python -m evals.main_distributed --fname configs/eval/vitl/ssv2.yaml --time 8600 --account my_account --qos my_qos
```

## Config System

- Every experiment is defined by a single YAML config.
- Training configs specify `app: <module>` (e.g., `vjepa`, `vjepa_2_1`, `vjepa_droid`) which `app/scaffold.py` uses to dynamically import `app.{module}.train`.
- Evaluation configs specify `eval_name: <module>` (e.g., `video_classification_frozen`, `action_anticipation_frozen`) which `evals/scaffold.py` uses to dynamically import `evals.{module}.eval`.
- Configs are loaded with `yaml.FullLoader`.
- Important config sections:
  - `data`: dataset paths, batch size, resolution, frame sampling.
  - `model` / `model_kwargs`: architecture and checkpoint loading.
  - `optimization`: lr, weight decay, warmup, EMA schedules, multi-head probe configs for evals.
  - `mask`: masking strategy parameters (for pretraining).
  - `meta`: dtype (commonly `bfloat16`), checkpoint frequency, seed.

## Distributed Training Architecture

- Local multi-GPU: uses `multiprocessing.spawn` with one process per GPU. `CUDA_VISIBLE_DEVICES` is set per process.
- SLURM: uses `submitit` to launch jobs. `init_distributed` in `src/utils/distributed.py` auto-detects SLURM environment variables (`SLURM_NTASKS`, `SLURM_PROCID`, `SLURM_LOCALID`) and initializes the NCCL process group.
- Supports FSDP via the `--use_fsdp` flag in evaluation.
- Common distributed utilities in `src/utils/distributed.py` include custom `AllGather`, `AllReduceSum`, and `AllReduce` autograd functions.

## Model Architecture Notes

- **Vision Transformer** (`src/models/vision_transformer.py`): Supports 2D image and 3D video patch embedding, RoPE positional embeddings, stochastic depth, activation checkpointing, and SDPA. Handles both square and non-square inputs via `handle_nonsquare_inputs`.
- **Predictor** (`src/models/predictor.py`): Latent feature predictor used in the self-supervised objective.
- **Attentive Pooler / Classifier** (`src/models/attentive_pooler.py`): Used for downstream evaluations; performs attentive pooling over patch tokens.
- **Masking** (`src/masks/`): `multiseq_multiblock3d` and `default` mask collators support spatial-temporal masking for video self-supervision.
- **V-JEPA 2.1** models live under `app/vjepa_2_1/models/` and have their own Vision Transformer, predictor, and masking distribution utilities, distinct from the base `src/models/`.
- **V-JEPA 2-AC** uses an action-conditioned predictor (`src/models/ac_predictor.py`) post-trained on robot trajectory data.

## Data Pipeline

- Video data is loaded via `decord`.
- `src/datasets/video_dataset.py` implements `VideoDataset` with clip sampling, frame stepping, and filtering.
- `src/datasets/data_manager.py` is the factory for initializing datasets and dataloaders (ImageNet-1K or VideoDataset).
- Transforms are in `src/datasets/utils/video/` and include normalization, random resized crop, random erasing, and RandAugment variants.
- Evaluations commonly use multi-clip testing (multiple spatial crops per video).
- The robot manipulation data pipeline (`app/vjepa_droid/droid.py`) is separate and loads trajectory CSVs with camera views.

## Checkpointing and Hub

- Pretrained checkpoints are hosted by Meta and on HuggingFace.
- `hubconf.py` exposes models for `torch.hub.load('facebookresearch/vjepa2', ...)`.
- Checkpoint keys commonly include `target_encoder`, `encoder`, `predictor`, and `classifiers`.
- The code expects checkpoints to be PyTorch state dicts and handles `module.` and `backbone.` prefix stripping during loading.
- Demo scripts prefer `torch.load(..., weights_only=True)`; some legacy paths may not enforce this.

## Security Considerations

- The project does not handle user credentials or sensitive data.
- Checkpoint loading uses `torch.load(..., weights_only=True)` in demo scripts, but some legacy paths may not; prefer `weights_only=True` when adding new loading code.
- SLURM launchers (`main_distributed.py`) copy the entire codebase to the experiment folder; be mindful that this will include any uncommitted local files.

## macOS Notes

- The project depends on `decord`, which does not officially support macOS and is no longer maintained.
- macOS users may need to replace `decord` with `eva-decord` or `decord2`. This is not automated in `requirements.txt`.

## License

The majority of the codebase is under the MIT license. A few files under `src/datasets/utils/video/` and `src/datasets/utils/worker_init_fn.py` are under the Apache 2.0 license. See `LICENSE` and `APACHE-LICENSE` in the repository root.
