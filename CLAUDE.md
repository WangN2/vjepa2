# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

V-JEPA 2 is Meta FAIR's self-supervised video representation learning framework using masked latent feature prediction. Three model families: V-JEPA 2 (ViT-L/H/g), V-JEPA 2.1 (ViT-B/L/g/G with dense features), V-JEPA 2-AC (action-conditioned for robotics).

## Key Commands

```bash
# Install (editable dev mode)
pip install -e .

# Run all tests
pytest tests

# Run single test file
pytest tests/models/test_vision_transformer.py

# Run a specific test
pytest tests/models/test_vision_transformer.py::TestViTGiant::test_square_inputs

# Lint (must pass in CI)
python -m isort app evals/*.py src tests --check
python -m flake8 --config .flake8 --show-source --statistics app evals/*.py src tests
python -m black --check app evals/*.py src tests

# Auto-format
python -m isort app evals/*.py src tests
python -m black app evals/*.py src tests

# Training (local)
python -m app.main --fname configs/train/vitl16/pretrain-256px-16f.yaml --devices cuda:0

# Evaluation (local)
python -m evals.main --fname configs/eval/vitl/ssv2.yaml --devices cuda:0

# Training/Eval with SLURM
python -m app.main_distributed --fname <config.yaml> --time <minutes> --account <account>
python -m evals.main_distributed --fname <config.yaml> --time <minutes> --account <account>
```

## Code Architecture

```
src/                          # Core library (imported as src.*)
├── datasets/                 # VideoDataset, ImageNet1K, transforms, samplers
├── models/                   # VisionTransformer, predictor, attentive_pooler
├── masks/                    # Mask collators (multiseq_multiblock3d, default)
└── utils/                    # Distributed helpers, schedulers, checkpointing, logging

app/                          # Training entry points
├── main.py                   # Local multi-GPU launcher (multiprocessing.spawn)
├── main_distributed.py       # SLURM launcher (submitit)
├── vjepa/                    # V-JEPA 2 training loop
├── vjepa_2_1/                # V-JEPA 2.1 training loop (has own models/ subdir)
└── vjepa_droid/              # Action-conditioned training loop

evals/                        # Frozen evaluation (probes on frozen backbone)
├── main.py                   # Local multi-GPU launcher
├── main_distributed.py       # SLURM launcher
├── video_classification_frozen/
├── image_classification_frozen/
└── action_anticipation_frozen/

configs/                      # YAML configs driving all experiments
├── train/  train_2_1/        # Pretraining configs by model size
├── eval/  eval_2_1/          # Evaluation configs
└── inference/                # Inference-only configs

tests/                        # pytest tests (CUDA-required for model tests)
├── models/                   # Vision transformer, predictor tests
└── datasets/                 # Data loader, transforms, sampler tests
```

## Key Design Patterns

- **Config-driven**: Every experiment is a single YAML config. The `app:` field in training configs selects which module to import (via `app/scaffold.py`). Similarly `eval_name:` for evaluation configs.
- **Dynamic imports**: `scaffold.py` uses `importlib` to load `app.{module}.train` or `evals.{module}.eval` based on config.
- **Two-phase training**: Phase 1 (pretrain) → Phase 2 (cooldown) using separate configs in the same directory.
- **Multi-head probe eval**: Evaluation configs can train multiple attentive probes in parallel with different optimization params.
- **Distributed**: `init_distributed()` in `src/utils/distributed.py` auto-detects SLURM env vars. Uses NCCL. Supports FSDP via `--use_fsdp`.
- **All tests requiring CUDA are skipped if no GPU available** (marked with `@pytest.mark.skipif`).

## Model Notes

- VisionTransformer uses RoPE positional embeddings, stochastic depth, activation checkpointing, SDPA.
- Models handle both square and non-square inputs via `handle_nonsquare_inputs`.
- V-JEPA 2.1 has its own model implementations in `app/vjepa_2_1/models/`, distinct from `src/models/`.
- Checkpoints use keys like `target_encoder`, `encoder`, `classifiers`. Prefix stripping (`module.`, `backbone.`) is handled during loading.

## Config Structure

Key YAML sections:
- `data`: dataset paths, batch size, resolution, frame sampling
- `model` / `model_kwargs`: architecture, checkpoint paths
- `optimization`: lr, weight decay, warmup, EMA, multi-head probe params
- `mask`: masking strategy (pretraining only)
- `meta`: dtype (usually bfloat16), ckpt frequency, seed

## Style Constraints

- Line length: 119 (black + flake8)
- Formatter: black, import sorter: isort (profile=black)
- Linter: flake8 (select E,F,W, ignore E203,E701,W503)
- All .py files must carry the MIT copyright header from Meta
