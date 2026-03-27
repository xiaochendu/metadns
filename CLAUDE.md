# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**MDNS (Masked Diffusion Neural Sampler)** is an implementation of a neural sampler for discrete distributions using masked discrete diffusion and stochastic optimal control (NeurIPS 2025, arXiv:2508.10684). It trains neural samplers for three physics models:
- **Ising model** — 2D square lattice spin systems
- **Potts model** — 2D lattice q-state spin systems
- **CuAu alloy** — 3D FCC alloy structure via cluster expansion

Optional **Well-Tempered Metadynamics (WT-ASBS)** bias potentials enhance exploration of complex energy landscapes.

## Environment Setup

```bash
conda env create -f environment.yml
conda activate mdns
```

Key dependencies: PyTorch, TIMM, ASE 3.24.0, CLEASE 1.1.0, iCET 3.1, wandb, hydra-core.

## Common Commands

### Training
```bash
# Ising model
python train_ising.py --L 16 --beta 0.6 --batch_size 256 --num_epochs 100000

# Ising with metadynamics bias
python train_ising.py --L 16 --beta 0.6 --use_bias --bias_height 0.1667 --bias_sigma 0.05 --bias_factor 10.0 --bias_grid_size 100 --cv_min -1.0 --cv_max 1.0

# Potts model with 2D CV bias
python train_potts.py --L 4 --q 3 --beta 1.2 --use_bias --bias_height 0.0833 --bias_sigma "0.05" --cv_min "-0.6,-1.0" --cv_max "1.1,1.0" --bias_grid_size "17"

# CuAu alloy
python train_cuau.py --size 4 4 4 --temp_min 200 --temp_max 200
```

### Sampling
```bash
# Sample from a trained checkpoint
python scripts/mdns_sampling.py \
    --model-type ising \
    --L 16 --embed-dim 64 --depth 6 --num-heads 4 \
    --ckpt checkpoints/L_16_ising/ising_low.pth \
    --temps 1.667 2.269 3.571 --fields 0.0 0.1 \
    --batch-size 1024 --num-samples 10000 \
    --output-folder outputs/ising_16x16

# MCMC biased sampling with pre-trained bias
python scripts/mcmc_biased_sampling.py
```

### Tests
```bash
python -m pytest tests/
python -m pytest tests/test_bias.py          # bias potential unit tests
python -m pytest tests/test_multi_temp_field.py  # multi-temp/field batch tests
```

### Evaluation
```bash
jupyter notebook ising_model_eval.ipynb
jupyter notebook potts_model_eval.ipynb
```

## Architecture

### Training Pipeline
1. **Energy model** initialized (Ising/Potts/CuAu)
2. **Transformer** initialized (`RopeVIT` for Ising/Potts, `MultiOutputTransformer` for CuAu)
3. **Bias potential** created if `--use_bias` (1D or multi-D metadynamics)
4. **Training loop** (`utils_train.py:train()`): sample → compute loss (WDCE/MSE) → update model → update bias → EMA step

### Neural Network (`model/`)
- `vit_rope.py` — `RopeVIT`: Vision Transformer with Rotary Positional Embeddings; primary model for Ising/Potts
- `transformer.py` — `MultiOutputTransformer`: multi-head transformer with thermodynamic embeddings; used for CuAu
- `nn_utils.py` — `ThermodynamicEmbedder`: sinusoidal embeddings for temperature and field, enabling multi-condition training
- `ema.py` — `ExponentialMovingAverage`: slowly-updated "teacher" model for stable targets

### Physics Models
- `potts.py` — `LatticePottsModel`: Potts energy, Hamiltonian, Gibbs sampling
- `energy_cuau.py` — `AuCuAlloyModel`: cluster expansion energy (CLEASE/iCET), grand potential
- `bias.py` — `BiasPotential` (1D) and `BiasPotentialMultiDim` (N-D): Well-Tempered Metadynamics with Gaussian/delta kernels

### Training Utilities (`utils_train.py`)
- `train()` — main training loop
- `ReplayBuffer` — CV-binned buffer mixing old samples with current batch to stabilize training
- Loss functions: WDCE (weighted discrete cross-entropy, preferred) or MSE
- RND (random network distillation) for reward normalization

### Collective Variables (CVs)
Each model uses CVs for metadynamics tracking:
- **Ising**: magnetization (1D)
- **Potts**: state concentrations (2D)
- **CuAu**: order parameter Q across 4 FCC sublattices (1D); sublattice mapping in `utils_cuau.py`

## Key Hyperparameters

| Category | Parameter | Typical Values |
|----------|-----------|----------------|
| Architecture | `--hidden_size` / `--n_embed` | 64 (Ising/CuAu), 128 (Potts) |
| Architecture | `--n_blocks` / `--depth` | 4–6 |
| Training | `--batch_size` | 128–256 |
| Training | `--loss_fn` | `wdce` (default) or `mse` |
| Training | `--resample_every_n_step` | 5–10 |
| Training | `--wdce_num_replicates` | 8–16 |
| Metadynamics | `--bias_height` | ~0.001–0.002 (Ising), ~0.0001–0.0007 (CuAu) |
| Metadynamics | `--bias_factor` (γ) | 5–10 |
| Metadynamics | `--bias_sigma` | 0.05 |
| Metadynamics | `--bias_grid_size` | 100 (1D), 17 (2D) |

`--bias_height` should be normalized by `batch_size` (i.e., `bias_height_per_sample * batch_size`).

## Checkpoints

Pre-trained model weights are in `checkpoints/`. Resume training with:
```bash
python scripts/resume_training.py --ckpt checkpoints/... [--modified-args ...]
```

Checkpoint files store: model weights, EMA state, bias grid, optimizer state.
