# PINN-KGS-code

Code, trained checkpoints, and figure scripts for the paper:

> **Physics-Informed Neural Networks for the Coupled Klein–Gordon–Schrödinger
> System: A Systematic Training Methodology**
> A. Shehzad and S. Haq (submitted to *Nonlinear Dynamics*).

This repository accompanies the manuscript and supports its *Data Availability*
statement. It contains the training scripts, the trained network checkpoints,
and the scripts used to generate every figure and table in the paper.

## Problem

We solve the one-dimensional coupled Klein–Gordon–Schrödinger (KGS) system

```
 i u_t + (1/2) u_xx + u v = 0
 v_tt - v_xx + v = |u|^2
```

for the single-soliton solution with wave velocity `ν = 0.8` on the domain
`x ∈ [-10, 10]`, `t ∈ [0, 6]`, using physics-informed neural networks (PINNs).
The second-order system is reduced to first order via the auxiliary variable
`p = v_t`, so the network maps `(x, t) → (u_R, u_I, v, p)`.

The conserved quantity (nucleon number) is

```
 C(t) = ∫ |u|^2 dx = 5.000000 (exact, for this soliton).
```

## Repository layout

| File | Role in the paper |
|------|-------------------|
| `phase0_kgs_warmstart_improved.py`        | Initial-condition warm-start (Sec. 3) |
| `phase1_kgs_baseline_chap5fixed.py`       | Baseline PINN training (Sec. 3; Table 3 baseline) |
| `phase2_kgs_diagnostics_chap5fixed.py`    | Diagnostic analysis of the baseline (Sec. 4) |
| `phase3_revised_chap5fixed.py`            | Intermediate revised method |
| `pinn_kgs_pipeline_multiseed_aligned.py`  | Full progressive pipeline, 5 seeds (Sec. 6; Tables 3–5) |
| `pinn_kgs_ablation_updated.py`            | Ablation study (Sec. 6.4; Table 6) |
| `generate_paper_figures.py`               | Generates the solution-profile, error, and conservation figures |
| `generate_phase2_figures.py`              | Generates the temporal-propagation and loss-imbalance figures |

### Checkpoints (`*.pt`)

| File | Contents |
|------|----------|
| `phase0_warmstart_checkpoint.pt` | Network after the IC warm-start |
| `phase1_baseline_checkpoint.pt`  | Trained baseline PINN |
| `phase3_revised_checkpoint.pt`   | Revised-method network |
| `phase4_final_checkpoint.pt`     | Final pipeline (best run); embeds per-run results |
| `phase4_runD/E/F_checkpoint.pt`  | Individual pipeline runs |
| `phase4b_final_checkpoint.pt`    | Final pipeline checkpoint used for the paper figures |

All checkpoints are dictionaries containing at least a `model_state_dict` for the
`PINN_KGS` architecture (6 hidden layers × 128 neurons, `tanh`, 83,460
parameters). Load with `torch.load(path, map_location='cpu', weights_only=False)`.

### Figures

`paper_figures/` holds the generated temporal-propagation and loss-imbalance
figures (PDF + PNG). The remaining figures are produced by the scripts above.

## Reproducing the results

The five reported seeds are `{42, 123, 456, 789, 2024}`. Each seed controls both
the network initialisation and the collocation sampling. Training was performed
on a single NVIDIA Tesla T4 GPU with PyTorch (single precision); the full
pipeline takes approximately 7.8 h per seed.

```bash
pip install -r requirements.txt

# 1. Warm-start, baseline, diagnostics
python phase0_kgs_warmstart_improved.py
python phase1_kgs_baseline_chap5fixed.py
python phase2_kgs_diagnostics_chap5fixed.py

# 2. Full progressive pipeline (5 seeds) and ablation
python pinn_kgs_pipeline_multiseed_aligned.py
python pinn_kgs_ablation_updated.py

# 3. Regenerate the paper figures from the saved checkpoints
python generate_phase2_figures.py
python generate_paper_figures.py
```

Scripts read and write checkpoints by filename in the working directory, so run
them from the repository root.

## Citation

If you use this code, please cite the paper (full reference to be added on
publication).

## License

Released under the MIT License. See [`LICENSE`](LICENSE).
