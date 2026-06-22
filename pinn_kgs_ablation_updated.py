#!/usr/bin/env python3
"""
================================================================================
PINN-KGS: Updated Ablation Study
================================================================================

This updated ablation script does four things the earlier ablation file did not:
  1. Computes BOTH L_inf and L2 errors
  2. Tracks conservation drift explicitly
  3. Includes A5 as an evaluation-only entry from phase4b_final_checkpoint.pt
  4. Saves a clean JSON summary for paper tables

Ablation configurations:
  A0: Phase 1 baseline (evaluation only)
  A1: + Conservation only (20K epochs on [0,6], beta=5)
  A2: + Time-march only (3 stages, no conservation)
  A3: + Time-march + Conservation
  A4: + Time-march + Conservation + Weighted Collocation (Run D recipe)
  A5: Full final pipeline result (evaluation only from phase4b_final_checkpoint.pt)

Required checkpoints:
  - phase1_baseline_checkpoint.pt   (for A0-A4)
  - phase4b_final_checkpoint.pt     (for A5)

Outputs:
  - ablation_results_updated.json
  - ablation_summary_updated.png
  - optional per-ablation checkpoints for A1-A4
"""

import os
import json
from copy import deepcopy
from time import time

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

np_trapz = np.trapezoid if hasattr(np, 'trapezoid') else np.trapz
torch_trapz = torch.trapezoid if hasattr(torch, 'trapezoid') else torch.trapz

# =============================================================================
# Configuration
# =============================================================================
SEED = 42
X_MIN, X_MAX = -10.0, 10.0
T_MAX = 6.0
NU_0 = 0.8
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

PHASE1_CKPT = 'phase1_baseline_checkpoint.pt'
FINAL_CKPT = 'phase4b_final_checkpoint.pt'


def set_seed(seed: int = 42) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_seed(SEED)
print(f"Device: {DEVICE}")
if DEVICE.type == 'cuda':
    print(f"GPU: {torch.cuda.get_device_name(0)}")


# =============================================================================
# Exact solution
# =============================================================================
class KGSExactSolution:
    def __init__(self, nu=0.8, x0=0.0):
        assert abs(nu) < 1.0
        self.nu = nu
        self.x0 = x0
        self.omn2 = 1.0 - nu**2
        self.sf = np.sqrt(self.omn2)
        self.amp_u = (3.0 * np.sqrt(2.0)) / (4.0 * self.sf)
        self.amp_v = 3.0 / (4.0 * self.omn2)

    def _eta(self, x, t):
        return (x - self.nu * t - self.x0) / (2.0 * self.sf)

    def _phase(self, x, t):
        return self.nu * x + ((1.0 - self.nu**2 + self.nu**4) / (2.0 * self.omn2)) * t

    def _sech2(self, x, t):
        return (1.0 / np.cosh(self._eta(x, t)))**2

    def u_real(self, x, t):
        return self.amp_u * self._sech2(x, t) * np.cos(self._phase(x, t))

    def u_imag(self, x, t):
        return self.amp_u * self._sech2(x, t) * np.sin(self._phase(x, t))

    def v(self, x, t):
        return self.amp_v * self._sech2(x, t)

    def p(self, x, t):
        eta = self._eta(x, t)
        return (3.0 * self.nu / (4.0 * self.omn2**1.5)) * (1.0 / np.cosh(eta))**2 * np.tanh(eta)

    def conserved_quantity(self, x_grid, t_val):
        eta = self._eta(x_grid, t_val)
        return np_trapz(self.amp_u**2 * (1.0 / np.cosh(eta))**4, x_grid)


kgs = KGSExactSolution(nu=NU_0, x0=0.0)
C_exact = kgs.conserved_quantity(np.linspace(X_MIN, X_MAX, 5000), 0.0)
print(f"C_exact = {C_exact:.6f}")


# =============================================================================
# Network
# =============================================================================
class PINN_KGS(nn.Module):
    def __init__(self, n_hidden=6, n_neurons=128):
        super().__init__()
        layers = [nn.Linear(2, n_neurons), nn.Tanh()]
        for _ in range(n_hidden - 1):
            layers += [nn.Linear(n_neurons, n_neurons), nn.Tanh()]
        layers.append(nn.Linear(n_neurons, 4))
        self.net = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self):
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        out = self.net(x)
        return {'u_R': out[:, 0:1], 'u_I': out[:, 1:2], 'v': out[:, 2:3], 'p': out[:, 3:4]}


# =============================================================================
# PDE residuals
# =============================================================================
def compute_pde_residuals(model, x, t):
    inp = torch.cat([x, t], dim=1)
    pred = model(inp)
    u_R, u_I, v, p = pred['u_R'], pred['u_I'], pred['v'], pred['p']
    ones = torch.ones_like(u_R)

    u_R_x = torch.autograd.grad(u_R, x, ones, create_graph=True)[0]
    u_R_t = torch.autograd.grad(u_R, t, ones, create_graph=True)[0]
    u_I_x = torch.autograd.grad(u_I, x, ones, create_graph=True)[0]
    u_I_t = torch.autograd.grad(u_I, t, ones, create_graph=True)[0]
    v_t = torch.autograd.grad(v, t, ones, create_graph=True)[0]
    v_x = torch.autograd.grad(v, x, ones, create_graph=True)[0]
    p_t = torch.autograd.grad(p, t, ones, create_graph=True)[0]

    u_R_xx = torch.autograd.grad(u_R_x, x, ones, create_graph=True)[0]
    u_I_xx = torch.autograd.grad(u_I_x, x, ones, create_graph=True)[0]
    v_xx = torch.autograd.grad(v_x, x, ones, create_graph=True)[0]

    residuals = {
        'sr': -u_I_t + 0.5 * u_R_xx + u_R * v,
        'si':  u_R_t + 0.5 * u_I_xx + u_I * v,
        'co':  v_t - p,
        'kg':  p_t - v_xx + v - (u_R**2 + u_I**2),
    }
    return residuals, pred


# =============================================================================
# Data generation
# =============================================================================
def lhs_1d(n, lo, hi, seed):
    rng = np.random.RandomState(seed)
    iv = np.linspace(0.0, 1.0, n + 1)
    pts = rng.uniform(iv[:-1], iv[1:])
    rng.shuffle(pts)
    return np.sort(lo + pts * (hi - lo))


def lhs_2d(n, bounds, seed):
    rng = np.random.RandomState(seed)
    out = np.zeros((n, len(bounds)))
    for i, (lo, hi) in enumerate(bounds):
        iv = np.linspace(0.0, 1.0, n + 1)
        pts = rng.uniform(iv[:-1], iv[1:])
        rng.shuffle(pts)
        out[:, i] = lo + pts * (hi - lo)
    return out


def make_data(tmax, n_col=30000, n_ic=2000, n_bc=1000, seed=42, time_weights=None, time_bounds=None):
    if time_weights is not None and time_bounds is not None:
        chunks = []
        for i, (w, tl, th) in enumerate(zip(time_weights, time_bounds[:-1], time_bounds[1:])):
            n_chunk = max(int(n_col * w), 100)
            chunk = lhs_2d(n_chunk, [(X_MIN, X_MAX), (tl, th)], seed + 17 * i)
            chunks.append(chunk)
        col = np.vstack(chunks)
        np.random.RandomState(seed + 99).shuffle(col)
    else:
        col = lhs_2d(n_col, [(X_MIN, X_MAX), (0.0, tmax)], seed)

    x_col = torch.tensor(col[:, 0:1], dtype=torch.float32, device=DEVICE, requires_grad=True)
    t_col = torch.tensor(col[:, 1:2], dtype=torch.float32, device=DEVICE, requires_grad=True)

    x_ic_np = lhs_1d(n_ic, X_MIN, X_MAX, seed + 1)
    x_ic = torch.tensor(x_ic_np.reshape(-1, 1), dtype=torch.float32, device=DEVICE)
    ic_inp = torch.cat([x_ic, torch.zeros_like(x_ic)], dim=1)
    tgt_ic = {
        'u_R': torch.tensor(kgs.u_real(x_ic_np, 0.0).reshape(-1, 1), dtype=torch.float32, device=DEVICE),
        'u_I': torch.tensor(kgs.u_imag(x_ic_np, 0.0).reshape(-1, 1), dtype=torch.float32, device=DEVICE),
        'v':   torch.tensor(kgs.v(x_ic_np, 0.0).reshape(-1, 1), dtype=torch.float32, device=DEVICE),
        'p':   torch.tensor(kgs.p(x_ic_np, 0.0).reshape(-1, 1), dtype=torch.float32, device=DEVICE),
    }

    t_bc_np = lhs_1d(n_bc, 0.0, tmax, seed + 2)
    x_bc_np = np.concatenate([np.full(n_bc, X_MIN), np.full(n_bc, X_MAX)])
    t_bc_all = np.concatenate([t_bc_np, t_bc_np])
    x_bc = torch.tensor(x_bc_np.reshape(-1, 1), dtype=torch.float32, device=DEVICE)
    t_bc = torch.tensor(t_bc_all.reshape(-1, 1), dtype=torch.float32, device=DEVICE)
    tgt_bc = {
        'u_R': torch.tensor(kgs.u_real(x_bc_np, t_bc_all).reshape(-1, 1), dtype=torch.float32, device=DEVICE),
        'u_I': torch.tensor(kgs.u_imag(x_bc_np, t_bc_all).reshape(-1, 1), dtype=torch.float32, device=DEVICE),
        'v':   torch.tensor(kgs.v(x_bc_np, t_bc_all).reshape(-1, 1), dtype=torch.float32, device=DEVICE),
        'p':   torch.tensor(kgs.p(x_bc_np, t_bc_all).reshape(-1, 1), dtype=torch.float32, device=DEVICE),
    }

    return {
        'xc': x_col,
        'tc': t_col,
        'ic_inp': ic_inp,
        'tic': tgt_ic,
        'xbc': x_bc,
        'tbc': t_bc,
        'tbc_d': tgt_bc,
        'n_col': len(col),
        'tmax': tmax,
    }


# =============================================================================
# Loss
# =============================================================================
def compute_loss(model, data, beta_cons=0.0, n_cons_times=5):
    residuals, _ = compute_pde_residuals(model, data['xc'], data['tc'])
    L_pde = sum(torch.mean(r**2) for r in residuals.values())

    pred_ic = model(data['ic_inp'])
    L_ic = sum(torch.mean((pred_ic[k] - data['tic'][k])**2) for k in pred_ic)

    pred_bc = model(torch.cat([data['xbc'], data['tbc']], dim=1))
    L_bc = sum(torch.mean((pred_bc[k] - data['tbc_d'][k])**2) for k in pred_bc)

    L_cons = torch.tensor(0.0, device=DEVICE)
    if beta_cons > 0:
        cons_times = np.linspace(0.0, data['tmax'], n_cons_times + 2)[1:-1]
        xq = torch.linspace(X_MIN, X_MAX, 500, device=DEVICE).unsqueeze(1)
        dx = (X_MAX - X_MIN) / 499
        viol = torch.tensor(0.0, device=DEVICE)
        for tc in cons_times:
            tq = torch.full_like(xq, float(tc))
            pred = model(torch.cat([xq, tq], dim=1))
            Ct = torch_trapz((pred['u_R']**2 + pred['u_I']**2).squeeze(), dx=dx)
            viol += (Ct - C_exact)**2
        L_cons = beta_cons * viol / max(len(cons_times), 1)

    total = L_pde + L_ic + L_bc + L_cons
    return total, {
        'total': total.item(),
        'pde': L_pde.item(),
        'ic': L_ic.item(),
        'bc': L_bc.item(),
        'cons': L_cons.item(),
    }


# =============================================================================
# Evaluation
# =============================================================================
def evaluate_at_time(model, t_val, n_points=1000):
    model.eval()
    x_np = np.linspace(X_MIN, X_MAX, n_points)
    with torch.no_grad():
        xt = torch.tensor(x_np.reshape(-1, 1), dtype=torch.float32, device=DEVICE)
        tt = torch.full_like(xt, float(t_val))
        pred = model(torch.cat([xt, tt], dim=1))

    preds = {k: pred[k].detach().cpu().numpy().flatten() for k in ['u_R', 'u_I', 'v', 'p']}
    exact = {
        'u_R': kgs.u_real(x_np, t_val),
        'u_I': kgs.u_imag(x_np, t_val),
        'v':   kgs.v(x_np, t_val),
        'p':   kgs.p(x_np, t_val),
    }

    metrics = {}
    for key in preds:
        diff = preds[key] - exact[key]
        metrics[f'{key}_Linf'] = float(np.max(np.abs(diff)))
        metrics[f'{key}_L2'] = float(np.sqrt(np.mean(diff**2)))

    return metrics


def compute_conservation(model, t_val, n_points=3000):
    model.eval()
    with torch.no_grad():
        xq = torch.linspace(X_MIN, X_MAX, n_points, device=DEVICE).unsqueeze(1)
        tq = torch.full_like(xq, float(t_val))
        pred = model(torch.cat([xq, tq], dim=1))
        uabs2 = pred['u_R']**2 + pred['u_I']**2
        dx = (X_MAX - X_MIN) / (n_points - 1)
        Ct = torch_trapz(uabs2.squeeze(), dx=dx).item()
    drift = abs(Ct - C_exact) / abs(C_exact)
    return Ct, drift


def full_evaluation(model, label, eval_times=(0.0, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0)):
    print("\n" + "=" * 110)
    print(f"EVAL: {label}")
    print("=" * 110)
    print(f"{'t':<6} {'uR_Linf':<12} {'uR_L2':<12} {'uI_Linf':<12} {'uI_L2':<12} {'v_Linf':<12} {'v_L2':<12} {'C(t)':<12} {'dC/C':<12}")
    print("-" * 110)

    out = {}
    drifts = []
    for t in eval_times:
        err = evaluate_at_time(model, t)
        Ct, drift = compute_conservation(model, t)
        drifts.append(drift)
        out[t] = {'errors': err, 'C': Ct, 'drift': drift}
        print(f"{t:<6.1f} {err['u_R_Linf']:<12.2e} {err['u_R_L2']:<12.2e} {err['u_I_Linf']:<12.2e} {err['u_I_L2']:<12.2e} {err['v_Linf']:<12.2e} {err['v_L2']:<12.2e} {Ct:<12.6f} {drift:<12.2e}")

    out['summary'] = {
        'max_conservation_drift': float(max(drifts)),
    }

    print(f"  Max drift:       {max(drifts):.2e}")

    return out


# =============================================================================
# Training helpers
# =============================================================================
def load_checkpoint_into_model(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Required checkpoint not found: {path}")
    ckpt = torch.load(path, map_location=DEVICE, weights_only=False)
    model = PINN_KGS().to(DEVICE)
    model.load_state_dict(ckpt['model_state_dict'])
    return model, ckpt


def train_epochs(model, data, epochs, lr, beta_cons=0.0, print_every=2000, ckpt_name=None):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=max(lr * 0.05, 1e-6))

    best_state = deepcopy(model.state_dict())
    best_loss = float('inf')
    history = []
    t0 = time()

    model.train()
    for ep in range(1, epochs + 1):
        optimizer.zero_grad(set_to_none=True)
        loss, ld = compute_loss(model, data, beta_cons=beta_cons)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        if loss.item() < best_loss:
            best_loss = loss.item()
            best_state = deepcopy(model.state_dict())

        if ep == 1 or ep % print_every == 0 or ep == epochs:
            msg = (f"Ep {ep:6d}/{epochs} | L={ld['total']:.2e} | PDE={ld['pde']:.2e} | "
                   f"IC={ld['ic']:.2e} | BC={ld['bc']:.2e} | Cons={ld['cons']:.2e} | "
                   f"lr={scheduler.get_last_lr()[0]:.1e}")
            print(msg)
            history.append({'epoch': ep, **ld, 'lr': scheduler.get_last_lr()[0]})

    model.load_state_dict(best_state)
    elapsed = time() - t0
    print(f"Training complete: {elapsed/60:.1f} min | best loss={best_loss:.2e}")

    if ckpt_name:
        torch.save({
            'model_state_dict': model.state_dict(),
            'best_loss': best_loss,
            'epochs': epochs,
            'lr': lr,
            'beta_cons': beta_cons,
            'history': history,
        }, ckpt_name)
        print(f"Saved: {ckpt_name}")

    return model, {'best_loss': best_loss, 'elapsed_sec': elapsed, 'history': history}


def run_time_march(model, stage_specs, label, ckpt_prefix, seed_offset=0):
    meta = {'stages': []}
    for i, spec in enumerate(stage_specs, start=1):
        print("\n" + "#" * 80)
        print(f"{label} | Stage {i}: {spec['name']}")
        print("#" * 80)
        data = make_data(
            tmax=spec['tmax'],
            n_col=spec.get('n_col', 30000),
            n_ic=spec.get('n_ic', 2000),
            n_bc=spec.get('n_bc', 1000),
            seed=SEED + seed_offset + i,
            time_weights=spec.get('time_weights'),
            time_bounds=spec.get('time_bounds'),
        )
        model, train_info = train_epochs(
            model=model,
            data=data,
            epochs=spec['epochs'],
            lr=spec['lr'],
            beta_cons=spec.get('beta_cons', 0.0),
            print_every=spec.get('print_every', max(1000, spec['epochs'] // 5)),
            ckpt_name=f"{ckpt_prefix}_stage{i}.pt",
        )
        meta['stages'].append({**spec, **train_info})
    return model, meta


# =============================================================================
# Plotting / serialization
# =============================================================================
def sanitize_for_json(obj):
    if isinstance(obj, dict):
        return {str(k): sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [sanitize_for_json(v) for v in obj]
    if isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    return obj


def plot_ablation_summary(results, filename='ablation_summary_updated.png'):
    labels = list(results.keys())
    x = np.arange(len(labels))

    linf_05 = [results[k][0.5]['errors']['u_R_Linf'] + results[k][0.5]['errors']['u_I_Linf'] + results[k][0.5]['errors']['v_Linf'] for k in labels]
    linf_10 = [results[k][1.0]['errors']['u_R_Linf'] + results[k][1.0]['errors']['u_I_Linf'] + results[k][1.0]['errors']['v_Linf'] for k in labels]
    drift = [results[k]['summary']['max_conservation_drift'] for k in labels]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))

    axes[0].plot(x, linf_05, marker='o', label='sum Linf @ t=0.5')
    axes[0].plot(x, linf_10, marker='s', label='sum Linf @ t=1.0')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, rotation=45)
    axes[0].set_yscale('log')
    axes[0].set_title('Error summary')
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].bar(x, drift)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=45)
    axes[1].set_yscale('log')
    axes[1].set_title('Max conservation drift')
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"Saved: {filename}")


# =============================================================================
# Main execution
# =============================================================================
def main():
    all_results = {}

    # ---------- A0: baseline evaluation ----------
    print("\n" + "=" * 80)
    print("A0: Phase 1 baseline (evaluation only)")
    print("=" * 80)
    model_a0, ckpt0 = load_checkpoint_into_model(PHASE1_CKPT)
    all_results['A0'] = full_evaluation(model_a0, 'A0 baseline')

    # ---------- A1: conservation only ----------
    print("\n" + "=" * 80)
    print("A1: Conservation only")
    print("=" * 80)
    model_a1, _ = load_checkpoint_into_model(PHASE1_CKPT)
    data_a1 = make_data(tmax=6.0, n_col=30000, n_ic=2000, n_bc=1000, seed=SEED + 10)
    model_a1, meta_a1 = train_epochs(model_a1, data_a1, epochs=20000, lr=1e-4, beta_cons=5.0,
                                     print_every=4000, ckpt_name='ablation_A1_conservation_only.pt')
    all_results['A1'] = full_evaluation(model_a1, 'A1 conservation only')
    all_results['A1']['train'] = meta_a1

    # ---------- A2: time-march only ----------
    print("\n" + "=" * 80)
    print("A2: Time-march only")
    print("=" * 80)
    model_a2, _ = load_checkpoint_into_model(PHASE1_CKPT)
    stages_a2 = [
        {'name': '[0,2]', 'tmax': 2.0, 'epochs': 12000, 'lr': 1e-4},
        {'name': '[0,4]', 'tmax': 4.0, 'epochs': 12000, 'lr': 5e-5},
        {'name': '[0,6]', 'tmax': 6.0, 'epochs': 15000, 'lr': 3e-5},
    ]
    model_a2, meta_a2 = run_time_march(model_a2, stages_a2, 'A2 time-march only', 'ablation_A2_time_march', seed_offset=20)
    all_results['A2'] = full_evaluation(model_a2, 'A2 time-march only')
    all_results['A2']['train'] = meta_a2

    # ---------- A3: time-march + conservation ----------
    print("\n" + "=" * 80)
    print("A3: Time-march + conservation")
    print("=" * 80)
    model_a3, _ = load_checkpoint_into_model(PHASE1_CKPT)
    stages_a3 = [
        {'name': '[0,2] + cons', 'tmax': 2.0, 'epochs': 12000, 'lr': 1e-4, 'beta_cons': 5.0},
        {'name': '[0,4] + cons', 'tmax': 4.0, 'epochs': 12000, 'lr': 5e-5, 'beta_cons': 5.0},
        {'name': '[0,6] + cons', 'tmax': 6.0, 'epochs': 15000, 'lr': 3e-5, 'beta_cons': 5.0},
    ]
    model_a3, meta_a3 = run_time_march(model_a3, stages_a3, 'A3 time-march + conservation', 'ablation_A3_tm_cons', seed_offset=40)
    all_results['A3'] = full_evaluation(model_a3, 'A3 time-march + conservation')
    all_results['A3']['train'] = meta_a3

    # ---------- A4: weighted collocation / Run D style ----------
    print("\n" + "=" * 80)
    print("A4: Time-march + conservation + weighted collocation")
    print("=" * 80)
    model_a4, _ = load_checkpoint_into_model(PHASE1_CKPT)
    stages_a4 = [
        {'name': '[0,2] + cons', 'tmax': 2.0, 'epochs': 12000, 'lr': 1e-4, 'beta_cons': 5.0},
        {'name': '[0,4] weighted 60/40 + cons', 'tmax': 4.0, 'epochs': 12000, 'lr': 5e-5,
         'beta_cons': 5.0, 'time_weights': [0.6, 0.4], 'time_bounds': [0.0, 2.0, 4.0]},
        {'name': '[0,6] weighted 50/30/20 + cons', 'tmax': 6.0, 'epochs': 15000, 'lr': 3e-5,
         'beta_cons': 5.0, 'time_weights': [0.5, 0.3, 0.2], 'time_bounds': [0.0, 2.0, 4.0, 6.0]},
    ]
    model_a4, meta_a4 = run_time_march(model_a4, stages_a4, 'A4 weighted collocation', 'ablation_A4_weighted', seed_offset=60)
    all_results['A4'] = full_evaluation(model_a4, 'A4 weighted collocation')
    all_results['A4']['train'] = meta_a4

    # ---------- A5: final full pipeline result from existing checkpoint ----------
    print("\n" + "=" * 80)
    print("A5: Full final pipeline result (evaluation only)")
    print("=" * 80)
    model_a5, ckpt5 = load_checkpoint_into_model(FINAL_CKPT)
    all_results['A5'] = full_evaluation(model_a5, 'A5 final pipeline / phase4b_final_checkpoint')
    all_results['A5']['checkpoint_meta'] = {
        'path': FINAL_CKPT,
        'keys': list(ckpt5.keys()) if isinstance(ckpt5, dict) else None,
    }

    # ---------- final summary ----------
    print("\n" + "=" * 95)
    print("UPDATED ABLATION GRAND SUMMARY")
    print("=" * 95)
    print(f"{'Cfg':<6} {'t=0.5 sum Linf':<16} {'t=1.0 sum Linf':<16} {'Max drift':<12}")
    print("-" * 95)
    for cfg in ['A0', 'A1', 'A2', 'A3', 'A4', 'A5']:
        r = all_results[cfg]
        s05 = r[0.5]['errors']['u_R_Linf'] + r[0.5]['errors']['u_I_Linf'] + r[0.5]['errors']['v_Linf']
        s10 = r[1.0]['errors']['u_R_Linf'] + r[1.0]['errors']['u_I_Linf'] + r[1.0]['errors']['v_Linf']
        print(f"{cfg:<6} {s05:<16.2e} {s10:<16.2e} {r['summary']['max_conservation_drift']:<12.2e}")

    best_cfg_linf = min(['A0', 'A1', 'A2', 'A3', 'A4', 'A5'],
                        key=lambda c: all_results[c][0.5]['errors']['u_R_Linf'] + all_results[c][0.5]['errors']['u_I_Linf'] + all_results[c][0.5]['errors']['v_Linf'])
    print(f"\nBest by t=0.5 Linf sum: {best_cfg_linf}")

    with open('ablation_results_updated.json', 'w', encoding='utf-8') as f:
        json.dump(sanitize_for_json(all_results), f, indent=2)
    print("Saved: ablation_results_updated.json")

    plot_ablation_summary(all_results)


if __name__ == '__main__':
    main()
