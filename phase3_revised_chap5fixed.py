#!/usr/bin/env python3
"""
================================================================================
Phase 3 (Revised): Fine-Tuning from Phase 1 Baseline
================================================================================

LESSON LEARNED: Phase 3 v1 failed because it:
  1. Started from Phase 0 (throwing away Phase 1 PDE learning)
  2. Used causal training (wrong tool; Phase 1 CAN propagate)
  3. Combined too many techniques at full learning rate

REVISED STRATEGY:
  - Start from Phase 1 checkpoint (already Linf ~ 1e-3 at t=1)
  - Fine-tune with LOW learning rate (1e-4, not 1e-3)
  - Apply techniques ONE AT A TIME as ablations:
      Run A: Conservation penalty only (soft quadratic, no Lagrange)
      Run B: Time-marching (extend domain progressively)
      Run C: Combined best of A + B
  - Each run is short (15-20K epochs) since we are refining, not training

Target: conservation < 1e-4, stable to t=6
"""

# =============================================================================
# Cell 1: Imports & Device
# =============================================================================

import os
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from time import time

np_trapz = np.trapezoid if hasattr(np, 'trapezoid') else np.trapz
torch_trapz = torch.trapezoid if hasattr(torch, 'trapezoid') else torch.trapz

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
if device.type == 'cuda':
    print(f"GPU: {torch.cuda.get_device_name(0)}")


# =============================================================================
# Cell 2: KGS Exact Solution
# =============================================================================

class KGSExactSolution:
    def __init__(self, nu=0.8, x0=0.0):
        assert abs(nu) < 1.0
        self.nu = nu
        self.x0 = x0
        self.one_minus_nu2 = 1.0 - nu**2
        self.sqrt_factor = np.sqrt(self.one_minus_nu2)
        self.amp_u = (3.0 * np.sqrt(2.0)) / (4.0 * self.sqrt_factor)
        self.amp_v = 3.0 / (4.0 * self.one_minus_nu2)

    def _eta(self, x, t):
        return (x - self.nu * t - self.x0) / (2.0 * self.sqrt_factor)

    def _sech(self, z):
        return 1.0 / np.cosh(z)

    def _phase(self, x, t):
        num = 1.0 - self.nu**2 + self.nu**4
        den = 2.0 * self.one_minus_nu2
        return self.nu * x + (num / den) * t

    def u_real(self, x, t):
        eta = self._eta(x, t)
        return self.amp_u * self._sech(eta)**2 * np.cos(self._phase(x, t))

    def u_imag(self, x, t):
        eta = self._eta(x, t)
        return self.amp_u * self._sech(eta)**2 * np.sin(self._phase(x, t))

    def v(self, x, t):
        return self.amp_v * self._sech(self._eta(x, t))**2

    def p(self, x, t):
        eta = self._eta(x, t)
        coeff = 3.0 * self.nu / (4.0 * self.one_minus_nu2**1.5)
        return coeff * self._sech(eta)**2 * np.tanh(eta)

    def u_abs_squared(self, x, t):
        return self.amp_u**2 * self._sech(self._eta(x, t))**4

    def conserved_quantity(self, x_grid, t_val):
        return np_trapz(self.u_abs_squared(x_grid, t_val), x_grid)


# =============================================================================
# Cell 3: Network Architecture (Identical to Phase 0/1)
# =============================================================================

class PINN_KGS(nn.Module):
    def __init__(self, n_hidden=6, n_neurons=128):
        super().__init__()
        layers = []
        layers.append(nn.Linear(2, n_neurons))
        layers.append(nn.Tanh())
        for _ in range(n_hidden - 1):
            layers.append(nn.Linear(n_neurons, n_neurons))
            layers.append(nn.Tanh())
        layers.append(nn.Linear(n_neurons, 4))
        self.network = nn.Sequential(*layers)
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.network:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        out = self.network(x)
        return {
            'u_R': out[:, 0:1],
            'u_I': out[:, 1:2],
            'v':   out[:, 2:3],
            'p':   out[:, 3:4],
        }


# =============================================================================
# Cell 4: PDE Residuals (Corrected)
# =============================================================================

def compute_pde_residuals(model, x, t):
    """
    Corrected KGS residuals:
        R1: -u_I_t + 0.5*u_R_xx + u_R*v = 0
        R2:  u_R_t + 0.5*u_I_xx + u_I*v = 0
        R3:  v_t - p = 0
        R4:  p_t - v_xx + v - |u|^2 = 0
    """
    inp = torch.cat([x, t], dim=1)
    pred = model(inp)
    u_R, u_I, v, p = pred['u_R'], pred['u_I'], pred['v'], pred['p']
    ones = torch.ones_like(u_R)

    u_R_x = torch.autograd.grad(u_R, x, grad_outputs=ones, create_graph=True)[0]
    u_R_t = torch.autograd.grad(u_R, t, grad_outputs=ones, create_graph=True)[0]
    u_I_x = torch.autograd.grad(u_I, x, grad_outputs=ones, create_graph=True)[0]
    u_I_t = torch.autograd.grad(u_I, t, grad_outputs=ones, create_graph=True)[0]
    v_t   = torch.autograd.grad(v,   t, grad_outputs=ones, create_graph=True)[0]
    v_x   = torch.autograd.grad(v,   x, grad_outputs=ones, create_graph=True)[0]
    p_t   = torch.autograd.grad(p,   t, grad_outputs=ones, create_graph=True)[0]

    u_R_xx = torch.autograd.grad(u_R_x, x, grad_outputs=ones, create_graph=True)[0]
    u_I_xx = torch.autograd.grad(u_I_x, x, grad_outputs=ones, create_graph=True)[0]
    v_xx   = torch.autograd.grad(v_x,   x, grad_outputs=ones, create_graph=True)[0]

    return {
        'R_schrod_re': -u_I_t + 0.5 * u_R_xx + u_R * v,
        'R_schrod_im':  u_R_t + 0.5 * u_I_xx + u_I * v,
        'R_compat':     v_t - p,
        'R_kg':         p_t - v_xx + v - (u_R**2 + u_I**2),
    }, pred


# =============================================================================
# Cell 5: Configuration
# =============================================================================

NU_0 = 0.8
X_0 = 0.0
X_MIN, X_MAX = -10.0, 10.0
T_MAX = 6.0

kgs = KGSExactSolution(nu=NU_0, x0=X_0)
C_exact = kgs.conserved_quantity(np.linspace(X_MIN, X_MAX, 5000), 0.0)
print(f"C_exact = {C_exact:.6f}")


# =============================================================================
# Cell 6: Data Generation
# =============================================================================

def latin_hypercube_2d(n, bounds, seed=None):
    rng = np.random.RandomState(seed)
    d = len(bounds)
    result = np.zeros((n, d))
    for i in range(d):
        intervals = np.linspace(0, 1, n + 1)
        pts = rng.uniform(intervals[:-1], intervals[1:])
        rng.shuffle(pts)
        result[:, i] = bounds[i][0] + pts * (bounds[i][1] - bounds[i][0])
    return result


def latin_hypercube_1d(n, lo, hi, seed=None):
    rng = np.random.RandomState(seed)
    intervals = np.linspace(0, 1, n + 1)
    pts = rng.uniform(intervals[:-1], intervals[1:])
    rng.shuffle(pts)
    return np.sort(lo + pts * (hi - lo))


def generate_training_data(t_max, n_colloc=30000, n_ic=2000, n_bc=1000, seed=42):
    """Generate training data for a given time domain [0, t_max]."""

    col = latin_hypercube_2d(n_colloc, [(X_MIN, X_MAX), (0.0, t_max)], seed=seed)
    x_col = torch.tensor(col[:, 0:1], dtype=torch.float32, device=device, requires_grad=True)
    t_col = torch.tensor(col[:, 1:2], dtype=torch.float32, device=device, requires_grad=True)

    x_ic_np = latin_hypercube_1d(n_ic, X_MIN, X_MAX, seed=seed + 1)
    x_ic = torch.tensor(x_ic_np.reshape(-1, 1), dtype=torch.float32, device=device)
    t_ic = torch.zeros_like(x_ic)
    ic_inp = torch.cat([x_ic, t_ic], dim=1)
    targets_ic = {
        'u_R': torch.tensor(kgs.u_real(x_ic_np, 0.0).reshape(-1, 1), dtype=torch.float32, device=device),
        'u_I': torch.tensor(kgs.u_imag(x_ic_np, 0.0).reshape(-1, 1), dtype=torch.float32, device=device),
        'v':   torch.tensor(kgs.v(x_ic_np, 0.0).reshape(-1, 1), dtype=torch.float32, device=device),
        'p':   torch.tensor(kgs.p(x_ic_np, 0.0).reshape(-1, 1), dtype=torch.float32, device=device),
    }

    t_bc_np = latin_hypercube_1d(n_bc, 0.0, t_max, seed=seed + 2)
    x_bc_np = np.concatenate([np.full(n_bc, X_MIN), np.full(n_bc, X_MAX)])
    t_bc_np_all = np.concatenate([t_bc_np, t_bc_np])
    x_bc = torch.tensor(x_bc_np.reshape(-1, 1), dtype=torch.float32, device=device)
    t_bc = torch.tensor(t_bc_np_all.reshape(-1, 1), dtype=torch.float32, device=device)
    targets_bc = {
        'u_R': torch.tensor(kgs.u_real(x_bc_np, t_bc_np_all).reshape(-1, 1), dtype=torch.float32, device=device),
        'u_I': torch.tensor(kgs.u_imag(x_bc_np, t_bc_np_all).reshape(-1, 1), dtype=torch.float32, device=device),
        'v':   torch.tensor(kgs.v(x_bc_np, t_bc_np_all).reshape(-1, 1), dtype=torch.float32, device=device),
        'p':   torch.tensor(kgs.p(x_bc_np, t_bc_np_all).reshape(-1, 1), dtype=torch.float32, device=device),
    }

    return {
        'x_col': x_col, 't_col': t_col,
        'ic_inp': ic_inp, 'targets_ic': targets_ic,
        'x_bc': x_bc, 't_bc': t_bc, 'targets_bc': targets_bc,
        'n_colloc': n_colloc, 'n_ic': n_ic, 'n_bc': 2 * n_bc,
        't_max': t_max,
    }


# =============================================================================
# Cell 7: Evaluation Utilities
# =============================================================================

def evaluate_at_time(model, kgs_exact, t_val, n_points=1000):
    model.eval()
    x_np = np.linspace(X_MIN, X_MAX, n_points)
    with torch.no_grad():
        x_t = torch.tensor(x_np.reshape(-1, 1), dtype=torch.float32, device=device)
        t_t = torch.full_like(x_t, t_val)
        pred = model(torch.cat([x_t, t_t], dim=1))
    preds = {k: pred[k].cpu().numpy().flatten() for k in ['u_R', 'u_I', 'v', 'p']}
    exact = {
        'u_R': kgs_exact.u_real(x_np, t_val),
        'u_I': kgs_exact.u_imag(x_np, t_val),
        'v': kgs_exact.v(x_np, t_val),
        'p': kgs_exact.p(x_np, t_val),
    }
    errors = {}
    for key in preds:
        diff = np.abs(preds[key] - exact[key])
        errors[f'{key}_Linf'] = float(np.max(diff))
        errors[f'{key}_L2'] = float(np.sqrt(np.mean(diff**2)))
    return {'x': x_np, 'preds': preds, 'exact': exact, 'errors': errors}


def compute_conserved_quantity(model, t_val, n_points=3000):
    model.eval()
    with torch.no_grad():
        x_q = torch.linspace(X_MIN, X_MAX, n_points, device=device).unsqueeze(1)
        t_q = torch.full_like(x_q, t_val)
        pred = model(torch.cat([x_q, t_q], dim=1))
        u_abs2 = pred['u_R']**2 + pred['u_I']**2
        dx = (X_MAX - X_MIN) / (n_points - 1)
        return torch_trapz(u_abs2.squeeze(), dx=dx).item()


def full_evaluation(model, kgs_exact, label="", eval_times=None):
    """Comprehensive evaluation at multiple times."""
    if eval_times is None:
        eval_times = [0.0, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    results = {}
    print(f"\n{'='*70}")
    print(f"EVALUATION: {label}")
    print(f"{'='*70}")
    print(f"{'t':<6} {'Linf(uR)':<12} {'Linf(uI)':<12} {'Linf(v)':<12} {'C(t)':<12} {'dC/C':<12}")
    print("-" * 66)
    for t_val in eval_times:
        res = evaluate_at_time(model, kgs_exact, t_val)
        C = compute_conserved_quantity(model, t_val)
        drift = abs(C - C_exact) / C_exact
        results[t_val] = {'errors': res['errors'], 'C': C, 'drift': drift}
        print(f"{t_val:<6.1f} {res['errors']['u_R_Linf']:<12.2e} "
              f"{res['errors']['u_I_Linf']:<12.2e} "
              f"{res['errors']['v_Linf']:<12.2e} "
              f"{C:<12.6f} {drift:<12.2e}")


    return results


# =============================================================================
# Cell 8: Loss Functions
# =============================================================================

def compute_loss_standard(model, data):
    """
    Standard PDE + IC + BC loss (same as Phase 1 with ramp=1).
    No fancy techniques. Clean and reliable.
    """
    residuals, _ = compute_pde_residuals(model, data['x_col'], data['t_col'])

    L_PDE = sum(torch.mean(r**2) for r in residuals.values())

    pred_ic = model(data['ic_inp'])
    L_IC = sum(torch.mean((pred_ic[k] - data['targets_ic'][k])**2) for k in pred_ic)

    bc_inp = torch.cat([data['x_bc'], data['t_bc']], dim=1)
    pred_bc = model(bc_inp)
    L_BC = sum(torch.mean((pred_bc[k] - data['targets_bc'][k])**2) for k in pred_bc)

    loss = L_PDE + L_IC + L_BC

    return loss, {
        'total': loss.item(), 'pde': L_PDE.item(),
        'ic': L_IC.item(), 'bc': L_BC.item(),
    }


def compute_loss_with_conservation(model, data, beta_cons=1.0,
                                   n_cons_times=5, C_exact_val=5.0):
    """
    Standard loss + soft conservation penalty.

    L = L_PDE + L_IC + L_BC + beta * mean_t[ (C_pinn(t) - C_exact)^2 ]

    No Lagrange multiplier. Just a simple quadratic penalty.
    beta should be tuned: too small = no effect, too large = fights PDE.
    """
    # Standard loss
    residuals, _ = compute_pde_residuals(model, data['x_col'], data['t_col'])
    L_PDE = sum(torch.mean(r**2) for r in residuals.values())

    pred_ic = model(data['ic_inp'])
    L_IC = sum(torch.mean((pred_ic[k] - data['targets_ic'][k])**2) for k in pred_ic)

    bc_inp = torch.cat([data['x_bc'], data['t_bc']], dim=1)
    pred_bc = model(bc_inp)
    L_BC = sum(torch.mean((pred_bc[k] - data['targets_bc'][k])**2) for k in pred_bc)

    # Conservation penalty at multiple time slices
    t_max = data['t_max']
    cons_times = np.linspace(0, t_max, n_cons_times + 2)[1:-1]  # Exclude endpoints
    x_q = torch.linspace(X_MIN, X_MAX, 500, device=device).unsqueeze(1)
    dx = (X_MAX - X_MIN) / 499

    violation = torch.tensor(0.0, device=device)
    C_mean = 0.0
    for t_c in cons_times:
        t_q = torch.full_like(x_q, float(t_c))
        pred_q = model(torch.cat([x_q, t_q], dim=1))
        u_abs2 = pred_q['u_R']**2 + pred_q['u_I']**2
        C_t = torch_trapz(u_abs2.squeeze(), dx=dx)
        violation = violation + (C_t - C_exact_val)**2
        C_mean += C_t.item()

    L_cons = beta_cons * violation / len(cons_times)
    C_mean /= len(cons_times)

    loss = L_PDE + L_IC + L_BC + L_cons

    return loss, {
        'total': loss.item(), 'pde': L_PDE.item(),
        'ic': L_IC.item(), 'bc': L_BC.item(),
        'cons': L_cons.item(), 'C_mean': C_mean,
    }


# =============================================================================
# Cell 9: Generic Training Loop
# =============================================================================

def train(model, data, kgs_exact, loss_fn, loss_fn_kwargs,
          n_epochs=20000, lr=1e-4, lr_min=1e-6,
          eval_every=2000, print_every=500, label=""):
    """
    Simple, clean training loop.

    Parameters
    ----------
    loss_fn : callable(model, data, **kwargs) -> (loss_tensor, loss_dict)
    loss_fn_kwargs : dict of extra arguments to loss_fn
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=n_epochs, eta_min=lr_min
    )

    history = {
        'epoch': [], 'loss_total': [], 'loss_pde': [],
        'loss_ic': [], 'loss_bc': [],
        'eval_epochs': [],
        'Linf_uR_t05': [], 'Linf_uI_t05': [], 'Linf_v_t05': [],
        'Linf_uR_t1': [], 'Linf_uI_t1': [], 'Linf_v_t1': [],
    }

    print(f"\n{'='*80}")
    print(f"TRAINING: {label}")
    print(f"Epochs: {n_epochs} | LR: {lr} -> {lr_min}")
    print(f"Domain: t in [0, {data['t_max']}] | Colloc: {data['n_colloc']}")
    print(f"{'='*80}")

    t_start = time()
    model.train()

    for epoch in range(1, n_epochs + 1):
        optimizer.zero_grad()
        loss, ld = loss_fn(model, data, **loss_fn_kwargs)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        if epoch % print_every == 0 or epoch == 1:
            current_lr = optimizer.param_groups[0]['lr']
            history['epoch'].append(epoch)
            history['loss_total'].append(ld['total'])
            history['loss_pde'].append(ld['pde'])
            history['loss_ic'].append(ld['ic'])
            history['loss_bc'].append(ld['bc'])

            extra = ""
            if 'cons' in ld:
                extra = f" Cons={ld['cons']:.2e} C={ld.get('C_mean',0):.4f}"

            print(f"Ep {epoch:6d}/{n_epochs} | "
                  f"L={ld['total']:.2e} | PDE={ld['pde']:.2e} | "
                  f"IC={ld['ic']:.2e} | BC={ld['bc']:.2e}{extra} | "
                  f"lr={current_lr:.1e}")

        if epoch % eval_every == 0 or epoch == 1:
            model.eval()
            history['eval_epochs'].append(epoch)
            for t_val, sfx in [(0.5, 't05'), (1.0, 't1')]:
                res = evaluate_at_time(model, kgs_exact, t_val)
                history[f'Linf_uR_{sfx}'].append(res['errors']['u_R_Linf'])
                history[f'Linf_uI_{sfx}'].append(res['errors']['u_I_Linf'])
                history[f'Linf_v_{sfx}'].append(res['errors']['v_Linf'])

            if epoch % (eval_every * 2) == 0 or epoch == eval_every:
                e05 = evaluate_at_time(model, kgs_exact, 0.5)
                e10 = evaluate_at_time(model, kgs_exact, 1.0)
                print(f"  >> t=0.5: uR={e05['errors']['u_R_Linf']:.2e}, "
                      f"uI={e05['errors']['u_I_Linf']:.2e}, "
                      f"v={e05['errors']['v_Linf']:.2e}")
                print(f"  >> t=1.0: uR={e10['errors']['u_R_Linf']:.2e}, "
                      f"uI={e10['errors']['u_I_Linf']:.2e}, "
                      f"v={e10['errors']['v_Linf']:.2e}")
            model.train()

    elapsed = time() - t_start
    print(f"\nComplete: {elapsed:.0f}s ({elapsed/60:.1f} min)")
    return history


# =============================================================================
# Cell 10: L-BFGS Fine-Tuning
# =============================================================================

def lbfgs_finetune(model, data, loss_fn, loss_fn_kwargs,
                   n_iters=3000, lr=0.5, print_every=500):
    """L-BFGS second-stage optimizer."""
    optimizer = torch.optim.LBFGS(
        model.parameters(), lr=lr, max_iter=20,
        history_size=50, tolerance_grad=1e-9, tolerance_change=1e-11,
        line_search_fn='strong_wolfe'
    )
    losses = []
    print(f"\n{'='*60}")
    print("L-BFGS FINE-TUNING")
    print(f"{'='*60}")
    t_start = time()
    model.train()

    for i in range(1, n_iters + 1):
        def closure():
            optimizer.zero_grad()
            loss, _ = loss_fn(model, data, **loss_fn_kwargs)
            loss.backward()
            return loss

        loss = optimizer.step(closure)
        losses.append(loss.item())
        if i % print_every == 0 or i == 1:
            print(f"  L-BFGS {i:5d}/{n_iters} | Loss: {loss.item():.2e}")

    print(f"Complete: {time()-t_start:.0f}s | Final: {losses[-1]:.2e}")
    return losses


# =============================================================================
# Cell 11: Load Phase 1 Checkpoint
# =============================================================================

PHASE1_PATH = 'phase1_baseline_checkpoint.pt'

def load_phase1_model():
    """Load a fresh copy of the Phase 1 trained model."""
    model = PINN_KGS(n_hidden=6, n_neurons=128).to(device)
    if os.path.exists(PHASE1_PATH):
        ckpt = torch.load(PHASE1_PATH, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'])
        print(f"Phase 1 checkpoint loaded: {PHASE1_PATH}")
        print(f"  Final loss: {ckpt['final_loss']:.2e}, Epochs: {ckpt['epoch']}")
    else:
        raise FileNotFoundError(f"{PHASE1_PATH} not found. Run Phase 1 first.")
    return model


# Baseline evaluation
model_baseline = load_phase1_model()
print("\n--- Phase 1 Baseline (reference) ---")
baseline_results = full_evaluation(model_baseline, kgs, label="Phase 1 Baseline")


# =============================================================================
# Cell 12: RUN A - Fine-tune Phase 1 with Conservation Penalty (t_max=6)
# =============================================================================

print("\n" + "#" * 80)
print("# RUN A: Phase 1 + Conservation Penalty (beta=1.0)")
print("#" * 80)

model_A = load_phase1_model()
data_A = generate_training_data(t_max=T_MAX, n_colloc=30000, n_ic=2000, n_bc=1000, seed=SEED)

history_A = train(
    model_A, data_A, kgs,
    loss_fn=compute_loss_with_conservation,
    loss_fn_kwargs={'beta_cons': 1.0, 'n_cons_times': 5, 'C_exact_val': C_exact},
    n_epochs=20000,
    lr=1e-4, lr_min=1e-6,
    eval_every=2000, print_every=1000,
    label="Run A: Phase1 + Conservation(beta=1.0)",
)

results_A = full_evaluation(model_A, kgs, label="Run A Final")


# =============================================================================
# Cell 13: RUN B - Time-Marching (Progressive Domain Extension)
# =============================================================================

print("\n" + "#" * 80)
print("# RUN B: Time-Marching from Phase 1")
print("#   Stage 1: t=[0, 2] (15K epochs)")
print("#   Stage 2: t=[0, 4] (15K epochs)")
print("#   Stage 3: t=[0, 6] (15K epochs)")
print("#" * 80)

model_B = load_phase1_model()

time_windows = [2.0, 4.0, 6.0]
epochs_per_window = 15000

for stage, t_max_stage in enumerate(time_windows, 1):
    print(f"\n--- Stage {stage}: t_max = {t_max_stage} ---")

    # Scale collocation points with domain size
    n_col = int(30000 * t_max_stage / T_MAX)
    data_stage = generate_training_data(
        t_max=t_max_stage, n_colloc=n_col, n_ic=2000, n_bc=1000,
        seed=SEED + stage * 100
    )

    # Lower LR in later stages (refinement)
    stage_lr = 1e-4 / stage

    train(
        model_B, data_stage, kgs,
        loss_fn=compute_loss_standard,
        loss_fn_kwargs={},
        n_epochs=epochs_per_window,
        lr=stage_lr, lr_min=1e-6,
        eval_every=3000, print_every=1000,
        label=f"Run B Stage {stage}: t=[0, {t_max_stage}], lr={stage_lr:.0e}",
    )

    # Evaluate at the end of each stage
    full_evaluation(model_B, kgs, label=f"Run B after Stage {stage}")

results_B = full_evaluation(model_B, kgs, label="Run B Final (Time-Marching)")


# =============================================================================
# Cell 14: RUN C - Best of A+B: Time-Marching with Conservation
# =============================================================================

print("\n" + "#" * 80)
print("# RUN C: Time-Marching + Conservation")
print("#" * 80)

model_C = load_phase1_model()

for stage, t_max_stage in enumerate(time_windows, 1):
    print(f"\n--- Stage {stage}: t_max = {t_max_stage} ---")
    n_col = int(30000 * t_max_stage / T_MAX)
    data_stage = generate_training_data(
        t_max=t_max_stage, n_colloc=n_col, n_ic=2000, n_bc=1000,
        seed=SEED + stage * 200
    )
    stage_lr = 1e-4 / stage

    train(
        model_C, data_stage, kgs,
        loss_fn=compute_loss_with_conservation,
        loss_fn_kwargs={'beta_cons': 1.0, 'n_cons_times': 5, 'C_exact_val': C_exact},
        n_epochs=epochs_per_window,
        lr=stage_lr, lr_min=1e-6,
        eval_every=3000, print_every=1000,
        label=f"Run C Stage {stage}: t=[0, {t_max_stage}], lr={stage_lr:.0e}, beta=1.0",
    )

results_C = full_evaluation(model_C, kgs, label="Run C Final (Time-March + Conservation)")


# =============================================================================
# Cell 15: L-BFGS Polish on Best Model
# =============================================================================

# Determine which run was best at t=1.0
runs = {'A': (model_A, results_A), 'B': (model_B, results_B), 'C': (model_C, results_C)}
best_run = min(runs.keys(), key=lambda r: (
    runs[r][1][1.0]['errors']['u_R_Linf'] +
    runs[r][1][1.0]['errors']['u_I_Linf'] +
    runs[r][1][1.0]['errors']['v_Linf']
) if 1.0 in runs[r][1] else float('inf'))

print(f"\n{'#'*80}")
print(f"# L-BFGS POLISH on Run {best_run} (best at t=1)")
print(f"{'#'*80}")

best_model = runs[best_run][0]

# Use full domain data for L-BFGS
data_full = generate_training_data(t_max=T_MAX, n_colloc=30000, n_ic=2000, n_bc=1000, seed=SEED+999)

lbfgs_losses = lbfgs_finetune(
    best_model, data_full,
    loss_fn=compute_loss_with_conservation,
    loss_fn_kwargs={'beta_cons': 1.0, 'n_cons_times': 5, 'C_exact_val': C_exact},
    n_iters=3000, lr=0.5, print_every=500,
)

results_final = full_evaluation(best_model, kgs, label=f"FINAL (Run {best_run} + L-BFGS)")


# =============================================================================
# Cell 16: Visualization - Comparison Plot
# =============================================================================

plt.rcParams.update({
    'font.size': 11, 'axes.labelsize': 12, 'axes.titlesize': 13,
    'legend.fontsize': 9, 'figure.dpi': 120,
})

# Temporal error profiles
time_snapshots = np.array([0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 2.5,
                           3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0])

def get_temporal_profile(model):
    errs = {k: [] for k in ['u_R', 'u_I', 'v']}
    cons = []
    for t_val in time_snapshots:
        res = evaluate_at_time(model, kgs, t_val)
        for k in errs:
            errs[k].append(res['errors'][f'{k}_Linf'])
        cons.append(compute_conserved_quantity(model, t_val))
    return errs, cons


# Phase 1 baseline errors (from Phase 2 output, verified)
errors_p1 = {
    'u_R': [4.27e-04, 8.57e-04, 9.71e-04, 1.17e-03, 1.87e-03, 4.23e-03, 3.16e-03,
            8.06e-03, 1.90e-02, 2.01e-02, 2.09e-02, 4.13e-02, 4.99e-02, 3.53e-02, 5.73e-02],
    'u_I': [3.14e-04, 7.46e-04, 1.16e-03, 1.92e-03, 2.02e-03, 2.03e-03, 6.65e-03,
            9.89e-03, 7.05e-03, 2.22e-02, 3.43e-02, 2.86e-02, 3.75e-02, 6.06e-02, 6.14e-02],
    'v':   [1.96e-04, 3.96e-04, 6.07e-04, 1.16e-03, 2.00e-03, 3.51e-03, 7.04e-03,
            9.33e-03, 1.18e-02, 1.37e-02, 1.50e-02, 1.51e-02, 1.61e-02, 1.59e-02, 1.60e-02],
}

errors_best, cons_best = get_temporal_profile(best_model)

fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Panel 1: uR errors over time
ax = axes[0, 0]
ax.semilogy(time_snapshots, errors_p1['u_R'], 'k--o', ms=3, lw=1, label='Phase 1 Baseline', alpha=0.5)
ax.semilogy(time_snapshots, errors_best['u_R'], 'r-s', ms=4, lw=1.5, label=f'Phase 3 (Run {best_run})')
ax.set_xlabel('Time $t$')
ax.set_ylabel('$L_\\infty$ Error')
ax.set_title('Re($u$) Error vs Time')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# Panel 2: uI
ax = axes[0, 1]
ax.semilogy(time_snapshots, errors_p1['u_I'], 'k--o', ms=3, lw=1, label='Phase 1', alpha=0.5)
ax.semilogy(time_snapshots, errors_best['u_I'], 'r-s', ms=4, lw=1.5, label=f'Phase 3')
ax.set_xlabel('Time $t$')
ax.set_ylabel('$L_\\infty$ Error')
ax.set_title('Im($u$) Error vs Time')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# Panel 3: v
ax = axes[1, 0]
ax.semilogy(time_snapshots, errors_p1['v'], 'k--o', ms=3, lw=1, label='Phase 1', alpha=0.5)
ax.semilogy(time_snapshots, errors_best['v'], 'r-s', ms=4, lw=1.5, label=f'Phase 3')
ax.set_xlabel('Time $t$')
ax.set_ylabel('$L_\\infty$ Error')
ax.set_title('$v$ Error vs Time')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# Panel 4: Conservation
ax = axes[1, 1]
cons_p1 = [4.998411, None, 4.993267, None, 4.988786, None, None, None,
           None, None, None, None, None, None, 4.973992]
drift_best = [abs(c - C_exact)/C_exact for c in cons_best]
ax.semilogy(time_snapshots, drift_best, 'r-s', ms=4, lw=1.5, label=f'Phase 3 (Run {best_run})')
ax.axhline(y=1e-5, color='k', ls='--', alpha=0.5, label='Target $10^{-5}$')
ax.axhline(y=5.39e-3, color='gray', ls=':', alpha=0.5, label='Phase 1 max drift')
ax.set_xlabel('Time $t$')
ax.set_ylabel('$|\\Delta C / C|$')
ax.set_title('Conservation Drift')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

plt.suptitle('Phase 3 (Revised): Phase 1 Baseline vs Best Fine-Tuned Model',
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('phase3_revised_comparison.png', bbox_inches='tight', dpi=150)
plt.show()
print("Saved: phase3_revised_comparison.png")


# =============================================================================
# Cell 17: Solution Profiles
# =============================================================================

fig, axes = plt.subplots(4, 3, figsize=(16, 18))
plot_times = [0.5, 1.0, 3.0, 6.0]

for row, t_val in enumerate(plot_times):
    res = evaluate_at_time(best_model, kgs, t_val)
    x = res['x']
    for col, (key, ylabel) in enumerate([
        ('u_R', 'Re($u$)'), ('u_I', 'Im($u$)'), ('v', '$v$')
    ]):
        ax = axes[row, col]
        ax.plot(x, res['exact'][key], 'b-', lw=2, label='Exact')
        ax.plot(x, res['preds'][key], 'r--', lw=1.5, alpha=0.85, label='PINN')
        ax.set_title(f'{ylabel} at $t={t_val}$, '
                     f'$L_\\infty$={res["errors"][f"{key}_Linf"]:.2e}', fontsize=10)
        ax.grid(True, alpha=0.2)
        ax.set_xlim(X_MIN, X_MAX)
        if row == 3:
            ax.set_xlabel('$x$')
        if col == 0:
            ax.set_ylabel(f'$t = {t_val}$', fontsize=11, fontweight='bold')
        ax.legend(fontsize=7, loc='best')

plt.suptitle(f'Phase 3 (Run {best_run} + L-BFGS): Solution Profiles',
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('phase3_solution_profiles.png', bbox_inches='tight', dpi=150)
plt.show()
print("Saved: phase3_solution_profiles.png")


# =============================================================================
# Cell 18: Save Checkpoint & Summary
# =============================================================================

torch.save({
    'model_state_dict': best_model.state_dict(),
    'best_run': best_run,
    'results_final': {t: {k: v for k, v in r.items() if k != 'preds'}
                      for t, r in results_final.items()},
    'errors_temporal': errors_best,
    'conservation_temporal': cons_best,
    'C_exact': C_exact,
}, 'phase3_revised_checkpoint.pt')
print("\nSaved: phase3_revised_checkpoint.pt")


# Final summary table
print("\n" + "=" * 80)
print("PHASE 3 (REVISED) FINAL SUMMARY")
print("=" * 80)

print(f"\n{'t':<6} {'Comp':<6} {'Phase1':<12} {'Phase3':<12} {'Improve':<10}")
print("-" * 68)

for t_val in [0.5, 1.0]:
    t_idx = list(time_snapshots).index(t_val)
    for key in ['u_R', 'u_I', 'v']:
        p1 = errors_p1[key][t_idx]
        p3 = errors_best[key][t_idx]
        improve = p1 / p3 if p3 > 0 else float('inf')
        print(f"{t_val:<6.1f} {key:<6} {p1:<12.2e} {p3:<12.2e} {improve:<10.1f}x")
    print()

# Conservation
max_drift = max(abs(c - C_exact)/C_exact for c in cons_best)
print(f"Conservation max drift: {max_drift:.2e}")
print(f"  Phase 1 was: 5.39e-03")
print(f"  Target:      1e-05")

print(f"\nBest run: {best_run}")
print("=" * 80)
