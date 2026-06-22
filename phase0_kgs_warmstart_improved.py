#!/usr/bin/env python3
"""
================================================================================
Phase 0 (Improved): Warm-Start Pre-Training for the Coupled KGS PINN
================================================================================

What is improved here compared with the earlier Phase 0 script?
  1. Correct Chapter 5 exact solution (nu = 0.8, positive v, correct phase, correct p)
  2. Better IC sampling: global coverage + extra concentration near the soliton core
  3. Supports resuming from an existing Phase 0 checkpoint automatically
  4. Two-stage optimization:
       - Adam for coarse fitting
       - L-BFGS for final high-accuracy refinement
  5. Tracks and restores the best model state
  6. Uses stricter evaluation and clearer success criteria

This script is architecture-compatible with Phase 1 baseline scripts that expect
exactly the same 6x128 tanh network with outputs (u_R, u_I, v, p).

Reference KGS system from Chapter 5:
    i u_t + 0.5 u_xx + u v = 0
    v_tt - v_xx + v - |u|^2 = 0

Reduction of order for later phases:
    p = v_t

Warm-start objective in Phase 0:
    Fit only the t = 0 initial data for (u_R, u_I, v, p)
    before PDE residuals are activated in Phase 1.
"""

# =============================================================================
# Cell 1: Imports & Device Setup
# =============================================================================

import copy
import os
from time import time

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import torch
import torch.nn as nn

np_trapz = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
_torch_trapz = torch.trapezoid if hasattr(torch, "trapezoid") else torch.trapz

# ----------------------------- User Config -----------------------------------
SEED = 42
X_MIN, X_MAX = -10.0, 10.0
NU_0 = 0.8
X_0 = 0.0

# Sampling
N_IC_GLOBAL = 3000
N_IC_CENTER = 2000
CENTER_FOCUS_HALF_WIDTH = 3.0   # add extra IC points in [-3, 3]

# Training options
RESUME_IF_AVAILABLE = True
RESUME_CHECKPOINT_PATH = "phase0_warmstart_checkpoint.pt"
OUTPUT_CHECKPOINT_PATH = "phase0_warmstart_checkpoint.pt"

# Fresh full run settings (used if no checkpoint exists or resume=False)
ADAM_EPOCHS_FRESH = 5000
ADAM_LR_FRESH = 1e-3
ADAM_ETA_MIN_FRESH = 1e-5

# Resume refinement settings (used if checkpoint exists and resume=True)
ADAM_EPOCHS_RESUME = 3000
ADAM_LR_RESUME = 1e-4
ADAM_ETA_MIN_RESUME = 1e-6

# Final polishing stage
USE_LBFGS = True
LBFGS_MAX_ITER = 1200
LBFGS_LR = 0.8
LBFGS_HISTORY_SIZE = 100
LBFGS_TOL_GRAD = 1e-10
LBFGS_TOL_CHANGE = 1e-12

PRINT_EVERY = 100
SUCCESS_LOSS = 1e-6
SUCCESS_CONSERVATION_RELERR = 1e-4
# -----------------------------------------------------------------------------


torch.manual_seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# Determinism helpers
try:
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
except Exception:
    pass

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
if device.type == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")


# =============================================================================
# Cell 2: Exact Chapter 5 KGS Solution
# =============================================================================

class KGSExactSolution:
    """
    Chapter 5, Test Problem I exact solution.

    u(x,t) = A_u sech^2(eta) exp(i*phi)
    v(x,t) = A_v sech^2(eta)

    eta = (x - nu*t - x0) / (2*sqrt(1-nu^2))
    phi = nu*x + ((1 - nu^2 + nu^4)/(2*(1-nu^2))) * t

    with
      A_u = 3*sqrt(2) / (4*sqrt(1-nu^2))
      A_v = 3 / (4*(1-nu^2))

    Auxiliary field:
      p = v_t = 3*nu / (4*(1-nu^2)^(3/2)) * sech^2(eta) * tanh(eta)
    """

    def __init__(self, nu=0.8, x0=0.0):
        if abs(nu) >= 1.0:
            raise ValueError("nu must satisfy |nu| < 1")
        self.nu = float(nu)
        self.x0 = float(x0)
        self.one_minus_nu2 = 1.0 - self.nu**2
        self.sqrt_factor = np.sqrt(self.one_minus_nu2)
        self.amp_u = (3.0 * np.sqrt(2.0)) / (4.0 * self.sqrt_factor)
        self.amp_v = 3.0 / (4.0 * self.one_minus_nu2)

    def _eta(self, x, t):
        return (x - self.x0 - self.nu * t) / (2.0 * self.sqrt_factor)

    def _sech(self, z):
        return 1.0 / np.cosh(z)

    def _phase(self, x, t):
        return self.nu * x + ((1.0 - self.nu**2 + self.nu**4) /
                              (2.0 * self.one_minus_nu2)) * t

    def u_real(self, x, t):
        eta = self._eta(x, t)
        return self.amp_u * self._sech(eta)**2 * np.cos(self._phase(x, t))

    def u_imag(self, x, t):
        eta = self._eta(x, t)
        return self.amp_u * self._sech(eta)**2 * np.sin(self._phase(x, t))

    def v(self, x, t):
        eta = self._eta(x, t)
        return self.amp_v * self._sech(eta)**2

    def p(self, x, t):
        eta = self._eta(x, t)
        coeff = 3.0 * self.nu / (4.0 * self.one_minus_nu2**1.5)
        return coeff * self._sech(eta)**2 * np.tanh(eta)

    def u_abs_squared(self, x, t):
        eta = self._eta(x, t)
        return self.amp_u**2 * self._sech(eta)**4

    def conserved_quantity(self, x_grid, t_val):
        return np_trapz(self.u_abs_squared(x_grid, t_val), x_grid)


kgs = KGSExactSolution(nu=NU_0, x0=X_0)
C_exact = kgs.conserved_quantity(np.linspace(X_MIN, X_MAX, 5000), 0.0)
print(f"\nKGS Exact Solution (nu={NU_0}, x0={X_0})")
print(f"Conserved quantity at t=0: {C_exact:.6f}")
print(f"Soliton amplitude |u|_max: {kgs.amp_u:.6f}")
print(f"Soliton width (2*sqrt(1-nu^2)): {2*kgs.sqrt_factor:.6f}")


# =============================================================================
# Cell 3: Data Generation (Global + Focused Near Soliton Core)
# =============================================================================

def latin_hypercube_1d(n_samples, x_min, x_max, seed=None):
    rng = np.random.RandomState(seed)
    intervals = np.linspace(0.0, 1.0, n_samples + 1)
    lower = intervals[:-1]
    upper = intervals[1:]
    pts = rng.uniform(lower, upper)
    rng.shuffle(pts)
    return np.sort(x_min + pts * (x_max - x_min))


def make_phase0_points(n_global, n_center, center_half_width, seed):
    """
    Create t=0 IC points with two parts:
      1. global LHS on [-10,10]
      2. extra focused LHS on [-center_half_width, center_half_width]

    This gives the network more pressure where the sech^2 peak is sharpest.
    """
    x_global = latin_hypercube_1d(n_global, X_MIN, X_MAX, seed=seed)
    x_center = latin_hypercube_1d(n_center, -center_half_width, center_half_width, seed=seed + 17)
    x_all = np.concatenate([x_global, x_center])
    x_all.sort()
    return x_all


x_train_np = make_phase0_points(
    n_global=N_IC_GLOBAL,
    n_center=N_IC_CENTER,
    center_half_width=CENTER_FOCUS_HALF_WIDTH,
    seed=SEED,
)
N_IC_TOTAL = len(x_train_np)

t_train_np = np.zeros_like(x_train_np)
uR_exact = kgs.u_real(x_train_np, t_train_np)
uI_exact = kgs.u_imag(x_train_np, t_train_np)
v_exact = kgs.v(x_train_np, t_train_np)
p_exact = kgs.p(x_train_np, t_train_np)

x_train = torch.tensor(x_train_np, dtype=torch.float32, device=device).unsqueeze(1)
t_train = torch.tensor(t_train_np, dtype=torch.float32, device=device).unsqueeze(1)
inputs = torch.cat([x_train, t_train], dim=1)

targets = {
    "u_R": torch.tensor(uR_exact, dtype=torch.float32, device=device).unsqueeze(1),
    "u_I": torch.tensor(uI_exact, dtype=torch.float32, device=device).unsqueeze(1),
    "v": torch.tensor(v_exact, dtype=torch.float32, device=device).unsqueeze(1),
    "p": torch.tensor(p_exact, dtype=torch.float32, device=device).unsqueeze(1),
}

print("\nTraining data prepared:")
print(f"  Global IC points: {N_IC_GLOBAL}")
print(f"  Focused IC points near center: {N_IC_CENTER}")
print(f"  Total IC points: {N_IC_TOTAL}")
print(f"  Input shape: {inputs.shape}")
print(f"  Device: {inputs.device}")


# =============================================================================
# Cell 4: Network Architecture (Phase-1 Compatible)
# =============================================================================

class PINN_KGS(nn.Module):
    """
    Architecture must remain identical to the Phase 1 baseline loader.
    """

    def __init__(self, n_hidden=6, n_neurons=128):
        super().__init__()
        layers = [nn.Linear(2, n_neurons), nn.Tanh()]
        for _ in range(n_hidden - 1):
            layers += [nn.Linear(n_neurons, n_neurons), nn.Tanh()]
        layers.append(nn.Linear(n_neurons, 4))
        self.network = nn.Sequential(*layers)
        self._initialize_weights()

    def _initialize_weights(self):
        for layer in self.network:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(self, x):
        out = self.network(x)
        return {
            "u_R": out[:, 0:1],
            "u_I": out[:, 1:2],
            "v": out[:, 2:3],
            "p": out[:, 3:4],
        }


model = PINN_KGS(n_hidden=6, n_neurons=128).to(device)
n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print("\nModel Architecture:")
print("  Hidden layers: 6 x 128 neurons")
print("  Activation: tanh")
print(f"  Total trainable parameters: {n_params:,}")
print(f"  Device: {next(model.parameters()).device}")


# =============================================================================
# Cell 5: Utilities
# =============================================================================

def compute_ic_losses(model, inputs, targets):
    pred = model(inputs)
    mse = nn.MSELoss()
    loss_uR = mse(pred["u_R"], targets["u_R"])
    loss_uI = mse(pred["u_I"], targets["u_I"])
    loss_v = mse(pred["v"], targets["v"])
    loss_p = mse(pred["p"], targets["p"])
    loss_total = loss_uR + loss_uI + loss_v + loss_p
    loss_dict = {
        "total": float(loss_total.item()),
        "u_R": float(loss_uR.item()),
        "u_I": float(loss_uI.item()),
        "v": float(loss_v.item()),
        "p": float(loss_p.item()),
    }
    return loss_total, loss_dict


def compute_conserved_quantity_pinn(model, x_min, x_max, t_val, n_points=4000):
    model.eval()
    with torch.no_grad():
        x_quad = torch.linspace(x_min, x_max, n_points, device=device).unsqueeze(1)
        t_quad = torch.full_like(x_quad, float(t_val))
        pred = model(torch.cat([x_quad, t_quad], dim=1))
        u_abs2 = pred["u_R"]**2 + pred["u_I"]**2
        dx = (x_max - x_min) / (n_points - 1)
        C = _torch_trapz(u_abs2.squeeze(), dx=dx).item()
    return C


def evaluate_ic_errors(model, kgs_exact, x_eval):
    model.eval()
    with torch.no_grad():
        x_t = torch.tensor(x_eval, dtype=torch.float32, device=device).unsqueeze(1)
        t_t = torch.zeros_like(x_t)
        pred = model(torch.cat([x_t, t_t], dim=1))

    preds = {k: pred[k].detach().cpu().numpy().flatten() for k in ["u_R", "u_I", "v", "p"]}
    exact = {
        "u_R": kgs_exact.u_real(x_eval, 0.0),
        "u_I": kgs_exact.u_imag(x_eval, 0.0),
        "v": kgs_exact.v(x_eval, 0.0),
        "p": kgs_exact.p(x_eval, 0.0),
    }

    errors = {}
    for key in preds:
        diff = np.abs(preds[key] - exact[key])
        errors[f"{key}_Linf"] = float(np.max(diff))
        errors[f"{key}_L2"] = float(np.sqrt(np.mean(diff**2)))
    return preds, exact, errors


def print_loss_row(prefix, epoch, total_epochs, loss_dict, lr=None):
    msg = (
        f"{prefix} {epoch:5d}/{total_epochs} | "
        f"L_total: {loss_dict['total']:.2e} | "
        f"L_uR: {loss_dict['u_R']:.2e} | "
        f"L_uI: {loss_dict['u_I']:.2e} | "
        f"L_v: {loss_dict['v']:.2e} | "
        f"L_p: {loss_dict['p']:.2e}"
    )
    if lr is not None:
        msg += f" | lr: {lr:.1e}"
    print(msg)


# =============================================================================
# Cell 6: Optional Resume
# =============================================================================

history = {
    "adam_epoch": [],
    "adam_total": [],
    "adam_uR": [],
    "adam_uI": [],
    "adam_v": [],
    "adam_p": [],
    "adam_lr": [],
    "lbfgs_step": [],
    "lbfgs_total": [],
    "lbfgs_uR": [],
    "lbfgs_uI": [],
    "lbfgs_v": [],
    "lbfgs_p": [],
}

best_state_dict = copy.deepcopy(model.state_dict())
best_total_loss = float("inf")
start_mode = "fresh"

if RESUME_IF_AVAILABLE and os.path.exists(RESUME_CHECKPOINT_PATH):
    ckpt = torch.load(RESUME_CHECKPOINT_PATH, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    start_mode = "resume"
    print("\nExisting Phase 0 checkpoint found. Resuming refinement from it.")
    if "final_loss" in ckpt:
        print(f"  Previous final loss: {ckpt['final_loss']:.2e}")
    if "conserved_quantity_pinn" in ckpt:
        print(f"  Previous C_pinn: {ckpt['conserved_quantity_pinn']:.6f}")

    current_loss_tensor, current_loss_dict = compute_ic_losses(model, inputs, targets)
    best_total_loss = current_loss_dict["total"]
    best_state_dict = copy.deepcopy(model.state_dict())
    print(f"  Resume starting loss: {best_total_loss:.2e}")
else:
    print("\nNo previous Phase 0 checkpoint loaded. Starting fresh.")


# =============================================================================
# Cell 7: Adam Stage
# =============================================================================

def run_adam_stage(model, inputs, targets, n_epochs, lr, eta_min, print_every, history,
                   best_total_loss, best_state_dict, stage_label):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs, eta_min=eta_min)

    print("\n" + "=" * 78)
    print(f"{stage_label}: ADAM OPTIMIZATION")
    print("=" * 78)
    print(f"Epochs: {n_epochs} | LR: {lr} | eta_min: {eta_min}")
    print("-" * 78)

    t0 = time()
    model.train()

    for epoch in range(1, n_epochs + 1):
        optimizer.zero_grad()
        loss_total, loss_dict = compute_ic_losses(model, inputs, targets)
        loss_total.backward()
        optimizer.step()
        scheduler.step()

        current_lr = optimizer.param_groups[0]["lr"]
        if loss_dict["total"] < best_total_loss:
            best_total_loss = loss_dict["total"]
            best_state_dict = copy.deepcopy(model.state_dict())

        if epoch % print_every == 0 or epoch == 1 or epoch == n_epochs:
            history["adam_epoch"].append(epoch)
            history["adam_total"].append(loss_dict["total"])
            history["adam_uR"].append(loss_dict["u_R"])
            history["adam_uI"].append(loss_dict["u_I"])
            history["adam_v"].append(loss_dict["v"])
            history["adam_p"].append(loss_dict["p"])
            history["adam_lr"].append(current_lr)
            print_loss_row("Epoch", epoch, n_epochs, loss_dict, lr=current_lr)

    elapsed = time() - t0
    print("-" * 78)
    print(f"Adam stage complete in {elapsed:.1f}s")
    print(f"Best Adam loss so far: {best_total_loss:.2e}")
    print("=" * 78)

    return best_total_loss, best_state_dict


if start_mode == "resume":
    adam_epochs = ADAM_EPOCHS_RESUME
    adam_lr = ADAM_LR_RESUME
    adam_eta_min = ADAM_ETA_MIN_RESUME
    stage_name = "PHASE 0 REFINEMENT"
else:
    adam_epochs = ADAM_EPOCHS_FRESH
    adam_lr = ADAM_LR_FRESH
    adam_eta_min = ADAM_ETA_MIN_FRESH
    stage_name = "PHASE 0 WARM-START"

best_total_loss, best_state_dict = run_adam_stage(
    model=model,
    inputs=inputs,
    targets=targets,
    n_epochs=adam_epochs,
    lr=adam_lr,
    eta_min=adam_eta_min,
    print_every=PRINT_EVERY,
    history=history,
    best_total_loss=best_total_loss,
    best_state_dict=best_state_dict,
    stage_label=stage_name,
)

# Restore best state before LBFGS
model.load_state_dict(best_state_dict)


# =============================================================================
# Cell 8: L-BFGS Refinement Stage
# =============================================================================

def run_lbfgs_stage(model, inputs, targets, history, best_total_loss, best_state_dict):
    if not USE_LBFGS:
        return best_total_loss, best_state_dict

    print("\n" + "=" * 78)
    print("PHASE 0 FINAL POLISH: L-BFGS")
    print("=" * 78)
    print(
        f"max_iter: {LBFGS_MAX_ITER} | lr: {LBFGS_LR} | "
        f"history_size: {LBFGS_HISTORY_SIZE}"
    )
    print("-" * 78)

    optimizer = torch.optim.LBFGS(
        model.parameters(),
        lr=LBFGS_LR,
        max_iter=LBFGS_MAX_ITER,
        max_eval=LBFGS_MAX_ITER + 200,
        tolerance_grad=LBFGS_TOL_GRAD,
        tolerance_change=LBFGS_TOL_CHANGE,
        history_size=LBFGS_HISTORY_SIZE,
        line_search_fn="strong_wolfe",
    )

    closure_calls = {"n": 0}
    t0 = time()
    model.train()

    def closure():
        optimizer.zero_grad()
        loss_total, loss_dict = compute_ic_losses(model, inputs, targets)
        loss_total.backward()

        closure_calls["n"] += 1
        step_id = closure_calls["n"]

        if loss_dict["total"] < run_lbfgs_stage.best_total_loss:
            run_lbfgs_stage.best_total_loss = loss_dict["total"]
            run_lbfgs_stage.best_state_dict = copy.deepcopy(model.state_dict())

        if step_id % 25 == 0 or step_id == 1:
            history["lbfgs_step"].append(step_id)
            history["lbfgs_total"].append(loss_dict["total"])
            history["lbfgs_uR"].append(loss_dict["u_R"])
            history["lbfgs_uI"].append(loss_dict["u_I"])
            history["lbfgs_v"].append(loss_dict["v"])
            history["lbfgs_p"].append(loss_dict["p"])
            print_loss_row("Step ", step_id, LBFGS_MAX_ITER, loss_dict, lr=None)
        return loss_total

    run_lbfgs_stage.best_total_loss = best_total_loss
    run_lbfgs_stage.best_state_dict = copy.deepcopy(best_state_dict)

    optimizer.step(closure)

    elapsed = time() - t0
    best_total_loss = run_lbfgs_stage.best_total_loss
    best_state_dict = run_lbfgs_stage.best_state_dict
    print("-" * 78)
    print(f"L-BFGS stage complete in {elapsed:.1f}s")
    print(f"Best total loss after L-BFGS: {best_total_loss:.2e}")
    print("=" * 78)

    return best_total_loss, best_state_dict


best_total_loss, best_state_dict = run_lbfgs_stage(
    model=model,
    inputs=inputs,
    targets=targets,
    history=history,
    best_total_loss=best_total_loss,
    best_state_dict=best_state_dict,
)

# Restore best overall state
model.load_state_dict(best_state_dict)


# =============================================================================
# Cell 9: Final Evaluation
# =============================================================================

x_eval = np.linspace(X_MIN, X_MAX, 4000)
preds, exact, errors = evaluate_ic_errors(model, kgs, x_eval)
C_pinn = compute_conserved_quantity_pinn(model, X_MIN, X_MAX, t_val=0.0, n_points=5000)
C_relative_error = abs(C_pinn - C_exact) / abs(C_exact)
final_loss_tensor, final_loss_dict = compute_ic_losses(model, inputs, targets)

print("\n" + "=" * 78)
print("POST-TRAINING EVALUATION (Improved Phase 0)")
print("=" * 78)
print(f"{'Component':<12} {'L_inf Error':<15} {'L2 Error':<15}")
print("-" * 46)
for key in ["u_R", "u_I", "v", "p"]:
    print(f"{key:<12} {errors[f'{key}_Linf']:<15.2e} {errors[f'{key}_L2']:<15.2e}")

print("\nConserved Quantity Check:")
print(f"  C_exact (numerical):  {C_exact:.6f}")
print(f"  C_pinn  (at t=0):     {C_pinn:.6f}")
print(f"  Relative error:       {C_relative_error:.2e}")
print(f"  Target (< {SUCCESS_CONSERVATION_RELERR:.0e}):      "
      f"{'PASS' if C_relative_error < SUCCESS_CONSERVATION_RELERR else 'NEEDS IMPROVEMENT'}")

print("\nPhase 0 Success Criterion:")
print(f"  Final L_total:        {final_loss_dict['total']:.2e}")
print(f"  Target (< {SUCCESS_LOSS:.0e}):      "
      f"{'PASS' if final_loss_dict['total'] < SUCCESS_LOSS else 'NEEDS IMPROVEMENT'}")

phase0_pass = (final_loss_dict["total"] < SUCCESS_LOSS) and (C_relative_error < SUCCESS_CONSERVATION_RELERR)
print(f"  Overall Phase 0:      {'READY FOR PHASE 1' if phase0_pass else 'RUN AGAIN / KEEP REFINING'}")


# =============================================================================
# Cell 10: Figures
# =============================================================================

fig = plt.figure(figsize=(16, 12))
gs = gridspec.GridSpec(2, 2, figure=fig)

# Solution overlay
ax1 = fig.add_subplot(gs[0, 0])
ax1.plot(x_eval, exact["u_R"], "k-", lw=2, label="Exact u_R")
ax1.plot(x_eval, preds["u_R"], "r--", lw=1.5, label="PINN u_R")
ax1.plot(x_eval, exact["u_I"], "b-", lw=2, alpha=0.8, label="Exact u_I")
ax1.plot(x_eval, preds["u_I"], "c--", lw=1.5, alpha=0.9, label="PINN u_I")
ax1.set_title("Phase 0 Fit: Schrödinger Components at t=0")
ax1.set_xlabel("x")
ax1.grid(True, alpha=0.3)
ax1.legend(fontsize=9)

ax2 = fig.add_subplot(gs[0, 1])
ax2.plot(x_eval, exact["v"], "k-", lw=2, label="Exact v")
ax2.plot(x_eval, preds["v"], "r--", lw=1.5, label="PINN v")
ax2.plot(x_eval, exact["p"], "b-", lw=2, alpha=0.8, label="Exact p")
ax2.plot(x_eval, preds["p"], "c--", lw=1.5, alpha=0.9, label="PINN p")
ax2.set_title("Phase 0 Fit: Klein-Gordon Components at t=0")
ax2.set_xlabel("x")
ax2.grid(True, alpha=0.3)
ax2.legend(fontsize=9)

# Error curves
ax3 = fig.add_subplot(gs[1, 0])
for key, color in [("u_R", "red"), ("u_I", "blue"), ("v", "green"), ("p", "purple")]:
    ax3.semilogy(x_eval, np.abs(preds[key] - exact[key]) + 1e-16, color=color, lw=1.5, label=key)
ax3.set_title("Pointwise Absolute Errors at t=0")
ax3.set_xlabel("x")
ax3.grid(True, alpha=0.3)
ax3.legend(fontsize=9)

# Training curves
ax4 = fig.add_subplot(gs[1, 1])
if history["adam_epoch"]:
    ax4.semilogy(history["adam_epoch"], history["adam_total"], "k-", lw=2, label="Adam total")
    ax4.semilogy(history["adam_epoch"], history["adam_uR"], "r--", alpha=0.8, label="Adam u_R")
    ax4.semilogy(history["adam_epoch"], history["adam_uI"], "b--", alpha=0.8, label="Adam u_I")
    ax4.semilogy(history["adam_epoch"], history["adam_v"], "g--", alpha=0.8, label="Adam v")
    ax4.semilogy(history["adam_epoch"], history["adam_p"], color="purple", ls="--", alpha=0.8, label="Adam p")
if history["lbfgs_step"]:
    # plot LBFGS on a separate x-axis scale by offsetting after Adam for visualization
    offset = history["adam_epoch"][-1] if history["adam_epoch"] else 0
    lbfgs_x = [offset + s for s in history["lbfgs_step"]]
    ax4.semilogy(lbfgs_x, history["lbfgs_total"], "k:", lw=2, label="L-BFGS total")
ax4.axhline(SUCCESS_LOSS, color="gray", ls=":", lw=1, label=f"Target {SUCCESS_LOSS:.0e}")
ax4.set_title("Training Loss Curves")
ax4.set_xlabel("Optimization progress")
ax4.grid(True, alpha=0.3)
ax4.legend(fontsize=8)

plt.tight_layout()
plt.savefig("phase0_warm_start_results_improved.png", dpi=150, bbox_inches="tight")
print("\nFigure saved: phase0_warm_start_results_improved.png")

# LR figure
if history["adam_epoch"]:
    plt.figure(figsize=(10, 4))
    plt.plot(history["adam_epoch"], history["adam_lr"], "b-", lw=2)
    plt.yscale("log")
    plt.title("Adam Learning Rate Schedule")
    plt.xlabel("Epoch")
    plt.ylabel("Learning rate")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("phase0_training_lr_improved.png", dpi=150, bbox_inches="tight")
    print("Figure saved: phase0_training_lr_improved.png")


# =============================================================================
# Cell 11: Save Checkpoint
# =============================================================================

checkpoint = {
    "model_state_dict": model.state_dict(),
    "nu_0": NU_0,
    "x_0": X_0,
    "x_min": X_MIN,
    "x_max": X_MAX,
    "seed": SEED,
    "final_loss": final_loss_dict["total"],
    "loss_uR": final_loss_dict["u_R"],
    "loss_uI": final_loss_dict["u_I"],
    "loss_v": final_loss_dict["v"],
    "loss_p": final_loss_dict["p"],
    "conserved_quantity_exact": C_exact,
    "conserved_quantity_pinn": C_pinn,
    "conserved_quantity_relative_error": C_relative_error,
    "n_ic_global": N_IC_GLOBAL,
    "n_ic_center": N_IC_CENTER,
    "n_ic_total": N_IC_TOTAL,
    "center_focus_half_width": CENTER_FOCUS_HALF_WIDTH,
    "history": history,
    "phase0_pass": phase0_pass,
}

torch.save(checkpoint, OUTPUT_CHECKPOINT_PATH)
print(f"\nCheckpoint saved: {OUTPUT_CHECKPOINT_PATH}")
if os.path.exists(OUTPUT_CHECKPOINT_PATH):
    size_kb = os.path.getsize(OUTPUT_CHECKPOINT_PATH) / 1024.0
    print(f"  File size: {size_kb:.1f} KB")


# =============================================================================
# Cell 12: Summary
# =============================================================================

print("\n" + "=" * 78)
print("IMPROVED PHASE 0 SUMMARY REPORT")
print("=" * 78)
print("System:          Klein-Gordon-Schrodinger (KGS)")
print(f"Wave velocity:   nu = {NU_0}")
print(f"Domain:          x in [{X_MIN}, {X_MAX}], t = 0 (IC only)")
print(f"IC points:       {N_IC_TOTAL} = {N_IC_GLOBAL} global + {N_IC_CENTER} focused")
print("Network:         6 hidden layers x 128 neurons, tanh")
print(f"Parameters:      {n_params:,}")
print(f"Start mode:      {start_mode}")
print(f"Best total loss: {final_loss_dict['total']:.2e}")
print("\nComponent L_inf:")
print(f"  u_R: {errors['u_R_Linf']:.2e}")
print(f"  u_I: {errors['u_I_Linf']:.2e}")
print(f"  v:   {errors['v_Linf']:.2e}")
print(f"  p:   {errors['p_Linf']:.2e}")
print("\nConservation:")
print(f"  C_exact  = {C_exact:.6f}")
print(f"  C_pinn   = {C_pinn:.6f}")
print(f"  Rel. err = {C_relative_error:.2e}")
print("\nDecision:")
print(f"  {'READY FOR PHASE 1' if phase0_pass else 'KEEP REFINING PHASE 0'}")
print("=" * 78)
