#!/usr/bin/env python3
"""
================================================================================
Phase 1: Baseline PINN with PDE Residuals for the Coupled KGS System
================================================================================

Builds on Phase 0 (warm-start pre-training on initial conditions).
This script activates the PDE residual losses and boundary conditions,
using a sigmoid ramp to transition smoothly from the Phase 0 soliton lock.

KGS System with Reduction of Order:
    (Real Schrodinger):   -u_I_t + 0.5*u_R_xx + u_R*v = 0
    (Imag Schrodinger):    u_R_t + 0.5*u_I_xx + u_I*v = 0
    (Compatibility):       v_t - p = 0
    (Klein-Gordon):        p_t - v_xx + v - (u_R^2 + u_I^2) = 0

Connects to Phase 0 checkpoint: phase0_warmstart_checkpoint.pt
"""

# =============================================================================
# Cell 1: Imports & Device Setup
# =============================================================================

import os
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
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
# Cell 2: KGS Exact Solution (Identical to Phase 0)
# =============================================================================

class KGSExactSolution:
    """
    Exact soliton solution for the Coupled Klein-Gordon-Schrodinger system.
    Identical to the Phase 0 class for consistency.
    """

    def __init__(self, nu=0.8, x0=0.0):
        assert abs(nu) < 1.0, "Velocity nu must satisfy |nu| < 1"
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
        phase = self._phase(x, t)
        return self.amp_u * self._sech(eta)**2 * np.cos(phase)

    def u_imag(self, x, t):
        eta = self._eta(x, t)
        phase = self._phase(x, t)
        return self.amp_u * self._sech(eta)**2 * np.sin(phase)

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
        integrand = self.u_abs_squared(x_grid, t_val)
        return np_trapz(integrand, x_grid)


# =============================================================================
# Cell 3: Network Architecture (Identical to Phase 0)
# =============================================================================

class PINN_KGS(nn.Module):
    """
    Fully connected PINN for the KGS system.
    Input: (x, t) -> 2 neurons
    Hidden: n_hidden layers x n_neurons, tanh
    Output: (u_R, u_I, v, p) -> 4 neurons
    """

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
# Cell 4: Load Phase 0 Checkpoint
# =============================================================================

PHASE0_PATH = 'phase0_warmstart_checkpoint.pt'

# ── Configuration ──
NU_0 = 0.8
X_0 = 0.0
X_MIN, X_MAX = -10.0, 10.0
T_MAX = 6.0           # Chapter 5 KGS test problem horizon

kgs = KGSExactSolution(nu=NU_0, x0=X_0)
model = PINN_KGS(n_hidden=6, n_neurons=128).to(device)

if os.path.exists(PHASE0_PATH):
    ckpt = torch.load(PHASE0_PATH, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    print(f"Phase 0 checkpoint loaded from: {PHASE0_PATH}")
    print(f"  Phase 0 final loss: {ckpt['final_loss']:.2e}")
    print(f"  Phase 0 epochs:     {ckpt['epoch']}")
    print(f"  Conserved qty (P0): {ckpt['conserved_quantity_pinn']:.6f}")
else:
    print(f"WARNING: {PHASE0_PATH} not found. Starting from random initialization.")
    print("         Phase 1 will still run, but convergence will be slower.")

n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"  Model parameters:   {n_params:,}")
print(f"  Domain: x in [{X_MIN}, {X_MAX}], t in [0, {T_MAX}]")


# =============================================================================
# Cell 5: PDE Residual Computation (Core of Phase 1)
# =============================================================================

def compute_pde_residuals(model, x, t):
    """
    Compute all 4 PDE residuals of the KGS system using automatic
    differentiation. Inputs x and t must have requires_grad=True.

    KGS with Reduction of Order:
        R1 (Real Schrodinger):  -u_I_t + 0.5*u_R_xx + u_R*v = 0
        R2 (Imag Schrodinger):   u_R_t + 0.5*u_I_xx + u_I*v = 0
        R3 (Compatibility):      v_t - p = 0
        R4 (Klein-Gordon):       p_t - v_xx + v - (u_R^2 + u_I^2) = 0

    Parameters
    ----------
    model : PINN_KGS
    x : torch.Tensor (N, 1), requires_grad=True
    t : torch.Tensor (N, 1), requires_grad=True

    Returns
    -------
    residuals : dict with keys 'R_schrod_re', 'R_schrod_im', 'R_compat', 'R_kg'
    pred : dict with model outputs (for reuse in loss computation)
    """
    inp = torch.cat([x, t], dim=1)
    pred = model(inp)

    u_R = pred['u_R']
    u_I = pred['u_I']
    v   = pred['v']
    p   = pred['p']

    ones = torch.ones_like(u_R)

    # ── First-order derivatives ──
    u_R_x = torch.autograd.grad(u_R, x, grad_outputs=ones, create_graph=True)[0]
    u_R_t = torch.autograd.grad(u_R, t, grad_outputs=ones, create_graph=True)[0]
    u_I_x = torch.autograd.grad(u_I, x, grad_outputs=ones, create_graph=True)[0]
    u_I_t = torch.autograd.grad(u_I, t, grad_outputs=ones, create_graph=True)[0]
    v_t   = torch.autograd.grad(v,   t, grad_outputs=ones, create_graph=True)[0]
    v_x   = torch.autograd.grad(v,   x, grad_outputs=ones, create_graph=True)[0]
    p_t   = torch.autograd.grad(p,   t, grad_outputs=ones, create_graph=True)[0]

    # ── Second-order spatial derivatives ──
    u_R_xx = torch.autograd.grad(u_R_x, x, grad_outputs=ones, create_graph=True)[0]
    u_I_xx = torch.autograd.grad(u_I_x, x, grad_outputs=ones, create_graph=True)[0]
    v_xx   = torch.autograd.grad(v_x,   x, grad_outputs=ones, create_graph=True)[0]

    # ── PDE Residuals ──
    # R1: Real part of Schrodinger equation
    #     i*(u_t + 0.5*u_xx) + u*v = 0
    #     Separating: Real => -u_I_t + 0.5*u_R_xx + u_R*v = 0
    R_schrod_re = -u_I_t + 0.5 * u_R_xx + u_R * v

    # R2: Imaginary part of Schrodinger equation
    #     Separating: Imag => u_R_t + 0.5*u_I_xx + u_I*v = 0
    R_schrod_im = u_R_t + 0.5 * u_I_xx + u_I * v

    # R3: Compatibility constraint (reduction of order)
    #     v_t - p = 0
    R_compat = v_t - p

    # R4: Klein-Gordon equation (reduced order)
    #     p_t - v_xx + v - |u|^2 = 0
    R_kg = p_t - v_xx + v - (u_R**2 + u_I**2)

    residuals = {
        'R_schrod_re': R_schrod_re,
        'R_schrod_im': R_schrod_im,
        'R_compat':    R_compat,
        'R_kg':        R_kg,
    }

    return residuals, pred


# =============================================================================
# Cell 6: Data Sampling (Collocation, Boundary, Initial Condition)
# =============================================================================

def latin_hypercube_2d(n_samples, bounds, seed=None):
    """
    2D Latin Hypercube Sampling.

    Parameters
    ----------
    n_samples : int
    bounds : list of (min, max) for each dimension
    seed : int or None

    Returns
    -------
    np.ndarray of shape (n_samples, 2)
    """
    rng = np.random.RandomState(seed)
    d = len(bounds)
    result = np.zeros((n_samples, d))
    for i in range(d):
        intervals = np.linspace(0, 1, n_samples + 1)
        lower = intervals[:-1]
        upper = intervals[1:]
        points = rng.uniform(lower, upper)
        rng.shuffle(points)
        result[:, i] = bounds[i][0] + points * (bounds[i][1] - bounds[i][0])
    return result


def latin_hypercube_1d(n_samples, x_min, x_max, seed=None):
    """1D Latin Hypercube Sampling (reused from Phase 0)."""
    rng = np.random.RandomState(seed)
    intervals = np.linspace(0, 1, n_samples + 1)
    lower = intervals[:-1]
    upper = intervals[1:]
    points = rng.uniform(lower, upper)
    rng.shuffle(points)
    return np.sort(x_min + points * (x_max - x_min))


def generate_training_data(x_min, x_max, t_max, kgs_exact,
                           n_colloc=10000, n_ic=2000, n_bc=500,
                           seed=42):
    """
    Generate all training data for Phase 1.

    Returns
    -------
    data : dict of torch.Tensor on device, with keys:
        'x_col', 't_col'           : collocation points (requires_grad)
        'x_ic', 't_ic', targets_ic : initial condition data
        'x_bc', 't_bc', targets_bc : boundary condition data
    """
    # ── Interior collocation points (LHS) ──
    col_pts = latin_hypercube_2d(
        n_colloc, [(x_min, x_max), (0.0, t_max)], seed=seed
    )
    x_col = torch.tensor(col_pts[:, 0:1], dtype=torch.float32, device=device,
                         requires_grad=True)
    t_col = torch.tensor(col_pts[:, 1:2], dtype=torch.float32, device=device,
                         requires_grad=True)

    # ── Initial condition points (t = 0) ──
    x_ic_np = latin_hypercube_1d(n_ic, x_min, x_max, seed=seed + 1)
    t_ic_np = np.zeros_like(x_ic_np)

    x_ic = torch.tensor(x_ic_np.reshape(-1, 1), dtype=torch.float32, device=device)
    t_ic = torch.zeros_like(x_ic, device=device)
    ic_inp = torch.cat([x_ic, t_ic], dim=1)

    targets_ic = {
        'u_R': torch.tensor(kgs_exact.u_real(x_ic_np, 0.0).reshape(-1, 1),
                            dtype=torch.float32, device=device),
        'u_I': torch.tensor(kgs_exact.u_imag(x_ic_np, 0.0).reshape(-1, 1),
                            dtype=torch.float32, device=device),
        'v':   torch.tensor(kgs_exact.v(x_ic_np, 0.0).reshape(-1, 1),
                            dtype=torch.float32, device=device),
        'p':   torch.tensor(kgs_exact.p(x_ic_np, 0.0).reshape(-1, 1),
                            dtype=torch.float32, device=device),
    }

    # ── Boundary condition points (x = x_min and x = x_max) ──
    t_bc_np = latin_hypercube_1d(n_bc, 0.0, t_max, seed=seed + 2)

    # Left boundary: x = x_min
    x_bc_left = np.full_like(t_bc_np, x_min)
    # Right boundary: x = x_max
    x_bc_right = np.full_like(t_bc_np, x_max)

    # Stack both boundaries
    x_bc_np = np.concatenate([x_bc_left, x_bc_right])
    t_bc_np_all = np.concatenate([t_bc_np, t_bc_np])

    x_bc = torch.tensor(x_bc_np.reshape(-1, 1), dtype=torch.float32, device=device)
    t_bc = torch.tensor(t_bc_np_all.reshape(-1, 1), dtype=torch.float32, device=device)

    targets_bc = {
        'u_R': torch.tensor(kgs_exact.u_real(x_bc_np, t_bc_np_all).reshape(-1, 1),
                            dtype=torch.float32, device=device),
        'u_I': torch.tensor(kgs_exact.u_imag(x_bc_np, t_bc_np_all).reshape(-1, 1),
                            dtype=torch.float32, device=device),
        'v':   torch.tensor(kgs_exact.v(x_bc_np, t_bc_np_all).reshape(-1, 1),
                            dtype=torch.float32, device=device),
        'p':   torch.tensor(kgs_exact.p(x_bc_np, t_bc_np_all).reshape(-1, 1),
                            dtype=torch.float32, device=device),
    }

    data = {
        'x_col': x_col, 't_col': t_col,
        'x_ic': x_ic, 't_ic': t_ic, 'ic_inp': ic_inp, 'targets_ic': targets_ic,
        'x_bc': x_bc, 't_bc': t_bc, 'targets_bc': targets_bc,
        'n_colloc': n_colloc, 'n_ic': n_ic, 'n_bc': 2 * n_bc,
    }

    return data


# Generate training data
data = generate_training_data(
    X_MIN, X_MAX, T_MAX, kgs,
    n_colloc=10000, n_ic=2000, n_bc=500, seed=SEED
)

print(f"\nTraining data generated for Phase 1:")
print(f"  Collocation points: {data['n_colloc']}")
print(f"  IC points:          {data['n_ic']}")
print(f"  BC points:          {data['n_bc']} ({data['n_bc']//2} per boundary)")
print(f"  Domain: x in [{X_MIN}, {X_MAX}], t in [0, {T_MAX}]")


# =============================================================================
# Cell 7: Loss Function with Sigmoid Ramp
# =============================================================================

def sigmoid_ramp(epoch, center=250, steepness=0.02):
    """
    Smooth sigmoid activation for transitioning from Phase 0 to Phase 1.

    Returns a value in [0, 1] that is ~0 for epoch << center and ~1 for
    epoch >> center. Used to ramp up PDE and BC losses gradually.

    Parameters
    ----------
    epoch : int
        Current training epoch.
    center : int
        Epoch at which the ramp reaches 0.5.
    steepness : float
        Controls transition sharpness. Smaller = softer ramp.
    """
    return 1.0 / (1.0 + np.exp(-steepness * (epoch - center)))


def compute_loss(model, data, epoch, ramp_center=250, ramp_steepness=0.02):
    """
    Compute the composite Phase 1 loss.

    L_total = w_pde * L_PDE + L_IC + w_bc * L_BC

    where w_pde and w_bc ramp from 0 to 1 via a sigmoid schedule.

    The IC loss is always active (carries over from Phase 0).

    Returns
    -------
    loss_total : torch.Tensor (scalar)
    loss_dict : dict of float values for logging
    """
    # ── Sigmoid ramp weight for PDE and BC ──
    w_ramp = sigmoid_ramp(epoch, center=ramp_center, steepness=ramp_steepness)

    # ── 1. PDE Residual Loss ──
    residuals, _ = compute_pde_residuals(model, data['x_col'], data['t_col'])

    loss_schrod_re = torch.mean(residuals['R_schrod_re']**2)
    loss_schrod_im = torch.mean(residuals['R_schrod_im']**2)
    loss_compat    = torch.mean(residuals['R_compat']**2)
    loss_kg        = torch.mean(residuals['R_kg']**2)

    L_PDE = loss_schrod_re + loss_schrod_im + loss_compat + loss_kg

    # ── 2. Initial Condition Loss ──
    pred_ic = model(data['ic_inp'])

    loss_ic_uR = torch.mean((pred_ic['u_R'] - data['targets_ic']['u_R'])**2)
    loss_ic_uI = torch.mean((pred_ic['u_I'] - data['targets_ic']['u_I'])**2)
    loss_ic_v  = torch.mean((pred_ic['v']   - data['targets_ic']['v'])**2)
    loss_ic_p  = torch.mean((pred_ic['p']   - data['targets_ic']['p'])**2)

    L_IC = loss_ic_uR + loss_ic_uI + loss_ic_v + loss_ic_p

    # ── 3. Boundary Condition Loss ──
    bc_inp = torch.cat([data['x_bc'], data['t_bc']], dim=1)
    pred_bc = model(bc_inp)

    loss_bc_uR = torch.mean((pred_bc['u_R'] - data['targets_bc']['u_R'])**2)
    loss_bc_uI = torch.mean((pred_bc['u_I'] - data['targets_bc']['u_I'])**2)
    loss_bc_v  = torch.mean((pred_bc['v']   - data['targets_bc']['v'])**2)
    loss_bc_p  = torch.mean((pred_bc['p']   - data['targets_bc']['p'])**2)

    L_BC = loss_bc_uR + loss_bc_uI + loss_bc_v + loss_bc_p

    # ── Composite Loss ──
    # IC always active (weight=1), PDE and BC ramped in
    loss_total = w_ramp * L_PDE + L_IC + w_ramp * L_BC

    loss_dict = {
        'total':      loss_total.item(),
        'pde':        L_PDE.item(),
        'ic':         L_IC.item(),
        'bc':         L_BC.item(),
        'schrod_re':  loss_schrod_re.item(),
        'schrod_im':  loss_schrod_im.item(),
        'compat':     loss_compat.item(),
        'kg':         loss_kg.item(),
        'ic_uR':      loss_ic_uR.item(),
        'ic_uI':      loss_ic_uI.item(),
        'ic_v':       loss_ic_v.item(),
        'ic_p':       loss_ic_p.item(),
        'w_ramp':     w_ramp,
    }

    return loss_total, loss_dict


# =============================================================================
# Cell 8: Evaluation Utilities
# =============================================================================

def evaluate_at_time(model, kgs_exact, t_val, x_min=-10.0, x_max=10.0,
                     n_points=1000):
    """
    Evaluate the PINN against the exact solution at a specific time.

    Returns
    -------
    results : dict with predictions, exact values, and error norms
    """
    model.eval()
    x_np = np.linspace(x_min, x_max, n_points)

    with torch.no_grad():
        x_t = torch.tensor(x_np.reshape(-1, 1), dtype=torch.float32, device=device)
        t_t = torch.full_like(x_t, t_val)
        inp = torch.cat([x_t, t_t], dim=1)
        pred = model(inp)

    preds = {k: pred[k].cpu().numpy().flatten() for k in ['u_R', 'u_I', 'v', 'p']}
    exact = {
        'u_R': kgs_exact.u_real(x_np, t_val),
        'u_I': kgs_exact.u_imag(x_np, t_val),
        'v':   kgs_exact.v(x_np, t_val),
        'p':   kgs_exact.p(x_np, t_val),
    }

    errors = {}
    for key in preds:
        diff = np.abs(preds[key] - exact[key])
        errors[f'{key}_Linf'] = float(np.max(diff))
        errors[f'{key}_L2']   = float(np.sqrt(np.mean(diff**2)))

    return {'x': x_np, 'preds': preds, 'exact': exact, 'errors': errors}


def compute_conserved_quantity(model, x_min, x_max, t_val, n_points=2000):
    """Compute C(t) = integral(|u_NN|^2 dx) via trapezoidal rule."""
    model.eval()
    with torch.no_grad():
        x_q = torch.linspace(x_min, x_max, n_points, device=device).unsqueeze(1)
        t_q = torch.full_like(x_q, t_val)
        inp = torch.cat([x_q, t_q], dim=1)
        pred = model(inp)
        u_abs2 = pred['u_R']**2 + pred['u_I']**2
        dx = (x_max - x_min) / (n_points - 1)
        C = torch_trapz(u_abs2.squeeze(), dx=dx).item()
    return C


def compute_gradient_norms(model, data):
    """
    Compute per-component gradient norms for diagnosis.
    Returns dict of gradient L2 norms for each loss term.
    """
    model.train()
    grad_norms = {}

    # PDE residuals
    residuals, _ = compute_pde_residuals(model, data['x_col'], data['t_col'])
    for name, res in residuals.items():
        loss_i = torch.mean(res**2)
        model.zero_grad()
        loss_i.backward(retain_graph=True)
        total_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                total_norm += p.grad.norm(2).item()**2
        grad_norms[name] = np.sqrt(total_norm)

    # IC loss
    pred_ic = model(data['ic_inp'])
    for key in ['u_R', 'u_I', 'v', 'p']:
        loss_i = torch.mean((pred_ic[key] - data['targets_ic'][key])**2)
        model.zero_grad()
        loss_i.backward(retain_graph=True)
        total_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                total_norm += p.grad.norm(2).item()**2
        grad_norms[f'ic_{key}'] = np.sqrt(total_norm)

    model.zero_grad()
    return grad_norms


# =============================================================================
# Cell 9: Phase 1 Training Loop
# =============================================================================

def train_phase1(model, data, kgs_exact,
                 n_epochs=30000,
                 lr=1e-3,
                 lr_min=1e-6,
                 ramp_center=250,
                 ramp_steepness=0.02,
                 eval_every=500,
                 print_every=500,
                 grad_diag_every=5000):
    """
    Phase 1 training: PDE residual + IC + BC losses with sigmoid ramp.

    Parameters
    ----------
    model : PINN_KGS
    data : dict from generate_training_data
    kgs_exact : KGSExactSolution
    n_epochs : int
    lr : float
        Initial learning rate.
    lr_min : float
        Minimum learning rate for cosine annealing.
    ramp_center : int
        Epoch at which sigmoid ramp = 0.5.
    ramp_steepness : float
        Sigmoid ramp steepness.
    eval_every : int
        Evaluation interval (expensive: computes error norms).
    print_every : int
        Loss printing interval.
    grad_diag_every : int
        Gradient diagnostics interval.

    Returns
    -------
    history : dict of training history
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=n_epochs, eta_min=lr_min
    )

    history = {
        'epoch': [], 'loss_total': [], 'loss_pde': [], 'loss_ic': [],
        'loss_bc': [], 'loss_schrod_re': [], 'loss_schrod_im': [],
        'loss_compat': [], 'loss_kg': [], 'w_ramp': [], 'lr': [],
        # Evaluation metrics (computed less frequently)
        'eval_epochs': [],
        'Linf_uR_t05': [], 'Linf_uI_t05': [], 'Linf_v_t05': [],
        'Linf_uR_t1': [],  'Linf_uI_t1': [],  'Linf_v_t1': [],
        'conserved_t0': [], 'conserved_t05': [], 'conserved_t1': [],
        # Gradient diagnostics
        'grad_epochs': [], 'grad_norms': [],
    }

    C_exact = kgs_exact.conserved_quantity(np.linspace(X_MIN, X_MAX, 2000), 0.0)

    print("=" * 80)
    print("PHASE 1: BASELINE PINN TRAINING (PDE + IC + BC)")
    print("=" * 80)
    print(f"Epochs: {n_epochs} | LR: {lr} -> {lr_min} (cosine)")
    print(f"Sigmoid ramp: center={ramp_center}, steepness={ramp_steepness}")
    print(f"Collocation: {data['n_colloc']} | IC: {data['n_ic']} | BC: {data['n_bc']}")
    print(f"Domain: x in [{X_MIN}, {X_MAX}], t in [0, {T_MAX}]")
    print(f"Conserved quantity (exact): {C_exact:.6f}")
    print("-" * 80)

    t_start = time()
    model.train()

    for epoch in range(1, n_epochs + 1):
        optimizer.zero_grad()

        loss_total, loss_dict = compute_loss(
            model, data, epoch,
            ramp_center=ramp_center, ramp_steepness=ramp_steepness
        )

        loss_total.backward()

        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()
        scheduler.step()

        # ── Logging ──
        if epoch % print_every == 0 or epoch == 1:
            current_lr = optimizer.param_groups[0]['lr']
            history['epoch'].append(epoch)
            history['loss_total'].append(loss_dict['total'])
            history['loss_pde'].append(loss_dict['pde'])
            history['loss_ic'].append(loss_dict['ic'])
            history['loss_bc'].append(loss_dict['bc'])
            history['loss_schrod_re'].append(loss_dict['schrod_re'])
            history['loss_schrod_im'].append(loss_dict['schrod_im'])
            history['loss_compat'].append(loss_dict['compat'])
            history['loss_kg'].append(loss_dict['kg'])
            history['w_ramp'].append(loss_dict['w_ramp'])
            history['lr'].append(current_lr)

            print(f"Ep {epoch:6d}/{n_epochs} | "
                  f"L={loss_dict['total']:.2e} | "
                  f"PDE={loss_dict['pde']:.2e} | "
                  f"IC={loss_dict['ic']:.2e} | "
                  f"BC={loss_dict['bc']:.2e} | "
                  f"w={loss_dict['w_ramp']:.3f} | "
                  f"lr={current_lr:.1e}")

        # ── Evaluation at benchmark times ──
        if epoch % eval_every == 0 or epoch == 1:
            model.eval()

            res_05 = evaluate_at_time(model, kgs_exact, t_val=0.5)
            res_10 = evaluate_at_time(model, kgs_exact, t_val=1.0)

            C_t0  = compute_conserved_quantity(model, X_MIN, X_MAX, 0.0)
            C_t05 = compute_conserved_quantity(model, X_MIN, X_MAX, 0.5)
            C_t1  = compute_conserved_quantity(model, X_MIN, X_MAX, 1.0)

            history['eval_epochs'].append(epoch)
            history['Linf_uR_t05'].append(res_05['errors']['u_R_Linf'])
            history['Linf_uI_t05'].append(res_05['errors']['u_I_Linf'])
            history['Linf_v_t05'].append(res_05['errors']['v_Linf'])
            history['Linf_uR_t1'].append(res_10['errors']['u_R_Linf'])
            history['Linf_uI_t1'].append(res_10['errors']['u_I_Linf'])
            history['Linf_v_t1'].append(res_10['errors']['v_Linf'])
            history['conserved_t0'].append(C_t0)
            history['conserved_t05'].append(C_t05)
            history['conserved_t1'].append(C_t1)

            if epoch % (eval_every * 4) == 0 or epoch == eval_every:
                print(f"  >> EVAL at epoch {epoch}:")
                print(f"     t=0.5: Linf(uR)={res_05['errors']['u_R_Linf']:.2e}, "
                      f"Linf(uI)={res_05['errors']['u_I_Linf']:.2e}, "
                      f"Linf(v)={res_05['errors']['v_Linf']:.2e}")
                print(f"     t=1.0: Linf(uR)={res_10['errors']['u_R_Linf']:.2e}, "
                      f"Linf(uI)={res_10['errors']['u_I_Linf']:.2e}, "
                      f"Linf(v)={res_10['errors']['v_Linf']:.2e}")
                print(f"     Conserved: C(0)={C_t0:.6f}, C(0.5)={C_t05:.6f}, "
                      f"C(1)={C_t1:.6f} [exact={C_exact:.6f}]")

            model.train()

        # ── Gradient diagnostics ──
        if epoch % grad_diag_every == 0:
            gnorms = compute_gradient_norms(model, data)
            history['grad_epochs'].append(epoch)
            history['grad_norms'].append(gnorms)
            stiffness = max(gnorms.values()) / (min(gnorms.values()) + 1e-30)
            print(f"  >> GRAD DIAG at epoch {epoch}: stiffness ratio = {stiffness:.1f}")
            model.train()

    t_end = time()
    elapsed = t_end - t_start

    print("-" * 80)
    print(f"Phase 1 complete in {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print(f"Final total loss: {history['loss_total'][-1]:.2e}")
    print("=" * 80)

    return history


# ── Run Phase 1 Training ──
history = train_phase1(
    model, data, kgs,
    n_epochs=30000,
    lr=1e-3,
    lr_min=1e-6,
    ramp_center=250,
    ramp_steepness=0.02,
    eval_every=500,
    print_every=500,
    grad_diag_every=5000,
)


# =============================================================================
# Cell 10: L-BFGS Fine-Tuning (Optional but Recommended)
# =============================================================================

def lbfgs_finetune(model, data, n_iters=5000, lr=0.5, print_every=500):
    """
    L-BFGS second-stage optimizer for fine-tuning.

    L-BFGS is a quasi-Newton method that uses curvature information.
    It often squeezes out the last 1-2 orders of magnitude in PINN training.
    """
    optimizer = torch.optim.LBFGS(
        model.parameters(), lr=lr, max_iter=20,
        history_size=50, tolerance_grad=1e-9, tolerance_change=1e-11,
        line_search_fn='strong_wolfe'
    )

    losses = []
    print("\n" + "=" * 60)
    print("L-BFGS FINE-TUNING")
    print("=" * 60)

    t_start = time()
    model.train()

    for i in range(1, n_iters + 1):
        def closure():
            optimizer.zero_grad()
            # Use a large epoch number so ramp = 1.0
            loss, _ = compute_loss(model, data, epoch=99999)
            loss.backward()
            return loss

        loss = optimizer.step(closure)
        losses.append(loss.item())

        if i % print_every == 0 or i == 1:
            print(f"  L-BFGS iter {i:5d}/{n_iters} | Loss: {loss.item():.2e}")

    print(f"L-BFGS complete in {time()-t_start:.1f}s")
    print(f"Final loss: {losses[-1]:.2e}")
    print("=" * 60)

    return losses


# Uncomment below to run L-BFGS (can take several minutes):
# lbfgs_losses = lbfgs_finetune(model, data, n_iters=3000, print_every=500)


# =============================================================================
# Cell 11: Final Evaluation
# =============================================================================

print("\n" + "=" * 80)
print("PHASE 1: FINAL EVALUATION")
print("=" * 80)

# ── Error norms at benchmark times ──
eval_times = [0.0, 0.5, 1.0]
C_exact = kgs.conserved_quantity(np.linspace(X_MIN, X_MAX, 2000), 0.0)

all_results = {}
for t_val in eval_times:
    res = evaluate_at_time(model, kgs, t_val)
    C_pinn = compute_conserved_quantity(model, X_MIN, X_MAX, t_val)
    all_results[t_val] = {'res': res, 'C_pinn': C_pinn}

# Print error table
print(f"\n{'Time':<6} {'Component':<10} {'L_inf':<12} {'L2':<12}")
print("-" * 42)
for t_val in eval_times:
    res = all_results[t_val]['res']
    for key in ['u_R', 'u_I', 'v']:
        print(f"{t_val:<6.1f} {key:<10} "
              f"{res['errors'][f'{key}_Linf']:<12.2e} "
              f"{res['errors'][f'{key}_L2']:<12.2e}")
    print()

# Conservation quantity report
print("Conservation Quantity C(t) = integral(|u|^2 dx):")
print(f"  C_exact    = {C_exact:.6f}")
for t_val in eval_times:
    C_p = all_results[t_val]['C_pinn']
    rel = abs(C_p - C_exact) / abs(C_exact)
    print(f"  C({t_val:.1f})     = {C_p:.6f}  (rel. error = {rel:.2e})")

max_drift = max(
    abs(all_results[t]['C_pinn'] - C_exact) / abs(C_exact)
    for t in eval_times
)
print(f"  Max drift  = {max_drift:.2e}")


# =============================================================================
# Cell 12: Visualization - Solution Profiles at t=0.5 and t=1.0
# =============================================================================

plt.rcParams.update({
    'font.size': 11, 'axes.labelsize': 12, 'axes.titlesize': 13,
    'legend.fontsize': 9, 'figure.dpi': 120,
})

fig, axes = plt.subplots(3, 2, figsize=(15, 13))

for col, (t_val, label) in enumerate([(0.5, 't = 0.5'), (1.0, 't = 1.0')]):
    res = all_results[t_val]['res']
    x = res['x']

    # Row 0: u_R
    ax = axes[0, col]
    ax.plot(x, res['exact']['u_R'], 'b-', lw=2.0, label='Exact')
    ax.plot(x, res['preds']['u_R'], 'r--', lw=1.5, alpha=0.85, label='PINN')
    ax.set_ylabel('$u_R$')
    ax.set_title(f'Real part of $u$ at ${label}$')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(X_MIN, X_MAX)

    # Row 1: u_I
    ax = axes[1, col]
    ax.plot(x, res['exact']['u_I'], 'b-', lw=2.0, label='Exact')
    ax.plot(x, res['preds']['u_I'], 'r--', lw=1.5, alpha=0.85, label='PINN')
    ax.set_ylabel('$u_I$')
    ax.set_title(f'Imaginary part of $u$ at ${label}$')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(X_MIN, X_MAX)

    # Row 2: v
    ax = axes[2, col]
    ax.plot(x, res['exact']['v'], 'b-', lw=2.0, label='Exact')
    ax.plot(x, res['preds']['v'], 'r--', lw=1.5, alpha=0.85, label='PINN')
    ax.set_xlabel('$x$')
    ax.set_ylabel('$v$')
    ax.set_title(f'Klein-Gordon field $v$ at ${label}$')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(X_MIN, X_MAX)

plt.suptitle(f'Phase 1 Baseline: PINN vs Exact Solution '
             f'($\\nu_0={NU_0}$, {n_params:,} params)',
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('phase1_solution_profiles.png', bbox_inches='tight', dpi=150)
plt.show()
print("Saved: phase1_solution_profiles.png")


# =============================================================================
# Cell 13: Visualization - Error Distributions (cf. Figs 5.3-5.8)
# =============================================================================

fig, axes = plt.subplots(2, 3, figsize=(16, 9))

for row, (t_val, label) in enumerate([(0.5, 't = 0.5'), (1.0, 't = 1.0')]):
    res = all_results[t_val]['res']
    x = res['x']

    for col, (key, ylabel) in enumerate([
        ('u_R', 'Error in Re($u$)'),
        ('u_I', 'Error in Im($u$)'),
        ('v',   'Error in $v$'),
    ]):
        ax = axes[row, col]
        err = res['preds'][key] - res['exact'][key]
        ax.plot(x, err, 'b-', lw=1.0, alpha=0.8)
        ax.axhline(y=0, color='k', lw=0.5, alpha=0.3)
        ax.set_xlabel('$x$')
        ax.set_ylabel(ylabel)
        ax.set_title(f'{ylabel} at ${label}$\n'
                     f'$L_\\infty$ = {res["errors"][f"{key}_Linf"]:.2e}')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(X_MIN, X_MAX)

plt.suptitle('Phase 1: Pointwise Error Distributions (cf. Document Figs. 5.3-5.8)',
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('phase1_error_distributions.png', bbox_inches='tight', dpi=150)
plt.show()
print("Saved: phase1_error_distributions.png")


# =============================================================================
# Cell 14: Visualization - Training Convergence
# =============================================================================

fig, axes = plt.subplots(2, 2, figsize=(15, 10))

epochs = history['epoch']

# Panel 1: Loss components
ax = axes[0, 0]
ax.semilogy(epochs, history['loss_total'], 'k-', lw=1.5, label='$L_{total}$')
ax.semilogy(epochs, history['loss_pde'],   'r-', lw=1.0, alpha=0.8, label='$L_{PDE}$')
ax.semilogy(epochs, history['loss_ic'],    'b-', lw=1.0, alpha=0.8, label='$L_{IC}$')
ax.semilogy(epochs, history['loss_bc'],    'g-', lw=1.0, alpha=0.8, label='$L_{BC}$')
ax.set_xlabel('Epoch')
ax.set_ylabel('Loss (log scale)')
ax.set_title('Composite Loss Components')
ax.legend()
ax.grid(True, alpha=0.3)

# Panel 2: PDE residual breakdown
ax = axes[0, 1]
ax.semilogy(epochs, history['loss_schrod_re'], '-', lw=1.0, label='$R_{Schr,Re}$')
ax.semilogy(epochs, history['loss_schrod_im'], '-', lw=1.0, label='$R_{Schr,Im}$')
ax.semilogy(epochs, history['loss_compat'],    '-', lw=1.0, label='$R_{compat}$')
ax.semilogy(epochs, history['loss_kg'],        '-', lw=1.0, label='$R_{KG}$')
ax.set_xlabel('Epoch')
ax.set_ylabel('Loss (log scale)')
ax.set_title('PDE Residual Breakdown')
ax.legend()
ax.grid(True, alpha=0.3)

# Panel 3: Error norms vs epoch
ax = axes[1, 0]
eval_ep = history['eval_epochs']
ax.semilogy(eval_ep, history['Linf_uR_t1'], 'o-', ms=3, label='$L_\\infty(u_R)$ at $t=1$')
ax.semilogy(eval_ep, history['Linf_uI_t1'], 's-', ms=3, label='$L_\\infty(u_I)$ at $t=1$')
ax.semilogy(eval_ep, history['Linf_v_t1'],  '^-', ms=3, label='$L_\\infty(v)$ at $t=1$')
ax.set_xlabel('Epoch')
ax.set_ylabel('$L_\\infty$ Error')
ax.set_title('Error Convergence at $t=1$')
ax.legend(fontsize=8, ncol=2)
ax.grid(True, alpha=0.3)

# Panel 4: Conservation quantity over time
ax = axes[1, 1]
if len(history['conserved_t0']) > 0:
    C_exact_val = kgs.conserved_quantity(np.linspace(X_MIN, X_MAX, 2000), 0.0)
    ax.plot(eval_ep, history['conserved_t0'], 'o-', ms=3, label='$C(t=0)$')
    ax.plot(eval_ep, history['conserved_t05'], 's-', ms=3, label='$C(t=0.5)$')
    ax.plot(eval_ep, history['conserved_t1'], '^-', ms=3, label='$C(t=1)$')
    ax.axhline(y=C_exact_val, color='k', ls='--', lw=1.5, alpha=0.7, label=f'Exact = {C_exact_val:.4f}')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('$\\int |u|^2 dx$')
    ax.set_title('Conserved Quantity Tracking')
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.suptitle('Phase 1: Training Diagnostics', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('phase1_training_diagnostics.png', bbox_inches='tight', dpi=150)
plt.show()
print("Saved: phase1_training_diagnostics.png")


# =============================================================================
# Cell 15: Visualization - Sigmoid Ramp and IC Degradation Monitor
# =============================================================================

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Panel 1: Sigmoid ramp profile
ax = axes[0]
ramp_epochs = np.arange(1, 2000)
ramp_vals = [sigmoid_ramp(e, center=250, steepness=0.02) for e in ramp_epochs]
ax.plot(ramp_epochs, ramp_vals, 'b-', lw=2)
ax.axhline(y=0.5, color='gray', ls='--', alpha=0.5)
ax.axvline(x=250, color='gray', ls='--', alpha=0.5)
ax.set_xlabel('Epoch')
ax.set_ylabel('PDE/BC Loss Weight')
ax.set_title('Sigmoid Ramp: Phase 0 -> Phase 1 Transition')
ax.set_xlim(0, 1500)
ax.grid(True, alpha=0.3)
ax.annotate('Ramp center = 250', xy=(250, 0.5), xytext=(500, 0.3),
            arrowprops=dict(arrowstyle='->', color='gray'), fontsize=10)

# Panel 2: IC loss during Phase 1 (should not degrade > 1 order of magnitude)
ax = axes[1]
ax.semilogy(epochs, history['loss_ic'], 'b-', lw=1.5, label='$L_{IC}$ during Phase 1')
if len(epochs) > 0:
    ic_start = history['loss_ic'][0]
    ax.axhline(y=ic_start, color='gray', ls='--', alpha=0.5, label=f'Phase 0 exit: {ic_start:.2e}')
    ax.axhline(y=ic_start * 10, color='r', ls=':', alpha=0.5, label=f'1 OoM threshold: {ic_start*10:.2e}')
ax.set_xlabel('Epoch')
ax.set_ylabel('IC Loss (log scale)')
ax.set_title('IC Loss Monitoring (should stay within 1 OoM of Phase 0)')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('phase1_ramp_and_ic_monitor.png', bbox_inches='tight', dpi=150)
plt.show()
print("Saved: phase1_ramp_and_ic_monitor.png")


# =============================================================================
# Cell 16: Gradient Stiffness Diagnostics
# =============================================================================

if len(history['grad_norms']) > 0:
    fig, ax = plt.subplots(figsize=(12, 6))

    grad_ep = history['grad_epochs']
    components = list(history['grad_norms'][0].keys())
    for comp in components:
        vals = [gn[comp] for gn in history['grad_norms']]
        ax.semilogy(grad_ep, vals, 'o-', ms=4, label=comp)

    ax.set_xlabel('Epoch')
    ax.set_ylabel('Gradient $L_2$ Norm (log scale)')
    ax.set_title('Per-Component Gradient Norms (Stiffness Diagnostic)')
    ax.legend(fontsize=8, ncol=2, loc='upper right')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('phase1_gradient_diagnostics.png', bbox_inches='tight', dpi=150)
    plt.show()
    print("Saved: phase1_gradient_diagnostics.png")

    # Print stiffness ratios
    print("\nGradient Stiffness Ratios:")
    for i, ep in enumerate(grad_ep):
        gn = history['grad_norms'][i]
        S = max(gn.values()) / (min(gn.values()) + 1e-30)
        flag = " << SEVERE" if S > 100 else ""
        print(f"  Epoch {ep}: S = {S:.1f}{flag}")


# =============================================================================
# Cell 17: Save Phase 1 Checkpoint
# =============================================================================

phase1_checkpoint = {
    'model_state_dict': model.state_dict(),
    'epoch': history['epoch'][-1] if history['epoch'] else 0,
    'final_loss': history['loss_total'][-1] if history['loss_total'] else None,
    'history': history,
    'config': {
        'nu': NU_0, 'x0': X_0,
        'domain_x': [X_MIN, X_MAX], 'domain_t': [0.0, T_MAX],
        'n_hidden': 6, 'n_neurons': 128,
        'activation': 'tanh',
        'n_colloc': data['n_colloc'],
        'n_ic': data['n_ic'],
        'n_bc': data['n_bc'],
    },
    'final_errors': {
        t: all_results[t]['res']['errors'] for t in eval_times
    },
    'final_conservation': {
        t: all_results[t]['C_pinn'] for t in eval_times
    },
    'C_exact': C_exact,
}

ckpt_path = 'phase1_baseline_checkpoint.pt'
torch.save(phase1_checkpoint, ckpt_path)
print(f"\nPhase 1 checkpoint saved: {ckpt_path}")
print(f"  File size: {os.path.getsize(ckpt_path) / 1024:.1f} KB")


# =============================================================================
# Cell 18: Phase 1 Summary Report
# =============================================================================

print("\n" + "=" * 80)
print("PHASE 1 SUMMARY REPORT")
print("=" * 80)

best_t1 = all_results[1.0]['res']['errors']
best_t05 = all_results[0.5]['res']['errors']

print(f"""
System:            Klein-Gordon-Schrodinger (KGS)
Wave velocity:     nu = {NU_0}
Domain:            x in [{X_MIN}, {X_MAX}], t in [0, {T_MAX}]
Network:           6 x 128 tanh, {n_params:,} parameters

Training:
  Phase 0 epochs:  (from checkpoint)
  Phase 1 epochs:  {history['epoch'][-1] if history['epoch'] else 0}
  Final total loss: {history['loss_total'][-1]:.2e}
  Final PDE loss:   {history['loss_pde'][-1]:.2e}
  Final IC loss:    {history['loss_ic'][-1]:.2e}
  Final BC loss:    {history['loss_bc'][-1]:.2e}

Error Norms at t=0.5:
  Linf(u_R) = {best_t05['u_R_Linf']:.2e}
  Linf(u_I) = {best_t05['u_I_Linf']:.2e}
  Linf(v)   = {best_t05['v_Linf']:.2e}

Error Norms at t=1.0:
  Linf(u_R) = {best_t1['u_R_Linf']:.2e}
  Linf(u_I) = {best_t1['u_I_Linf']:.2e}
  Linf(v)   = {best_t1['v_Linf']:.2e}

Conservation (C_exact = {C_exact:.6f}):
  C(0.0)  = {all_results[0.0]['C_pinn']:.6f}
  C(0.5)  = {all_results[0.5]['C_pinn']:.6f}
  C(1.0)  = {all_results[1.0]['C_pinn']:.6f}
  Max drift = {max_drift:.2e}
""")

print("Phase 2 Diagnostics to Look For:")
print("  1. Check gradient stiffness ratios above (S > 100 = severe)")
print("  2. Check if IC loss degraded > 1 OoM from Phase 0")
print("  3. Check error distribution plots for spectral bias pattern")
print("  4. Check conservation drift for systematic vs. oscillatory behavior")
print()
print("Next Steps:")
print("  - If stiffness ratio S > 100: implement adaptive weighting (LRA)")
print("  - If conservation drift > 1e-2: implement Lagrange multiplier")
print("  - If error grows sharply with t: implement causal training")
print("=" * 80)
