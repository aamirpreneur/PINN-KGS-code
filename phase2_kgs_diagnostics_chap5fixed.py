#!/usr/bin/env python3
"""
================================================================================
Phase 2: Dynamic Diagnosis & Failure Analysis for the KGS Baseline PINN
================================================================================

This script systematically profiles the Phase 1 baseline PINN to identify
failure modes and prescribe Phase 3 remedies. It produces:

  1. Gradient flow diagnostics (per-component norms, stiffness ratios)
  2. Spectral bias analysis (Fourier decomposition of errors)
  3. Temporal propagation analysis (error vs. time, reliability horizon)
  4. Loss component imbalance profiling
  5. Conservation drift characterization
  6. NTK eigenvalue spectrum (optional, expensive)
  7. A ranked diagnostic summary with Phase 3 prescriptions

Loads: phase1_baseline_checkpoint.pt
No training is performed (pure evaluation/analysis).
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
# Cell 2: KGS Exact Solution (Identical to Phase 0/1)
# =============================================================================

class KGSExactSolution:
    """Exact soliton solution, identical to Phase 0/1."""

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
        return np_trapz(self.u_abs_squared(x_grid, t_val), x_grid)


# =============================================================================
# Cell 3: Network Architecture (Identical to Phase 0/1)
# =============================================================================

class PINN_KGS(nn.Module):
    """Identical to Phase 0/1."""

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
# Cell 4: PDE Residuals (Corrected, Identical to Phase 1)
# =============================================================================

def compute_pde_residuals(model, x, t):
    """
    Compute 4 PDE residuals with the CORRECTED Schrodinger splitting.

    R1 (Real Schrodinger):  -u_I_t + 0.5*u_R_xx + u_R*v = 0
    R2 (Imag Schrodinger):   u_R_t + 0.5*u_I_xx + u_I*v = 0
    R3 (Compatibility):      v_t - p = 0
    R4 (Klein-Gordon):       p_t - v_xx + v - (u_R^2 + u_I^2) = 0
    """
    inp = torch.cat([x, t], dim=1)
    pred = model(inp)

    u_R, u_I, v, p = pred['u_R'], pred['u_I'], pred['v'], pred['p']
    ones = torch.ones_like(u_R)

    # First-order derivatives
    u_R_x = torch.autograd.grad(u_R, x, grad_outputs=ones, create_graph=True)[0]
    u_R_t = torch.autograd.grad(u_R, t, grad_outputs=ones, create_graph=True)[0]
    u_I_x = torch.autograd.grad(u_I, x, grad_outputs=ones, create_graph=True)[0]
    u_I_t = torch.autograd.grad(u_I, t, grad_outputs=ones, create_graph=True)[0]
    v_t   = torch.autograd.grad(v,   t, grad_outputs=ones, create_graph=True)[0]
    v_x   = torch.autograd.grad(v,   x, grad_outputs=ones, create_graph=True)[0]
    p_t   = torch.autograd.grad(p,   t, grad_outputs=ones, create_graph=True)[0]

    # Second-order spatial derivatives
    u_R_xx = torch.autograd.grad(u_R_x, x, grad_outputs=ones, create_graph=True)[0]
    u_I_xx = torch.autograd.grad(u_I_x, x, grad_outputs=ones, create_graph=True)[0]
    v_xx   = torch.autograd.grad(v_x,   x, grad_outputs=ones, create_graph=True)[0]

    # CORRECTED residuals (matching your Phase 1 fix)
    R_schrod_re = -u_I_t + 0.5 * u_R_xx + u_R * v
    R_schrod_im =  u_R_t + 0.5 * u_I_xx + u_I * v
    R_compat    =  v_t - p
    R_kg        =  p_t - v_xx + v - (u_R**2 + u_I**2)

    return {
        'R_schrod_re': R_schrod_re,
        'R_schrod_im': R_schrod_im,
        'R_compat':    R_compat,
        'R_kg':        R_kg,
    }, pred


# =============================================================================
# Cell 5: Load Phase 1 Checkpoint & Configuration
# =============================================================================

NU_0 = 0.8
X_0 = 0.0
X_MIN, X_MAX = -10.0, 10.0
T_MAX = 6.0

kgs = KGSExactSolution(nu=NU_0, x0=X_0)
model = PINN_KGS(n_hidden=6, n_neurons=128).to(device)

PHASE1_PATH = 'phase1_baseline_checkpoint.pt'
if os.path.exists(PHASE1_PATH):
    ckpt = torch.load(PHASE1_PATH, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    print(f"Phase 1 checkpoint loaded: {PHASE1_PATH}")
    print(f"  Final loss:    {ckpt['final_loss']:.2e}")
    print(f"  Epochs:        {ckpt['epoch']}")
else:
    raise FileNotFoundError(
        f"{PHASE1_PATH} not found. Run Phase 1 first."
    )

n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
C_exact = kgs.conserved_quantity(np.linspace(X_MIN, X_MAX, 5000), 0.0)
print(f"  Parameters:    {n_params:,}")
print(f"  C_exact:       {C_exact:.6f}")
print(f"  Domain:        x in [{X_MIN}, {X_MAX}], t in [0, {T_MAX}]")

# Evaluation helper
def evaluate_at_time(model, kgs_exact, t_val, n_points=1000):
    """Evaluate PINN vs exact at a single time snapshot."""
    model.eval()
    x_np = np.linspace(X_MIN, X_MAX, n_points)
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


print("\n" + "=" * 80)
print("PHASE 2: DYNAMIC DIAGNOSIS & FAILURE ANALYSIS")
print("=" * 80)


# =============================================================================
# Cell 6: Task 2.1 - Gradient Flow Diagnostics
# =============================================================================

print("\n" + "-" * 60)
print("TASK 2.1: GRADIENT FLOW DIAGNOSTICS")
print("-" * 60)

def compute_detailed_gradient_norms(model, n_points=5000):
    """
    Compute per-component gradient norms using fresh collocation points.
    Returns gradient L2 norms for each of the 4 PDE residual components
    and the stiffness ratio.
    """
    model.train()

    # Fresh collocation points
    rng = np.random.RandomState(999)
    x_np = rng.uniform(X_MIN, X_MAX, (n_points, 1))
    t_np = rng.uniform(0.0, T_MAX, (n_points, 1))

    x = torch.tensor(x_np, dtype=torch.float32, device=device, requires_grad=True)
    t = torch.tensor(t_np, dtype=torch.float32, device=device, requires_grad=True)

    residuals, _ = compute_pde_residuals(model, x, t)

    grad_norms = {}
    for name, res in residuals.items():
        loss_i = torch.mean(res**2)
        model.zero_grad()
        loss_i.backward(retain_graph=True)
        total_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                total_norm += p.grad.norm(2).item()**2
        grad_norms[name] = np.sqrt(total_norm)

    model.zero_grad()
    model.eval()

    stiffness = max(grad_norms.values()) / (min(grad_norms.values()) + 1e-30)
    return grad_norms, stiffness


grad_norms, stiffness = compute_detailed_gradient_norms(model)

print("\nPer-Component Gradient L2 Norms:")
for name, norm in sorted(grad_norms.items(), key=lambda x: -x[1]):
    print(f"  {name:<16}: {norm:.4e}")
print(f"\nStiffness ratio S = max/min = {stiffness:.2f}")

if stiffness > 100:
    grad_verdict = "SEVERE: S > 100. Adaptive weighting (LRA/NTK) is ESSENTIAL."
elif stiffness > 10:
    grad_verdict = "MODERATE: 10 < S < 100. Adaptive weighting recommended but not critical."
else:
    grad_verdict = "MILD: S < 10. Gradient balance is acceptable."
print(f"Verdict: {grad_verdict}")


# =============================================================================
# Cell 7: Task 2.2 - Spectral Bias Analysis
# =============================================================================

print("\n" + "-" * 60)
print("TASK 2.2: SPECTRAL BIAS ANALYSIS")
print("-" * 60)

def spectral_analysis(model, kgs_exact, t_val, n_points=1024):
    """
    Perform Fourier decomposition of the exact solution, PINN prediction,
    and their error at a given time.

    Returns wavenumbers, power spectra, and the critical wavenumber k*
    where PINN error exceeds 10% of exact signal amplitude.
    """
    res = evaluate_at_time(model, kgs_exact, t_val, n_points=n_points)
    x = res['x']
    dx = x[1] - x[0]

    spectra = {}
    for key in ['u_R', 'u_I', 'v']:
        exact_fft = np.fft.rfft(res['exact'][key])
        pred_fft  = np.fft.rfft(res['preds'][key])
        error_fft = np.fft.rfft(res['preds'][key] - res['exact'][key])

        freqs = np.fft.rfftfreq(n_points, d=dx)
        wavenumbers = 2 * np.pi * freqs  # Convert frequency to wavenumber

        power_exact = np.abs(exact_fft)**2
        power_pred  = np.abs(pred_fft)**2
        power_error = np.abs(error_fft)**2

        # Find critical wavenumber k* where error > 10% of signal
        ratio = np.zeros_like(power_exact)
        mask = power_exact > 1e-20  # Avoid division by zero
        ratio[mask] = power_error[mask] / power_exact[mask]
        k_star_idx = np.where(ratio > 0.01)[0]  # 10% in power = 1% ratio
        k_star = wavenumbers[k_star_idx[0]] if len(k_star_idx) > 0 else wavenumbers[-1]

        spectra[key] = {
            'wavenumbers': wavenumbers,
            'power_exact': power_exact,
            'power_pred':  power_pred,
            'power_error': power_error,
            'k_star': k_star,
        }

    return spectra, res


# Analyze at t = 0.5, 1.0, 3.0, 6.0
spectral_times = [0.5, 1.0, 3.0, 6.0]
spectral_results = {}
for t_val in spectral_times:
    spectra, res = spectral_analysis(model, kgs, t_val)
    spectral_results[t_val] = {'spectra': spectra, 'spatial': res}

print("\nCritical wavenumber k* (error > 10% of signal power):")
print(f"{'Time':<6} {'u_R k*':<12} {'u_I k*':<12} {'v k*':<12}")
print("-" * 42)
for t_val in spectral_times:
    sp = spectral_results[t_val]['spectra']
    print(f"{t_val:<6.1f} "
          f"{sp['u_R']['k_star']:<12.2f} "
          f"{sp['u_I']['k_star']:<12.2f} "
          f"{sp['v']['k_star']:<12.2f}")

# Determine if spectral bias is the dominant issue
# Check where errors are concentrated spatially
print("\nSpatial Error Concentration at t=1.0:")
res_t1 = spectral_results[1.0]['spatial']
x = res_t1['x']
for key in ['u_R', 'u_I', 'v']:
    err = np.abs(res_t1['preds'][key] - res_t1['exact'][key])
    # Fraction of total error within |x - soliton_center| < 3
    soliton_center = NU_0 * 1.0  # at t=1
    inner_mask = np.abs(x - soliton_center) < 3.0
    err_inner = np.sum(err[inner_mask])
    err_total = np.sum(err)
    frac = err_inner / (err_total + 1e-30)
    print(f"  {key}: {frac*100:.1f}% of error within |x - {soliton_center:.1f}| < 3")

# Visualization: Power spectra
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
for idx, t_val in enumerate(spectral_times):
    ax = axes[idx // 2, idx % 2]
    sp = spectral_results[t_val]['spectra']

    for key, color, ls in [('u_R', 'blue', '-'), ('u_I', 'red', '--'), ('v', 'green', ':')]:
        k = sp[key]['wavenumbers']
        ax.semilogy(k, sp[key]['power_exact'] + 1e-30, color=color, ls=ls,
                    lw=1.5, alpha=0.7, label=f'Exact {key}')
        ax.semilogy(k, sp[key]['power_error'] + 1e-30, color=color, ls=ls,
                    lw=1.0, alpha=0.4)
        ax.axvline(x=sp[key]['k_star'], color=color, ls=':', alpha=0.4, lw=0.8)

    ax.set_xlabel('Wavenumber $k$')
    ax.set_ylabel('Power Spectrum')
    ax.set_title(f'Spectral Content at $t = {t_val}$')
    ax.set_xlim(0, 20)
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)

plt.suptitle('Task 2.2: Spectral Bias Analysis\n'
             '(Bold = exact signal, Faded = error spectrum, Dotted = $k^*$)',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('phase2_spectral_analysis.png', bbox_inches='tight', dpi=150)
plt.show()
print("Saved: phase2_spectral_analysis.png")

# Visualization: Spatial error distributions at multiple times (cf. Figs 5.3-5.8)
fig, axes = plt.subplots(len(spectral_times), 3, figsize=(16, 3.5 * len(spectral_times)))
for row, t_val in enumerate(spectral_times):
    res = spectral_results[t_val]['spatial']
    x = res['x']
    soliton_x = NU_0 * t_val
    for col, (key, ylabel) in enumerate([
        ('u_R', 'Error in Re($u$)'), ('u_I', 'Error in Im($u$)'), ('v', 'Error in $v$')
    ]):
        ax = axes[row, col]
        err = res['preds'][key] - res['exact'][key]
        ax.plot(x, err, 'b-', lw=0.8, alpha=0.8)
        ax.axhline(y=0, color='k', lw=0.3, alpha=0.3)
        ax.axvline(x=soliton_x, color='r', ls='--', lw=0.8, alpha=0.4,
                   label=f'Soliton center')
        ax.axvspan(soliton_x - 3, soliton_x + 3, alpha=0.05, color='red')
        ax.set_xlim(X_MIN, X_MAX)
        ax.set_title(f'{ylabel}, $t={t_val}$, $L_\\infty$={res["errors"][f"{key}_Linf"]:.2e}',
                     fontsize=10)
        ax.grid(True, alpha=0.2)
        if row == len(spectral_times) - 1:
            ax.set_xlabel('$x$')
        if col == 0:
            ax.set_ylabel(f'$t = {t_val}$', fontsize=11, fontweight='bold')

plt.suptitle('Task 2.2: Spatial Error Distributions (cf. Document Figs. 5.3-5.8)\n'
             'Red band = soliton region ($\\pm 3$ around center)',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('phase2_spatial_errors.png', bbox_inches='tight', dpi=150)
plt.show()
print("Saved: phase2_spatial_errors.png")


# =============================================================================
# Cell 8: Task 2.3 - Temporal Propagation Analysis
# =============================================================================

print("\n" + "-" * 60)
print("TASK 2.3: TEMPORAL PROPAGATION ANALYSIS")
print("-" * 60)

# Evaluate at many time snapshots
time_snapshots = np.array([0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 2.5,
                           3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0])
temporal_errors = {key: {'Linf': [], 'L2': []} for key in ['u_R', 'u_I', 'v', 'p']}
temporal_conservation = []

for t_val in time_snapshots:
    res = evaluate_at_time(model, kgs, t_val, n_points=1000)
    for key in ['u_R', 'u_I', 'v', 'p']:
        temporal_errors[key]['Linf'].append(res['errors'][f'{key}_Linf'])
        temporal_errors[key]['L2'].append(res['errors'][f'{key}_L2'])

    # Conservation
    with torch.no_grad():
        x_q = torch.linspace(X_MIN, X_MAX, 2000, device=device).unsqueeze(1)
        t_q = torch.full_like(x_q, t_val)
        inp = torch.cat([x_q, t_q], dim=1)
        pred = model(inp)
        u_abs2 = pred['u_R']**2 + pred['u_I']**2
        dx = (X_MAX - X_MIN) / 1999
        C = torch_trapz(u_abs2.squeeze(), dx=dx).item()
    temporal_conservation.append(C)

# Convert to arrays
for key in temporal_errors:
    for metric in ['Linf', 'L2']:
        temporal_errors[key][metric] = np.array(temporal_errors[key][metric])
temporal_conservation = np.array(temporal_conservation)

# Print error table
print(f"\n{'t':<6} {'Linf(uR)':<12} {'Linf(uI)':<12} {'Linf(v)':<12} "
      f"{'C(t)':<12} {'dC/C':<12}")
print("-" * 66)
for i, t_val in enumerate(time_snapshots):
    rel_drift = abs(temporal_conservation[i] - C_exact) / C_exact
    print(f"{t_val:<6.2f} "
          f"{temporal_errors['u_R']['Linf'][i]:<12.2e} "
          f"{temporal_errors['u_I']['Linf'][i]:<12.2e} "
          f"{temporal_errors['v']['Linf'][i]:<12.2e} "
          f"{temporal_conservation[i]:<12.6f} "
          f"{rel_drift:<12.2e}")

# Determine error growth type
# Fit log(error) vs t for t >= 0.5 to classify growth
from numpy.polynomial import polynomial as P

t_fit = time_snapshots[time_snapshots >= 0.5]
for key in ['u_R', 'u_I', 'v']:
    e_fit = temporal_errors[key]['Linf'][time_snapshots >= 0.5]
    log_e = np.log(e_fit + 1e-30)

    # Linear fit: log(e) = a + b*t
    coeffs = np.polyfit(t_fit, log_e, 1)
    exp_rate = coeffs[0]

    # Also fit linear: e = a + b*t
    lin_coeffs = np.polyfit(t_fit, e_fit, 1)
    lin_rate = lin_coeffs[0]

    # Residuals for both fits
    log_resid = np.std(log_e - np.polyval(coeffs, t_fit))
    lin_resid = np.std(e_fit - np.polyval(lin_coeffs, t_fit))

    if log_resid < lin_resid * 0.5:
        growth_type = f"EXPONENTIAL (rate={exp_rate:.3f}/unit_t)"
    else:
        growth_type = f"LINEAR (rate={lin_rate:.2e}/unit_t)"

    print(f"\n  {key} error growth: {growth_type}")

# Visualization
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# Panel 1: L_inf error vs time
ax = axes[0]
for key, color, marker in [('u_R', 'blue', 'o'), ('u_I', 'red', 's'),
                            ('v', 'green', '^'), ('p', 'purple', 'D')]:
    ax.semilogy(time_snapshots, temporal_errors[key]['Linf'],
                f'{marker}-', color=color, ms=4, lw=1.2, label=f'{key}')

ax.set_xlabel('Time $t$')
ax.set_ylabel('$L_\\infty$ Error (log scale)')
ax.set_title('Error vs. Time (Temporal Propagation)')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# Panel 2: L2 error vs time
ax = axes[1]
for key, color, marker in [('u_R', 'blue', 'o'), ('u_I', 'red', 's'),
                            ('v', 'green', '^')]:
    ax.semilogy(time_snapshots, temporal_errors[key]['L2'],
                f'{marker}-', color=color, ms=4, lw=1.2, label=f'{key}')
ax.set_xlabel('Time $t$')
ax.set_ylabel('$L_2$ Error (log scale)')
ax.set_title('$L_2$ Error vs. Time')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# Panel 3: Conservation drift
ax = axes[2]
rel_drift = np.abs(temporal_conservation - C_exact) / C_exact
ax.semilogy(time_snapshots, rel_drift, 'ko-', ms=5, lw=1.5)
ax.axhline(y=1e-2, color='orange', ls='--', alpha=0.6, label='1% threshold')
ax.axhline(y=1e-5, color='red', ls='--', alpha=0.6, label='Target ($10^{-5}$)')
ax.set_xlabel('Time $t$')
ax.set_ylabel('Relative Conservation Drift $|\\Delta C/C|$')
ax.set_title('Conservation Quantity Drift')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

plt.suptitle('Task 2.3: Temporal Propagation Analysis', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('phase2_temporal_propagation.png', bbox_inches='tight', dpi=150)
plt.show()
print("Saved: phase2_temporal_propagation.png")


# =============================================================================
# Cell 9: Task 2.4 - Loss Component Imbalance (from Phase 1 History)
# =============================================================================

print("\n" + "-" * 60)
print("TASK 2.4: LOSS COMPONENT IMBALANCE")
print("-" * 60)

# We can also compute current PDE residual magnitudes at different regions
def compute_regional_residuals(model, n_per_region=2000):
    """
    Compute mean PDE residual in early vs. late time regions to
    assess where the PDE is least well-satisfied.
    """
    model.train()
    regions = {
        'early (t<1)': (0.0, 1.0),
        'mid (1<t<3)': (1.0, 3.0),
        'late (3<t<6)': (3.0, 6.0),
    }
    results = {}
    for name, (t_lo, t_hi) in regions.items():
        rng = np.random.RandomState(42)
        x_np = rng.uniform(X_MIN, X_MAX, (n_per_region, 1))
        t_np = rng.uniform(t_lo, t_hi, (n_per_region, 1))

        x = torch.tensor(x_np, dtype=torch.float32, device=device, requires_grad=True)
        t = torch.tensor(t_np, dtype=torch.float32, device=device, requires_grad=True)

        with torch.enable_grad():
            residuals, _ = compute_pde_residuals(model, x, t)

        res_vals = {}
        for rname, rval in residuals.items():
            res_vals[rname] = torch.mean(rval**2).item()
        results[name] = res_vals

    model.eval()
    return results


regional = compute_regional_residuals(model)

print("\nMean Squared PDE Residuals by Time Region:")
print(f"{'Region':<16} {'R_schrod_re':<14} {'R_schrod_im':<14} "
      f"{'R_compat':<14} {'R_kg':<14} {'Total':<14}")
print("-" * 86)
for region, vals in regional.items():
    total = sum(vals.values())
    print(f"{region:<16} "
          f"{vals['R_schrod_re']:<14.2e} "
          f"{vals['R_schrod_im']:<14.2e} "
          f"{vals['R_compat']:<14.2e} "
          f"{vals['R_kg']:<14.2e} "
          f"{total:<14.2e}")

# Dynamic range
for region, vals in regional.items():
    v_list = list(vals.values())
    R = np.log10(max(v_list) / (min(v_list) + 1e-30))
    flag = " << IMBALANCED" if R > 2 else ""
    print(f"  {region}: dynamic range R = {R:.2f} (log10 ratio){flag}")

# Visualization
fig, ax = plt.subplots(figsize=(10, 6))
regions_list = list(regional.keys())
components = ['R_schrod_re', 'R_schrod_im', 'R_compat', 'R_kg']
x_pos = np.arange(len(regions_list))
width = 0.18
colors = ['#2196F3', '#F44336', '#4CAF50', '#FF9800']

for i, comp in enumerate(components):
    vals = [regional[r][comp] for r in regions_list]
    ax.bar(x_pos + i * width, vals, width, label=comp, color=colors[i], alpha=0.8)

ax.set_xticks(x_pos + 1.5 * width)
ax.set_xticklabels(regions_list)
ax.set_ylabel('Mean Squared Residual')
ax.set_title('Task 2.4: PDE Residual by Time Region and Component')
ax.legend()
ax.set_yscale('log')
ax.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig('phase2_residual_imbalance.png', bbox_inches='tight', dpi=150)
plt.show()
print("Saved: phase2_residual_imbalance.png")


# =============================================================================
# Cell 10: Task 2.5 - Conservation Drift Profiling
# =============================================================================

print("\n" + "-" * 60)
print("TASK 2.5: CONSERVATION DRIFT PROFILING")
print("-" * 60)

# Compute conservation at 30 time points for detailed profiling
t_cons = np.linspace(0, T_MAX, 30)
C_values = []
for t_val in t_cons:
    with torch.no_grad():
        x_q = torch.linspace(X_MIN, X_MAX, 3000, device=device).unsqueeze(1)
        t_q = torch.full_like(x_q, t_val)
        inp = torch.cat([x_q, t_q], dim=1)
        pred = model(inp)
        u_abs2 = pred['u_R']**2 + pred['u_I']**2
        dx = (X_MAX - X_MIN) / 2999
        C = torch_trapz(u_abs2.squeeze(), dx=dx).item()
    C_values.append(C)
C_values = np.array(C_values)

drift = C_values - C_exact
rel_drift_full = np.abs(drift) / C_exact

# Drift rate via finite differences
dC_dt = np.gradient(C_values, t_cons)

# Characterize drift type
drift_sign_changes = np.sum(np.diff(np.sign(drift)) != 0)
monotonic = drift_sign_changes <= 2

max_abs_drift = np.max(np.abs(drift))
max_rel_drift = np.max(rel_drift_full)
mean_rel_drift = np.mean(rel_drift_full)

print(f"\nConservation C(t) = integral(|u|^2 dx):")
print(f"  C_exact          = {C_exact:.6f}")
print(f"  C_pinn range     = [{np.min(C_values):.6f}, {np.max(C_values):.6f}]")
print(f"  Max absolute drift    = {max_abs_drift:.2e}")
print(f"  Max relative drift    = {max_rel_drift:.2e}")
print(f"  Mean relative drift   = {mean_rel_drift:.2e}")
print(f"  Sign changes in drift = {drift_sign_changes}")

if monotonic:
    drift_type = "MONOTONIC (systematic bias, likely curable with Lagrange multiplier)"
elif drift_sign_changes > 10:
    drift_type = "OSCILLATORY (training noise, may need tighter loss weighting)"
else:
    drift_type = "MIXED (combination of systematic and oscillatory components)"
print(f"  Drift type: {drift_type}")

# Target check
if max_rel_drift < 1e-5:
    cons_verdict = "PASS: drift < 1e-5 target"
elif max_rel_drift < 1e-2:
    cons_verdict = "SOFT PASS: drift < 1% but above 1e-5 target. Lagrange multiplier needed."
else:
    cons_verdict = "FAIL: drift > 1%. Conservation enforcement is ESSENTIAL."
print(f"  Verdict: {cons_verdict}")

# Visualization
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# Panel 1: C(t) vs t
ax = axes[0]
ax.plot(t_cons, C_values, 'bo-', ms=4, lw=1.5, label='$C_{PINN}(t)$')
ax.axhline(y=C_exact, color='k', ls='--', lw=1.5, alpha=0.7, label=f'$C_{{exact}} = {C_exact:.4f}$')
ax.fill_between(t_cons, C_exact - 0.01*C_exact, C_exact + 0.01*C_exact,
                alpha=0.1, color='green', label='$\\pm$1% band')
ax.set_xlabel('Time $t$')
ax.set_ylabel('$C(t) = \\int |u|^2 dx$')
ax.set_title('Conserved Quantity Over Time')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# Panel 2: Drift (absolute)
ax = axes[1]
ax.plot(t_cons, drift, 'ro-', ms=4, lw=1.5)
ax.axhline(y=0, color='k', ls='-', lw=0.5)
ax.set_xlabel('Time $t$')
ax.set_ylabel('$C(t) - C_{exact}$')
ax.set_title('Absolute Conservation Drift')
ax.grid(True, alpha=0.3)

# Panel 3: Drift rate
ax = axes[2]
ax.plot(t_cons, dC_dt, 'g^-', ms=4, lw=1.2)
ax.axhline(y=0, color='k', ls='-', lw=0.5)
ax.set_xlabel('Time $t$')
ax.set_ylabel('$dC/dt$ (finite difference)')
ax.set_title('Conservation Drift Rate')
ax.grid(True, alpha=0.3)

plt.suptitle('Task 2.5: Conservation Drift Profiling', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('phase2_conservation_drift.png', bbox_inches='tight', dpi=150)
plt.show()
print("Saved: phase2_conservation_drift.png")


# =============================================================================
# Cell 11: Task 2.6 - NTK Eigenvalue Spectrum (Optional)
# =============================================================================

print("\n" + "-" * 60)
print("TASK 2.6: NTK EIGENVALUE SPECTRUM")
print("-" * 60)

def compute_ntk_spectrum(model, n_points=200):
    """
    Compute the empirical NTK eigenvalue spectrum for a subset of
    collocation points. The NTK matrix K_{ij} = sum_p (df_i/dp)(df_j/dp)
    where f_i is the network output at point i and p are parameters.

    Due to memory constraints, we use a small subset and compute
    the Jacobian column by column.

    Returns eigenvalues (sorted descending) and condition number.
    """
    model.eval()
    rng = np.random.RandomState(42)
    x_np = rng.uniform(X_MIN, X_MAX, (n_points, 1))
    t_np = rng.uniform(0.0, T_MAX, (n_points, 1))

    x = torch.tensor(x_np, dtype=torch.float32, device=device)
    t = torch.tensor(t_np, dtype=torch.float32, device=device)
    inp = torch.cat([x, t], dim=1)

    # Compute Jacobian: J[i, p] = d(output_i)/d(param_p)
    # For the PDE residual, we want the Jacobian of the total loss-relevant output
    # Simplified: use the raw network output (4 components per point)
    params = [p for p in model.parameters() if p.requires_grad]
    n_params_total = sum(p.numel() for p in params)

    print(f"  Computing NTK for {n_points} points, {n_params_total} parameters...")
    print(f"  (Jacobian shape: {n_points * 4} x {n_params_total})")

    # Use torch.func.jacrev for efficient Jacobian computation if available
    # Fallback to manual loop for compatibility
    model.train()
    pred = model(inp)
    outputs = torch.cat([pred['u_R'], pred['u_I'], pred['v'], pred['p']], dim=1)  # (N, 4)
    outputs_flat = outputs.flatten()  # (4N,)

    n_out = outputs_flat.shape[0]
    jacobian = torch.zeros(n_out, n_params_total, device=device)

    for i in range(n_out):
        model.zero_grad()
        if i < n_out - 1:
            outputs_flat[i].backward(retain_graph=True)
        else:
            outputs_flat[i].backward(retain_graph=False)

        col_idx = 0
        for p in params:
            if p.grad is not None:
                g = p.grad.flatten()
                jacobian[i, col_idx:col_idx + g.shape[0]] = g
                col_idx += g.shape[0]

    # NTK = J @ J^T
    ntk = jacobian @ jacobian.T  # (4N, 4N)

    # Eigenvalues
    eigenvalues = torch.linalg.eigvalsh(ntk).cpu().numpy()
    eigenvalues = np.sort(eigenvalues)[::-1]  # Descending

    # Condition number
    pos_eigs = eigenvalues[eigenvalues > 1e-15]
    if len(pos_eigs) >= 2:
        kappa = pos_eigs[0] / pos_eigs[-1]
    else:
        kappa = float('inf')

    model.eval()
    return eigenvalues, kappa


# This is expensive; use a small subset
# Set n_points low to keep memory manageable on Kaggle
try:
    ntk_eigenvalues, ntk_kappa = compute_ntk_spectrum(model, n_points=100)
    print(f"  NTK condition number kappa = {ntk_kappa:.2e}")
    if ntk_kappa > 1e6:
        print(f"  Verdict: SEVERE ill-conditioning (kappa > 1e6)")
    elif ntk_kappa > 1e3:
        print(f"  Verdict: MODERATE ill-conditioning")
    else:
        print(f"  Verdict: ACCEPTABLE conditioning")

    # Visualization
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    ax.semilogy(np.arange(len(ntk_eigenvalues)), ntk_eigenvalues + 1e-30, 'b-', lw=1)
    ax.set_xlabel('Eigenvalue Index')
    ax.set_ylabel('Eigenvalue (log scale)')
    ax.set_title(f'NTK Eigenvalue Spectrum ($\\kappa$ = {ntk_kappa:.1e})')
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.semilogy(np.arange(min(50, len(ntk_eigenvalues))),
                ntk_eigenvalues[:50] + 1e-30, 'bo-', ms=3, lw=1)
    ax.set_xlabel('Eigenvalue Index (top 50)')
    ax.set_ylabel('Eigenvalue (log scale)')
    ax.set_title('Top 50 NTK Eigenvalues')
    ax.grid(True, alpha=0.3)

    plt.suptitle('Task 2.6: NTK Eigenvalue Spectrum', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig('phase2_ntk_spectrum.png', bbox_inches='tight', dpi=150)
    plt.show()
    print("Saved: phase2_ntk_spectrum.png")

except Exception as e:
    print(f"  NTK computation skipped due to: {e}")
    print("  (This is expected on memory-limited GPUs. Not critical for Phase 3 planning.)")
    ntk_kappa = None


# =============================================================================
# Cell 12: 3D Error Surface (Novel Diagnostic)
# =============================================================================

print("\n" + "-" * 60)
print("3D ERROR SURFACE")
print("-" * 60)

def compute_error_surface(model, kgs_exact, nx=200, nt=150):
    """Compute |error(x,t)| over the full space-time domain."""
    x_grid = np.linspace(X_MIN, X_MAX, nx)
    t_grid = np.linspace(0, T_MAX, nt)
    X, T = np.meshgrid(x_grid, t_grid)

    x_flat = X.flatten()
    t_flat = T.flatten()

    model.eval()
    with torch.no_grad():
        x_t = torch.tensor(x_flat.reshape(-1, 1), dtype=torch.float32, device=device)
        t_t = torch.tensor(t_flat.reshape(-1, 1), dtype=torch.float32, device=device)
        inp = torch.cat([x_t, t_t], dim=1)

        # Process in batches to avoid OOM
        batch_size = 10000
        all_preds = {k: [] for k in ['u_R', 'u_I', 'v']}
        for start in range(0, len(x_flat), batch_size):
            end = min(start + batch_size, len(x_flat))
            pred = model(inp[start:end])
            for k in all_preds:
                all_preds[k].append(pred[k].cpu().numpy())

    for k in all_preds:
        all_preds[k] = np.concatenate(all_preds[k]).flatten()

    errors = {}
    for key in ['u_R', 'u_I', 'v']:
        if key == 'u_R':
            exact_flat = kgs_exact.u_real(x_flat, t_flat)
        elif key == 'u_I':
            exact_flat = kgs_exact.u_imag(x_flat, t_flat)
        else:
            exact_flat = kgs_exact.v(x_flat, t_flat)
        errors[key] = np.abs(all_preds[key] - exact_flat).reshape(nt, nx)

    return X, T, errors


X_grid, T_grid, error_surfaces = compute_error_surface(model, kgs)

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for i, (key, title) in enumerate([('u_R', 'Re($u$)'), ('u_I', 'Im($u$)'), ('v', '$v$')]):
    ax = axes[i]
    c = ax.pcolormesh(X_grid, T_grid, error_surfaces[key],
                      cmap='hot', shading='auto')
    plt.colorbar(c, ax=ax, label='$|error|$')

    # Overlay soliton trajectory
    t_line = np.linspace(0, T_MAX, 100)
    x_soliton = X_0 + NU_0 * t_line
    ax.plot(x_soliton, t_line, 'c--', lw=1.5, alpha=0.7, label='Soliton center')

    ax.set_xlabel('$x$')
    ax.set_ylabel('$t$')
    ax.set_title(f'Error in {title}')
    ax.legend(fontsize=8, loc='upper left')

plt.suptitle('3D Error Surface Over $(x, t)$ Domain\n(Novel visualization for publication)',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('phase2_error_surface_3d.png', bbox_inches='tight', dpi=150)
plt.show()
print("Saved: phase2_error_surface_3d.png")


# =============================================================================
# Cell 13: Task 2.7 - Diagnostic Summary & Phase 3 Prescriptions
# =============================================================================

print("\n" + "=" * 80)
print("PHASE 2: DIAGNOSTIC SUMMARY REPORT")
print("=" * 80)

# Collect all diagnostic findings
findings = []

# 1. Gradient stiffness
findings.append({
    'mode': 'Gradient Stiffness',
    'severity': stiffness,
    'threshold': 100,
    'status': 'MILD' if stiffness < 10 else ('MODERATE' if stiffness < 100 else 'SEVERE'),
    'detail': f'S = {stiffness:.1f} (threshold: 100)',
    'prescription': 'Adaptive weighting (LRA) recommended if S > 10' if stiffness > 10
                    else 'No immediate action needed; uniform weighting is acceptable',
})

# 2. Spectral bias
# Aggregate k* across times and components
k_stars = []
for t_val in spectral_times:
    sp = spectral_results[t_val]['spectra']
    for key in ['u_R', 'u_I', 'v']:
        k_stars.append(sp[key]['k_star'])
mean_k_star = np.mean(k_stars)

# Check error concentration
res_t1 = spectral_results[1.0]['spatial']
x_eval = res_t1['x']
soliton_center_t1 = NU_0 * 1.0
inner_mask = np.abs(x_eval - soliton_center_t1) < 3.0
frac_inner = []
for key in ['u_R', 'u_I', 'v']:
    err = np.abs(res_t1['preds'][key] - res_t1['exact'][key])
    f = np.sum(err[inner_mask]) / (np.sum(err) + 1e-30)
    frac_inner.append(f)
mean_frac = np.mean(frac_inner)

spectral_severity = 5.0 if mean_frac > 0.7 else (3.0 if mean_frac > 0.4 else 1.0)
findings.append({
    'mode': 'Spectral Bias',
    'severity': spectral_severity,
    'threshold': 3.0,
    'status': 'CONCENTRATED' if mean_frac > 0.5 else 'DISTRIBUTED',
    'detail': f'{mean_frac*100:.0f}% of error near soliton, mean k* = {mean_k_star:.1f}',
    'prescription': 'Fourier feature embeddings or SIREN activation to resolve sharp sech^2 gradients'
                    if mean_frac > 0.4 else 'Spectral bias is mild; standard tanh may suffice',
})

# 3. Temporal propagation
max_Linf_t6 = max(
    temporal_errors['u_R']['Linf'][-1],
    temporal_errors['u_I']['Linf'][-1],
    temporal_errors['v']['Linf'][-1]
)
error_ratio_t6_t1 = max_Linf_t6 / max(
    temporal_errors['u_R']['Linf'][time_snapshots == 1.0][0],
    temporal_errors['u_I']['Linf'][time_snapshots == 1.0][0],
    temporal_errors['v']['Linf'][time_snapshots == 1.0][0],
)
temporal_severity = error_ratio_t6_t1

findings.append({
    'mode': 'Temporal Propagation',
    'severity': temporal_severity,
    'threshold': 10.0,
    'status': 'STABLE' if error_ratio_t6_t1 < 5 else ('DEGRADING' if error_ratio_t6_t1 < 20 else 'CATASTROPHIC'),
    'detail': f'Error ratio t=6/t=1 = {error_ratio_t6_t1:.1f}x, max Linf at t=6 = {max_Linf_t6:.2e}',
    'prescription': 'Causal training (epsilon sweep)' if error_ratio_t6_t1 > 5
                    else 'Temporal propagation is acceptable; causal training optional',
})

# 4. Conservation drift
cons_severity = max_rel_drift * 1000  # Scale for ranking
findings.append({
    'mode': 'Conservation Drift',
    'severity': cons_severity,
    'threshold': 0.01,  # 1% in relative terms
    'status': 'MILD' if max_rel_drift < 1e-3 else ('MODERATE' if max_rel_drift < 1e-2 else 'SEVERE'),
    'detail': f'Max relative drift = {max_rel_drift:.2e}, type = {"monotonic" if monotonic else "oscillatory"}',
    'prescription': 'Lagrange multiplier enforcement to achieve delta_C < 1e-5',
})

# 5. Loss imbalance
max_regional_R = 0
for region, vals in regional.items():
    v_list = list(vals.values())
    R = np.log10(max(v_list) / (min(v_list) + 1e-30))
    max_regional_R = max(max_regional_R, R)

findings.append({
    'mode': 'Loss Imbalance',
    'severity': max_regional_R,
    'threshold': 2.0,
    'status': 'ACCEPTABLE' if max_regional_R < 2 else 'IMBALANCED',
    'detail': f'Max dynamic range R = {max_regional_R:.2f} (log10 ratio)',
    'prescription': 'ReLoBRaLo softmax weighting' if max_regional_R > 2
                    else 'Loss balance is acceptable',
})

# NTK conditioning (if computed)
if ntk_kappa is not None:
    findings.append({
        'mode': 'NTK Conditioning',
        'severity': np.log10(ntk_kappa) if ntk_kappa > 0 else 0,
        'threshold': 6.0,
        'status': 'WELL-CONDITIONED' if ntk_kappa < 1e3 else
                 ('MODERATE' if ntk_kappa < 1e6 else 'ILL-CONDITIONED'),
        'detail': f'kappa = {ntk_kappa:.2e}',
        'prescription': 'NTK-based weighting if kappa > 1e6' if ntk_kappa > 1e6
                        else 'No action needed',
    })

# Sort by severity (highest first)
findings.sort(key=lambda x: -x['severity'])

print("\n" + "=" * 80)
print("RANKED FAILURE MODES (highest severity first)")
print("=" * 80)
for rank, f in enumerate(findings, 1):
    print(f"\n{'='*60}")
    print(f"  #{rank}: {f['mode']}")
    print(f"  Status:       {f['status']}")
    print(f"  Detail:       {f['detail']}")
    print(f"  Prescription: {f['prescription']}")

# Phase 3 technique priority
print("\n" + "=" * 80)
print("PHASE 3 TECHNIQUE PRIORITY (based on diagnostics)")
print("=" * 80)

techniques = [
    ('Causal Training', temporal_severity > 5,
     'ESSENTIAL' if temporal_severity > 10 else ('RECOMMENDED' if temporal_severity > 5 else 'OPTIONAL')),
    ('Fourier Features / SIREN', mean_frac > 0.4,
     'ESSENTIAL' if mean_frac > 0.7 else ('RECOMMENDED' if mean_frac > 0.4 else 'OPTIONAL')),
    ('Lagrange Multiplier (Conservation)', max_rel_drift > 1e-4,
     'ESSENTIAL' if max_rel_drift > 1e-2 else ('RECOMMENDED' if max_rel_drift > 1e-4 else 'OPTIONAL')),
    ('Adaptive Weighting (LRA)', stiffness > 10,
     'RECOMMENDED' if stiffness > 10 else 'OPTIONAL'),
    ('ReLoBRaLo Softmax', max_regional_R > 2,
     'RECOMMENDED' if max_regional_R > 2 else 'OPTIONAL'),
]

print(f"\n{'Technique':<35} {'Priority':<15} {'Reason':<40}")
print("-" * 90)
for name, needed, priority in techniques:
    reason = "Diagnostic threshold exceeded" if needed else "Below diagnostic threshold"
    print(f"{name:<35} {priority:<15} {reason}")


# =============================================================================
# Cell 14: Save Phase 2 Diagnostics
# =============================================================================

phase2_results = {
    'gradient_norms': grad_norms,
    'stiffness': stiffness,
    'spectral_results': {
        t: {key: {
            'k_star': spectral_results[t]['spectra'][key]['k_star'],
        } for key in ['u_R', 'u_I', 'v']}
        for t in spectral_times
    },
    'temporal_errors': {
        key: {
            'times': time_snapshots.tolist(),
            'Linf': temporal_errors[key]['Linf'].tolist(),
            'L2': temporal_errors[key]['L2'].tolist(),
        } for key in ['u_R', 'u_I', 'v', 'p']
    },
    'conservation': {
        't': t_cons.tolist(),
        'C': C_values.tolist(),
        'C_exact': C_exact,
        'max_rel_drift': max_rel_drift,
        'drift_type': 'monotonic' if monotonic else 'oscillatory',
    },
    'regional_residuals': regional,
    'ntk_kappa': ntk_kappa,
    'findings_ranked': [
        {'mode': f['mode'], 'status': f['status'], 'detail': f['detail'],
         'prescription': f['prescription']}
        for f in findings
    ],
    'error_surfaces': {
        'X': X_grid.tolist(),
        'T': T_grid.tolist(),
        'uR': error_surfaces['u_R'].tolist(),
        'uI': error_surfaces['u_I'].tolist(),
        'v':  error_surfaces['v'].tolist(),
    },
}

import json
diag_path = 'phase2_diagnostics.json'
with open(diag_path, 'w') as f:
    json.dump(phase2_results, f, indent=2, default=str)
print(f"\nPhase 2 diagnostics saved: {diag_path}")
print(f"  File size: {os.path.getsize(diag_path) / 1024:.1f} KB")


# =============================================================================
# Cell 15: Final Summary for Phase 3 Planning
# =============================================================================

print("\n" + "=" * 80)
print("PHASE 2 COMPLETE: READY FOR PHASE 3")
print("=" * 80)
print(f"""
Baseline Performance:
  L_inf(u_R) at t=1: {temporal_errors['u_R']['Linf'][time_snapshots == 1.0][0]:.2e}
  L_inf(u_I) at t=1: {temporal_errors['u_I']['Linf'][time_snapshots == 1.0][0]:.2e}
  L_inf(v)   at t=1: {temporal_errors['v']['Linf'][time_snapshots == 1.0][0]:.2e}

Top 3 Failure Modes:
  1. {findings[0]['mode']}: {findings[0]['status']} ({findings[0]['detail']})
  2. {findings[1]['mode']}: {findings[1]['status']} ({findings[1]['detail']})
  3. {findings[2]['mode']}: {findings[2]['status']} ({findings[2]['detail']})

Recommended Phase 3 Configuration (first attempt):
  - Causal training with epsilon sweep [1, 10, 50, 100]
  - Fourier feature embeddings (sigma=5) or SIREN (omega=30)
  - Lagrange multiplier for conservation (mu trainable)
  - LRA adaptive weighting (if stiffness > 10)
  - All combined: train 70K Adam + 10K L-BFGS

Target:
  - Conservation drift < 1e-5
  - Stable propagation to t=6.0
""")
print("=" * 80)
