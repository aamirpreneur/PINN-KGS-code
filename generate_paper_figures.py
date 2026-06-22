"""
generate_paper_figures.py
=========================
Run this on Kaggle to produce publication-ready figures for the paper.

Requires:
  - phase1_baseline_checkpoint.pt   (baseline model)
  - phase4b_final_checkpoint.pt     (pipeline model)

Produces (at 600 dpi, PDF + PNG):
  - fig1_solution_profiles.pdf      Section 6: exact vs pipeline at t=0.5, 1.0
  - fig2_error_comparison.pdf       Section 6: error over time, baseline vs pipeline
  - fig3_conservation.pdf           Section 6: conservation drift, baseline vs pipeline
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import os

# ── device ───────────────────────────────────────────────────────────────────
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

X_MIN, X_MAX = -10.0, 10.0

# ── exact solution ────────────────────────────────────────────────────────────
class KGSExact:
    def __init__(self, nu=0.8, x0=0.0):
        self.nu   = nu
        self.x0   = x0
        self.omn2 = 1.0 - nu**2
        self.sqf  = np.sqrt(self.omn2)
        self.Au   = 3.0 * np.sqrt(2.0) / (4.0 * self.sqf)
        self.Av   = 3.0 / (4.0 * self.omn2)

    def _eta(self, x, t):
        return (x - self.nu * t - self.x0) / (2.0 * self.sqf)

    def _phi(self, x, t):
        return self.nu * x + (1.0 - self.nu**2 + self.nu**4) / (2.0 * self.omn2) * t

    def u_real(self, x, t): return self.Au / np.cosh(self._eta(x, t))**2 * np.cos(self._phi(x, t))
    def u_imag(self, x, t): return self.Au / np.cosh(self._eta(x, t))**2 * np.sin(self._phi(x, t))
    def v(self, x, t):      return self.Av / np.cosh(self._eta(x, t))**2
    def p(self, x, t):
        eta = self._eta(x, t)
        return 3.0*self.nu / (4.0*self.omn2**1.5) / np.cosh(eta)**2 * np.tanh(eta)

    def C(self, x_grid, t):
        return np.trapz(self.Au**2 / np.cosh(self._eta(x_grid, t))**4, x_grid)

kgs = KGSExact()
C_exact = kgs.C(np.linspace(X_MIN, X_MAX, 5000), 0.0)
print(f"C_exact = {C_exact:.6f}")

# ── network ───────────────────────────────────────────────────────────────────
class PINN(nn.Module):
    def __init__(self, n_hidden=6, n_neurons=128):
        super().__init__()
        layers = [nn.Linear(2, n_neurons), nn.Tanh()]
        for _ in range(n_hidden - 1):
            layers += [nn.Linear(n_neurons, n_neurons), nn.Tanh()]
        layers.append(nn.Linear(n_neurons, 4))
        self.network = nn.Sequential(*layers)

    def forward(self, xt):
        out = self.network(xt)
        return {'u_R': out[:, 0:1], 'u_I': out[:, 1:2],
                'v':   out[:, 2:3], 'p':   out[:, 3:4]}

def load_model(path):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    m = PINN().to(device)
    key = 'model_state_dict'
    m.load_state_dict(ckpt[key] if key in ckpt else ckpt)
    m.eval()
    return m

# ── evaluation helpers ────────────────────────────────────────────────────────
def predict(model, x_np, t_val):
    with torch.no_grad():
        x_t = torch.tensor(x_np.reshape(-1, 1), dtype=torch.float32, device=device)
        t_t = torch.full_like(x_t, t_val)
        pred = model(torch.cat([x_t, t_t], dim=1))
    return (pred['u_R'].cpu().numpy().flatten(),
            pred['u_I'].cpu().numpy().flatten(),
            pred['v'].cpu().numpy().flatten())

def linf(pred, exact):
    return float(np.max(np.abs(pred - exact)))

def conservation(model, t_val, n=3000):
    with torch.no_grad():
        x_t = torch.linspace(X_MIN, X_MAX, n, device=device).unsqueeze(1)
        t_t = torch.full_like(x_t, t_val)
        pred = model(torch.cat([x_t, t_t], dim=1))
        u2 = pred['u_R']**2 + pred['u_I']**2
        return torch.trapezoid(u2.squeeze(), dx=(X_MAX-X_MIN)/(n-1)).item()

# ── load models ───────────────────────────────────────────────────────────────
BASELINE_CKPT = 'phase1_baseline_checkpoint.pt'
PIPELINE_CKPT = 'phase4b_final_checkpoint.pt'

print("Loading baseline model...")
baseline = load_model(BASELINE_CKPT)
print("Loading pipeline model...")
pipeline = load_model(PIPELINE_CKPT)

os.makedirs('paper_figures', exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 1 — Solution profiles: exact vs pipeline at t = 0.5 and t = 1.0
# ─────────────────────────────────────────────────────────────────────────────
x_np = np.linspace(X_MIN, X_MAX, 1000)

plt.rcParams.update({'font.size': 11, 'axes.labelsize': 11,
                     'legend.fontsize': 9, 'axes.titlesize': 11})

fig, axes = plt.subplots(3, 2, figsize=(10, 10))
labels = [r'$\mathrm{Re}(u)$', r'$\mathrm{Im}(u)$', r'$v$']
ylabels = [r'$u_R$', r'$u_I$', r'$v$']
times = [0.5, 1.0]

for col, t in enumerate(times):
    uR_ex = kgs.u_real(x_np, t)
    uI_ex = kgs.u_imag(x_np, t)
    v_ex  = kgs.v(x_np, t)
    exact_vals = [uR_ex, uI_ex, v_ex]

    uR_pp, uI_pp, v_pp = predict(pipeline, x_np, t)
    pred_vals = [uR_pp, uI_pp, v_pp]

    for row in range(3):
        ax = axes[row, col]
        ax.plot(x_np, exact_vals[row], 'b-',  lw=1.5, label='Exact')
        ax.plot(x_np, pred_vals[row],  'r--', lw=1.5, label='PINN')
        ax.set_xlabel(r'$x$')
        ax.set_ylabel(ylabels[row])
        ax.set_title(f'{labels[row]},  $t = {t}$')
        if row == 0 and col == 0:
            ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('paper_figures/fig1_solution_profiles.pdf', dpi=600, bbox_inches='tight')
plt.savefig('paper_figures/fig1_solution_profiles.png', dpi=600, bbox_inches='tight')
plt.close()
print("Saved fig1_solution_profiles")

# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 2 — Error over time: baseline vs pipeline (L-inf of u_R, u_I, v)
# ─────────────────────────────────────────────────────────────────────────────
time_pts = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0])
components = ['u_R', 'u_I', 'v']
comp_labels = [r'$u_R$', r'$u_I$', r'$v$']

err_base = {c: [] for c in components}
err_pipe = {c: [] for c in components}

for t in time_pts:
    uR_ex = kgs.u_real(x_np, t)
    uI_ex = kgs.u_imag(x_np, t)
    v_ex  = kgs.v(x_np, t)
    exact = {'u_R': uR_ex, 'u_I': uI_ex, 'v': v_ex}

    uR_b, uI_b, v_b = predict(baseline, x_np, t)
    uR_p, uI_p, v_p = predict(pipeline, x_np, t)
    pred_base = {'u_R': uR_b, 'u_I': uI_b, 'v': v_b}
    pred_pipe = {'u_R': uR_p, 'u_I': uI_p, 'v': v_p}

    for c in components:
        err_base[c].append(linf(pred_base[c], exact[c]))
        err_pipe[c].append(linf(pred_pipe[c], exact[c]))

fig, axes = plt.subplots(1, 3, figsize=(13, 4))
for i, (c, lbl) in enumerate(zip(components, comp_labels)):
    ax = axes[i]
    ax.semilogy(time_pts, err_base[c], 'k--o', ms=4, lw=1.2, label='Baseline')
    ax.semilogy(time_pts, err_pipe[c], 'b-s',  ms=4, lw=1.5, label='Pipeline')
    ax.set_xlabel(r'Time $t$')
    ax.set_ylabel(r'$L_\infty$ error')
    ax.set_title(lbl)
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('paper_figures/fig2_error_comparison.pdf', dpi=600, bbox_inches='tight')
plt.savefig('paper_figures/fig2_error_comparison.png', dpi=600, bbox_inches='tight')
plt.close()
print("Saved fig2_error_comparison")

# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 3 — Conservation drift: baseline vs pipeline
# ─────────────────────────────────────────────────────────────────────────────
cons_base = [conservation(baseline, t) for t in time_pts]
cons_pipe = [conservation(pipeline, t) for t in time_pts]

drift_base = [abs(c - C_exact) for c in cons_base]
drift_pipe = [abs(c - C_exact) for c in cons_pipe]

fig, ax = plt.subplots(figsize=(7, 4.5))
ax.semilogy(time_pts, drift_base, 'k--o', ms=4, lw=1.2, label='Baseline')
ax.semilogy(time_pts, drift_pipe, 'b-s',  ms=4, lw=1.5, label='Pipeline')
ax.set_xlabel(r'Time $t$')
ax.set_ylabel(r'$|\mathcal{C}(t) - \mathcal{C}_{\rm exact}|$')
ax.set_title('Nucleon number conservation drift')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('paper_figures/fig3_conservation.pdf', dpi=600, bbox_inches='tight')
plt.savefig('paper_figures/fig3_conservation.png', dpi=600, bbox_inches='tight')
plt.close()
print("Saved fig3_conservation")

print("\nAll figures saved in paper_figures/ folder.")
print("Files: fig1_solution_profiles, fig2_error_comparison, fig3_conservation")
print("Each saved as both .pdf (for LaTeX) and .png (for preview).")
