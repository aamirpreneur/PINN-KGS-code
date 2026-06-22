"""
generate_phase2_figures.py
==========================
Generates the three diagnostic figures for Section 4 of the paper.
Loads phase1_baseline_checkpoint.pt (no GPU needed, runs on CPU).

Produces (600 dpi, PDF + PNG):
  - fig4_temporal_propagation.pdf   Error grows over time (baseline only)
  - fig5_loss_imbalance.pdf         PDE vs IC vs BC loss magnitudes during training
  - (conservation drift already covered by fig3)
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import os

device = torch.device('cpu')
X_MIN, X_MAX = -10.0, 10.0

# ── exact solution ─────────────────────────────────────────────────────────
class KGSExact:
    def __init__(self, nu=0.8, x0=0.0):
        self.nu   = nu; self.x0 = x0
        self.omn2 = 1.0 - nu**2
        self.sqf  = np.sqrt(self.omn2)
        self.Au   = 3.0 * np.sqrt(2.0) / (4.0 * self.sqf)
        self.Av   = 3.0 / (4.0 * self.omn2)

    def _eta(self, x, t): return (x - self.nu*t - self.x0) / (2.0*self.sqf)
    def _phi(self, x, t): return self.nu*x + (1-self.nu**2+self.nu**4)/(2.0*self.omn2)*t
    def u_real(self, x, t): return self.Au/np.cosh(self._eta(x,t))**2 * np.cos(self._phi(x,t))
    def u_imag(self, x, t): return self.Au/np.cosh(self._eta(x,t))**2 * np.sin(self._phi(x,t))
    def v(self, x, t):      return self.Av/np.cosh(self._eta(x,t))**2
    def p(self, x, t):
        eta = self._eta(x,t)
        return 3.0*self.nu/(4.0*self.omn2**1.5)/np.cosh(eta)**2*np.tanh(eta)

kgs = KGSExact()

# ── network (must match checkpoint) ──────────────────────────────────────
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
        return {'u_R': out[:,0:1], 'u_I': out[:,1:2],
                'v':   out[:,2:3], 'p':   out[:,3:4]}

# ── load baseline ─────────────────────────────────────────────────────────
CKPT = 'phase1_baseline_checkpoint.pt'
print(f"Loading {CKPT} ...")
ckpt = torch.load(CKPT, map_location=device, weights_only=False)
model = PINN().to(device)
key = 'model_state_dict'
model.load_state_dict(ckpt[key] if key in ckpt else ckpt)
model.eval()
print("Model loaded.")

os.makedirs('paper_figures', exist_ok=True)

# ── helpers ────────────────────────────────────────────────────────────────
x_np = np.linspace(X_MIN, X_MAX, 1000)

def predict(t_val):
    with torch.no_grad():
        x_t = torch.tensor(x_np.reshape(-1,1), dtype=torch.float32)
        t_t = torch.full_like(x_t, t_val)
        pred = model(torch.cat([x_t, t_t], dim=1))
    return {k: pred[k].numpy().flatten() for k in ['u_R','u_I','v']}

def linf(a, b): return float(np.max(np.abs(a - b)))

# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 4 — Temporal propagation failure
# Shows error growing with time for the baseline PINN
# ═══════════════════════════════════════════════════════════════════════════
time_pts = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0])
print("Computing temporal error profile ...")

err = {'u_R': [], 'u_I': [], 'v': []}
for t in time_pts:
    pred = predict(t)
    err['u_R'].append(linf(pred['u_R'], kgs.u_real(x_np, t)))
    err['u_I'].append(linf(pred['u_I'], kgs.u_imag(x_np, t)))
    err['v'].append(  linf(pred['v'],   kgs.v(x_np, t)))

plt.rcParams.update({'font.size': 11, 'axes.labelsize': 11, 'legend.fontsize': 9})

fig, axes = plt.subplots(1, 3, figsize=(13, 4))
comp_labels = [r'$u_R$', r'$u_I$', r'$v$']
for i, (c, lbl) in enumerate(zip(['u_R','u_I','v'], comp_labels)):
    ax = axes[i]
    ax.semilogy(time_pts, err[c], 'k-o', ms=5, lw=1.5)
    ax.set_xlabel(r'Time $t$')
    ax.set_ylabel(r'$L_\infty$ error')
    ax.set_title(lbl)
    ax.grid(True, alpha=0.3)

plt.suptitle('Temporal propagation failure: error grows with time (baseline PINN)',
             fontsize=11, y=1.02)
plt.tight_layout()
plt.savefig('paper_figures/fig4_temporal_propagation.pdf', dpi=600, bbox_inches='tight')
plt.savefig('paper_figures/fig4_temporal_propagation.png', dpi=600, bbox_inches='tight')
plt.close()
print("Saved fig4_temporal_propagation")

# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 5 — Loss imbalance
# Re-evaluates the three loss components at each training step
# using the saved checkpoint's training history (if available),
# otherwise reconstructs from checkpoint by scanning loss magnitudes.
# ═══════════════════════════════════════════════════════════════════════════
print("\nChecking checkpoint for training history ...")
history_keys = [k for k in ckpt.keys() if 'history' in k.lower() or 'loss' in k.lower() or 'log' in k.lower()]
print(f"History keys found: {history_keys}")

if history_keys:
    # Use recorded loss history from checkpoint
    hist = ckpt.get('history', ckpt.get('loss_history', None))
    if hist and 'pde' in hist and 'ic' in hist and 'bc' in hist:
        epochs = np.arange(len(hist['pde']))
        L_pde = np.array(hist['pde'])
        L_ic  = np.array(hist['ic'])
        L_bc  = np.array(hist['bc'])

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.semilogy(epochs, L_pde, 'b-',  lw=1.5, label=r'$\mathcal{L}_{\rm PDE}$')
        ax.semilogy(epochs, L_ic,  'r--', lw=1.5, label=r'$\mathcal{L}_{\rm IC}$')
        ax.semilogy(epochs, L_bc,  'g:',  lw=1.5, label=r'$\mathcal{L}_{\rm BC}$')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('Loss imbalance: PDE residual dominates training')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('paper_figures/fig5_loss_imbalance.pdf', dpi=600, bbox_inches='tight')
        plt.savefig('paper_figures/fig5_loss_imbalance.png', dpi=600, bbox_inches='tight')
        plt.close()
        print("Saved fig5_loss_imbalance (from checkpoint history)")
    else:
        print("History key found but structure not recognised. Printing keys:")
        if isinstance(hist, dict):
            print(list(hist.keys()))
else:
    # No history in checkpoint — reconstruct loss magnitudes at a single snapshot
    print("No training history in checkpoint.")
    print("Generating loss-scale comparison at t=0 snapshot instead...")

    from scipy.integrate import trapezoid as trapz

    # Evaluate each loss component once on a fresh collocation set
    rng = np.random.default_rng(42)
    N_f = 5000

    # Collocation points
    x_col = rng.uniform(X_MIN, X_MAX, N_f)
    t_col = rng.uniform(0.0, 6.0,    N_f)
    x_ct  = torch.tensor(x_col.reshape(-1,1), dtype=torch.float32, requires_grad=True)
    t_ct  = torch.tensor(t_col.reshape(-1,1), dtype=torch.float32, requires_grad=True)

    inp = torch.cat([x_ct, t_ct], dim=1)
    pred = model(inp)
    uR, uI, v, p = pred['u_R'], pred['u_I'], pred['v'], pred['p']
    ones = torch.ones_like(uR)

    uR_x = torch.autograd.grad(uR, x_ct, ones, create_graph=False, retain_graph=True)[0]
    uR_t = torch.autograd.grad(uR, t_ct, ones, create_graph=False, retain_graph=True)[0]
    uI_x = torch.autograd.grad(uI, x_ct, ones, create_graph=False, retain_graph=True)[0]
    uI_t = torch.autograd.grad(uI, t_ct, ones, create_graph=False, retain_graph=True)[0]
    v_t  = torch.autograd.grad(v,  t_ct, ones, create_graph=False, retain_graph=True)[0]
    v_x  = torch.autograd.grad(v,  x_ct, ones, create_graph=False, retain_graph=True)[0]
    p_t  = torch.autograd.grad(p,  t_ct, ones, create_graph=False, retain_graph=True)[0]
    uR_xx = torch.autograd.grad(uR_x, x_ct, ones, create_graph=False, retain_graph=True)[0]
    uI_xx = torch.autograd.grad(uI_x, x_ct, ones, create_graph=False, retain_graph=True)[0]
    v_xx  = torch.autograd.grad(v_x,  x_ct, ones, create_graph=False)[0]

    R1 = -uI_t + 0.5*uR_xx + uR*v
    R2 =  uR_t + 0.5*uI_xx + uI*v
    R3 =  v_t  - p
    R4 =  p_t  - v_xx + v - (uR**2 + uI**2)
    L_pde = (torch.mean(R1**2)+torch.mean(R2**2)+torch.mean(R3**2)+torch.mean(R4**2)).item()/4

    # IC loss
    x_ic = torch.tensor(np.linspace(X_MIN,X_MAX,2000).reshape(-1,1), dtype=torch.float32)
    t_ic = torch.zeros_like(x_ic)
    with torch.no_grad():
        p_ic = model(torch.cat([x_ic,t_ic],dim=1))
    x_np_ic = x_ic.numpy().flatten()
    L_ic = (torch.mean((p_ic['u_R'].squeeze()-torch.tensor(kgs.u_real(x_np_ic,0),dtype=torch.float32))**2)
           +torch.mean((p_ic['u_I'].squeeze()-torch.tensor(kgs.u_imag(x_np_ic,0),dtype=torch.float32))**2)
           +torch.mean((p_ic['v'].squeeze()  -torch.tensor(kgs.v(x_np_ic,0),     dtype=torch.float32))**2)
           +torch.mean((p_ic['p'].squeeze()  -torch.tensor(kgs.p(x_np_ic,0),     dtype=torch.float32))**2)).item()/4

    # BC loss (homogeneous — predictions at boundaries should be ~0)
    t_bc = torch.tensor(np.linspace(0,6,500).reshape(-1,1), dtype=torch.float32)
    x_left  = torch.full_like(t_bc, X_MIN)
    x_right = torch.full_like(t_bc, X_MAX)
    with torch.no_grad():
        p_left  = model(torch.cat([x_left,  t_bc], dim=1))
        p_right = model(torch.cat([x_right, t_bc], dim=1))
    L_bc = sum(torch.mean(p_left[k]**2)+torch.mean(p_right[k]**2)
               for k in ['u_R','u_I','v','p']).item()/8

    print(f"\nLoss magnitudes at checkpoint:")
    print(f"  L_PDE = {L_pde:.4e}")
    print(f"  L_IC  = {L_ic:.4e}")
    print(f"  L_BC  = {L_bc:.4e}")
    print(f"  Ratio PDE/IC  = {L_pde/max(L_ic,1e-12):.1f}x")
    print(f"  Ratio PDE/BC  = {L_pde/max(L_bc,1e-12):.1f}x")

    # Bar chart showing loss scale disparity
    fig, ax = plt.subplots(figsize=(6, 5))
    names  = [r'$\mathcal{L}_{\rm PDE}$',
              r'$\mathcal{L}_{\rm IC}$',
              r'$\mathcal{L}_{\rm BC}$']
    values = [L_pde, L_ic, L_bc]
    colors = ['steelblue', 'tomato', 'seagreen']
    bars = ax.bar(names, values, color=colors, width=0.5, log=True)
    ax.set_ylabel('Loss magnitude (log scale)')
    ax.set_title('Loss imbalance: PDE residual dominates')
    ax.grid(True, axis='y', alpha=0.3)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x()+bar.get_width()/2, val*1.5,
                f'{val:.1e}', ha='center', va='bottom', fontsize=10)
    plt.tight_layout()
    plt.savefig('paper_figures/fig5_loss_imbalance.pdf', dpi=600, bbox_inches='tight')
    plt.savefig('paper_figures/fig5_loss_imbalance.png', dpi=600, bbox_inches='tight')
    plt.close()
    print("Saved fig5_loss_imbalance (snapshot bar chart)")

print("\nDone. Files in paper_figures/")
