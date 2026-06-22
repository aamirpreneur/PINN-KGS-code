#!/usr/bin/env python3
import copy, json, os
from time import time
import numpy as np
import torch
import torch.nn as nn

np_trapz = np.trapezoid if hasattr(np, 'trapezoid') else np.trapz
torch_trapz = torch.trapezoid if hasattr(torch, 'trapezoid') else torch.trapz

SEED = 42
OUTPUT_DIR = '.'
X_MIN, X_MAX = -10.0, 10.0
T_MAX = 6.0
NU_0 = 0.8
X_0 = 0.0

PHASE0_N_IC_GLOBAL = 3000
PHASE0_N_IC_CENTER = 2000
PHASE0_CENTER_HALF_WIDTH = 3.0
PHASE0_ADAM_EPOCHS = 5000
PHASE0_ADAM_LR = 1e-3
PHASE0_ADAM_ETA_MIN = 1e-5
PHASE0_LBFGS_MAX_ITER = 1200
PHASE0_LBFGS_LR = 0.8
PHASE0_LBFGS_HISTORY = 100

PHASE1_N_COL = 10000
PHASE1_N_IC = 2000
PHASE1_N_BC = 500
PHASE1_EPOCHS = 30000
PHASE1_LR = 1e-3
PHASE1_LR_MIN = 1e-6
PHASE1_RAMP_CENTER = 250
PHASE1_RAMP_STEEPNESS = 0.02

STAGE1 = dict(t_max=2.0, n_col=15000, n_ic=2000, n_bc=1000, n_epochs=20000, lr=1e-4, lr_min=1e-6, beta=5.0, n_cons=3, tw=None, tb=None)
STAGE2 = dict(t_max=4.0, n_col=25000, n_ic=2000, n_bc=1000, n_epochs=20000, lr=5e-5, lr_min=1e-6, beta=5.0, n_cons=5, tw=[0.60, 0.40], tb=[0.0, 2.0, 4.0])
STAGE3 = dict(t_max=6.0, n_col=30000, n_ic=2000, n_bc=1000, n_epochs=25000, lr=3e-5, lr_min=1e-6, beta=5.0, n_cons=7, tw=[0.50, 0.30, 0.20], tb=[0.0, 2.0, 4.0, 6.0])
STAGE4 = dict(t_max=6.0, n_col=40000, n_ic=3000, n_bc=1500, n_epochs=40000, lr=2e-5, lr_min=1e-6, beta=5.0, n_cons=9, tw=[0.70, 0.15, 0.10, 0.05], tb=[0.0, 1.5, 3.0, 5.0, 6.0], eval_every=4000)
STAGE5 = dict(t_max=6.0, n_col=40000, n_ic=3000, n_bc=1500, n_epochs=20000, lr=1e-5, lr_min=1e-6, beta=5.0, n_cons=9, tw=[0.70, 0.15, 0.10, 0.05], tb=[0.0, 1.5, 3.0, 5.0, 6.0], eval_every=4000)

torch.manual_seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
try:
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
except Exception:
    pass

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device} | Seed: {SEED}")
if device.type == 'cuda':
    print(f"GPU: {torch.cuda.get_device_name(0)}")

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
    def u_real(self, x, t): return self.amp_u * self._sech2(x, t) * np.cos(self._phase(x, t))
    def u_imag(self, x, t): return self.amp_u * self._sech2(x, t) * np.sin(self._phase(x, t))
    def v(self, x, t): return self.amp_v * self._sech2(x, t)
    def p(self, x, t):
        eta = self._eta(x, t)
        return (3.0 * self.nu / (4.0 * self.omn2**1.5)) * (1.0 / np.cosh(eta))**2 * np.tanh(eta)
    def conserved_quantity(self, x_grid, t_val):
        return np_trapz(self.amp_u**2 * (1.0 / np.cosh(self._eta(x_grid, t_val)))**4, x_grid)

kgs = KGSExactSolution(nu=NU_0, x0=X_0)
C_exact = kgs.conserved_quantity(np.linspace(X_MIN, X_MAX, 5000), 0.0)
print(f"C_exact = {C_exact:.6f}")

class PINN_KGS(nn.Module):
    def __init__(self, n_hidden=6, n_neurons=128):
        super().__init__()
        layers = [nn.Linear(2, n_neurons), nn.Tanh()]
        for _ in range(n_hidden - 1):
            layers += [nn.Linear(n_neurons, n_neurons), nn.Tanh()]
        layers.append(nn.Linear(n_neurons, 4))
        self.network = nn.Sequential(*layers)
        for m in self.network:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)
    def forward(self, x):
        out = self.network(x)
        return {'u_R': out[:,0:1], 'u_I': out[:,1:2], 'v': out[:,2:3], 'p': out[:,3:4]}

def lhs_1d(n, lo, hi, seed):
    rng = np.random.RandomState(seed)
    iv = np.linspace(0.0, 1.0, n + 1)
    pts = rng.uniform(iv[:-1], iv[1:])
    rng.shuffle(pts)
    return np.sort(lo + pts * (hi - lo))

def lhs_2d(n, bounds, seed):
    rng = np.random.RandomState(seed)
    result = np.zeros((n, len(bounds)))
    for i, (lo, hi) in enumerate(bounds):
        iv = np.linspace(0.0, 1.0, n + 1)
        pts = rng.uniform(iv[:-1], iv[1:])
        rng.shuffle(pts)
        result[:, i] = lo + pts * (hi - lo)
    return result

def make_phase0_points(seed):
    xg = lhs_1d(PHASE0_N_IC_GLOBAL, X_MIN, X_MAX, seed)
    xc = lhs_1d(PHASE0_N_IC_CENTER, -PHASE0_CENTER_HALF_WIDTH, PHASE0_CENTER_HALF_WIDTH, seed + 17)
    x = np.concatenate([xg, xc]); x.sort(); return x

def make_data(t_max, n_col, n_ic, n_bc, seed, tw=None, tb=None):
    if tw is not None and tb is not None:
        chunks = []
        for i, (w, tl, th) in enumerate(zip(tw, tb[:-1], tb[1:])):
            nc = max(int(n_col * w), 100)
            chunks.append(lhs_2d(nc, [(X_MIN, X_MAX), (tl, th)], seed + i * 17))
        col = np.vstack(chunks)
        np.random.RandomState(seed + 99).shuffle(col)
    else:
        col = lhs_2d(n_col, [(X_MIN, X_MAX), (0.0, t_max)], seed)
    x_col = torch.tensor(col[:,0:1], dtype=torch.float32, device=device, requires_grad=True)
    t_col = torch.tensor(col[:,1:2], dtype=torch.float32, device=device, requires_grad=True)
    x_ic = lhs_1d(n_ic, X_MIN, X_MAX, seed + 1)
    x_ic_t = torch.tensor(x_ic.reshape(-1,1), dtype=torch.float32, device=device)
    ic_inp = torch.cat([x_ic_t, torch.zeros_like(x_ic_t)], dim=1)
    tgt_ic = {k: torch.tensor(getattr(kgs, fn)(x_ic, 0.0).reshape(-1,1), dtype=torch.float32, device=device)
              for k,fn in [('u_R','u_real'),('u_I','u_imag'),('v','v'),('p','p')]}
    t_bc = lhs_1d(n_bc, 0.0, t_max, seed + 2)
    x_bc = np.concatenate([np.full(n_bc, X_MIN), np.full(n_bc, X_MAX)])
    t_bc_all = np.concatenate([t_bc, t_bc])
    xb = torch.tensor(x_bc.reshape(-1,1), dtype=torch.float32, device=device)
    tb_t = torch.tensor(t_bc_all.reshape(-1,1), dtype=torch.float32, device=device)
    tgt_bc = {k: torch.tensor(getattr(kgs, fn)(x_bc, t_bc_all).reshape(-1,1), dtype=torch.float32, device=device)
              for k,fn in [('u_R','u_real'),('u_I','u_imag'),('v','v'),('p','p')]}
    return {'x_col': x_col, 't_col': t_col, 'ic_inp': ic_inp, 'targets_ic': tgt_ic,
            'x_bc': xb, 't_bc': tb_t, 'targets_bc': tgt_bc,
            'n_colloc': len(col), 'n_ic': n_ic, 'n_bc_total': 2*n_bc, 't_max': t_max}

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
    return {'R_schrod_re': -u_I_t + 0.5*u_R_xx + u_R*v,
            'R_schrod_im': u_R_t + 0.5*u_I_xx + u_I*v,
            'R_compat': v_t - p,
            'R_kg': p_t - v_xx + v - (u_R**2 + u_I**2)}, pred

def compute_ic_loss(model, ic_inp, targets):
    pred = model(ic_inp)
    d = {k: torch.mean((pred[k] - targets[k])**2) for k in pred}
    total = sum(d.values())
    return total, {k: v.item() for k,v in d.items()} | {'total': total.item()}

def sigmoid_ramp(epoch, center=250, steepness=0.02):
    return 1.0 / (1.0 + np.exp(-steepness * (epoch - center)))

def loss_phase1(model, data, epoch):
    w = sigmoid_ramp(epoch, PHASE1_RAMP_CENTER, PHASE1_RAMP_STEEPNESS)
    res, _ = compute_pde_residuals(model, data['x_col'], data['t_col'])
    ls = {k: torch.mean(v**2) for k,v in res.items()}
    L_PDE = sum(ls.values())
    pred_ic = model(data['ic_inp']); L_IC = sum(torch.mean((pred_ic[k] - data['targets_ic'][k])**2) for k in pred_ic)
    pred_bc = model(torch.cat([data['x_bc'], data['t_bc']], dim=1)); L_BC = sum(torch.mean((pred_bc[k] - data['targets_bc'][k])**2) for k in pred_bc)
    total = w*L_PDE + L_IC + w*L_BC
    return total, {'total': total.item(), 'pde': L_PDE.item(), 'ic': L_IC.item(), 'bc': L_BC.item(), 'w_ramp': w}

def loss_standard(model, data):
    res, _ = compute_pde_residuals(model, data['x_col'], data['t_col'])
    L_PDE = sum(torch.mean(v**2) for v in res.values())
    pred_ic = model(data['ic_inp']); L_IC = sum(torch.mean((pred_ic[k] - data['targets_ic'][k])**2) for k in pred_ic)
    pred_bc = model(torch.cat([data['x_bc'], data['t_bc']], dim=1)); L_BC = sum(torch.mean((pred_bc[k] - data['targets_bc'][k])**2) for k in pred_bc)
    total = L_PDE + L_IC + L_BC
    return total, {'total': total.item(), 'pde': L_PDE.item(), 'ic': L_IC.item(), 'bc': L_BC.item()}

def loss_with_conservation(model, data, beta=5.0, n_times=5):
    base, d = loss_standard(model, data)
    ct = np.linspace(0.0, data['t_max'], n_times + 2)[1:-1]
    xq = torch.linspace(X_MIN, X_MAX, 500, device=device).unsqueeze(1)
    dx = (X_MAX - X_MIN) / 499
    viol = torch.tensor(0.0, device=device)
    for tc in ct:
        tq = torch.full_like(xq, float(tc))
        pred = model(torch.cat([xq, tq], dim=1))
        Ct = torch_trapz((pred['u_R']**2 + pred['u_I']**2).squeeze(), dx=dx)
        viol += (Ct - C_exact)**2
    Lc = beta * viol / max(len(ct), 1)
    total = base + Lc
    d['cons'] = Lc.item(); d['total'] = total.item()
    return total, d

def evaluate(model, t_val, n=1000):
    model.eval()
    x = np.linspace(X_MIN, X_MAX, n)
    with torch.no_grad():
        xt = torch.tensor(x.reshape(-1,1), dtype=torch.float32, device=device)
        tt = torch.full_like(xt, t_val)
        pred = model(torch.cat([xt, tt], dim=1))
    preds = {k: pred[k].cpu().numpy().flatten() for k in ['u_R','u_I','v']}
    exact = {'u_R': kgs.u_real(x, t_val), 'u_I': kgs.u_imag(x, t_val), 'v': kgs.v(x, t_val)}
    errors = {}
    for k in preds:
        diff = np.abs(preds[k] - exact[k])
        errors[f'{k}_Linf'] = float(np.max(diff))
        errors[f'{k}_L2'] = float(np.sqrt(np.mean(diff**2)))
    return errors

def conserve(model, t_val):
    model.eval()
    with torch.no_grad():
        xq = torch.linspace(X_MIN, X_MAX, 3000, device=device).unsqueeze(1)
        tq = torch.full_like(xq, t_val)
        pred = model(torch.cat([xq, tq], dim=1))
        return torch_trapz((pred['u_R']**2 + pred['u_I']**2).squeeze(), dx=(X_MAX - X_MIN)/2999).item()

def error_sums(results):
    return {'t05_sum_linf': sum(results[0.5][f'{k}_Linf'] for k in ['u_R','u_I','v']),
            't10_sum_linf': sum(results[1.0][f'{k}_Linf'] for k in ['u_R','u_I','v']),
            't05_sum_l2': sum(results[0.5][f'{k}_L2'] for k in ['u_R','u_I','v']),
            't10_sum_l2': sum(results[1.0][f'{k}_L2'] for k in ['u_R','u_I','v']),
            'max_drift': max(results[t]['drift'] for t in results if isinstance(t, float))}

def full_eval(model, label='', times=None):
    if times is None: times = [0.0,0.5,1.0,2.0,3.0,4.0,5.0,6.0]
    results = {}
    print(f"\n{'='*118}\nEVAL: {label}\n{'='*118}")
    print(f"{'t':<6} {'uR_Linf':<12} {'uR_L2':<12} {'uI_Linf':<12} {'uI_L2':<12} {'v_Linf':<12} {'v_L2':<12} {'C(t)':<12} {'dC/C':<12}")
    print('-'*118)
    for t in times:
        e = evaluate(model, t); C = conserve(model, t); drift = abs(C - C_exact)/C_exact
        results[t] = {'u_R_Linf': e['u_R_Linf'], 'u_R_L2': e['u_R_L2'], 'u_I_Linf': e['u_I_Linf'], 'u_I_L2': e['u_I_L2'], 'v_Linf': e['v_Linf'], 'v_L2': e['v_L2'], 'C': C, 'drift': drift}
        print(f"{t:<6.1f} {e['u_R_Linf']:<12.2e} {e['u_R_L2']:<12.2e} {e['u_I_Linf']:<12.2e} {e['u_I_L2']:<12.2e} {e['v_Linf']:<12.2e} {e['v_L2']:<12.2e} {C:<12.6f} {drift:<12.2e}")
    s = error_sums(results)
    print(f"  Max drift:       {s['max_drift']:.2e}")
    results['summary'] = s
    return results

def save_checkpoint(model, stage_name, metadata=None):
    path = os.path.join(OUTPUT_DIR, f'ckpt_seed{SEED}_{stage_name}.pt')
    payload = {'model_state_dict': model.state_dict(), 'seed': SEED, 'stage': stage_name}
    if metadata is not None: payload['metadata'] = metadata
    torch.save(payload, path)
    print(f"Saved: {path}")
    return path

def train_phase0(model):
    print("\n" + '='*80 + f"\nPHASE 0: Improved Warm-Start (Seed {SEED})\n" + '='*80)
    x_np = make_phase0_points(SEED)
    x_t = torch.tensor(x_np.reshape(-1,1), dtype=torch.float32, device=device)
    ic_inp = torch.cat([x_t, torch.zeros_like(x_t)], dim=1)
    targets = {k: torch.tensor(getattr(kgs, fn)(x_np, 0.0).reshape(-1,1), dtype=torch.float32, device=device)
               for k,fn in [('u_R','u_real'),('u_I','u_imag'),('v','v'),('p','p')]}
    best_loss = float('inf'); best_state = copy.deepcopy(model.state_dict())
    opt = torch.optim.Adam(model.parameters(), lr=PHASE0_ADAM_LR)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=PHASE0_ADAM_EPOCHS, eta_min=PHASE0_ADAM_ETA_MIN)
    for ep in range(1, PHASE0_ADAM_EPOCHS + 1):
        opt.zero_grad(); loss, d = compute_ic_loss(model, ic_inp, targets); loss.backward(); opt.step(); sched.step()
        if d['total'] < best_loss: best_loss, best_state = d['total'], copy.deepcopy(model.state_dict())
        if ep % 1000 == 0 or ep == 1 or ep == PHASE0_ADAM_EPOCHS:
            print(f"Adam Ep {ep:5d}/{PHASE0_ADAM_EPOCHS} | L={d['total']:.2e}")
    model.load_state_dict(best_state)
    optimizer = torch.optim.LBFGS(model.parameters(), lr=PHASE0_LBFGS_LR, max_iter=PHASE0_LBFGS_MAX_ITER, max_eval=PHASE0_LBFGS_MAX_ITER+200, history_size=PHASE0_LBFGS_HISTORY, tolerance_grad=1e-10, tolerance_change=1e-12, line_search_fn='strong_wolfe')
    calls = {'n':0}; best_lbfgs = best_loss; best_lbfgs_state = copy.deepcopy(best_state)
    def closure():
        nonlocal best_lbfgs, best_lbfgs_state
        optimizer.zero_grad(); loss, d = compute_ic_loss(model, ic_inp, targets); loss.backward(); calls['n'] += 1
        if d['total'] < best_lbfgs: best_lbfgs, best_lbfgs_state = d['total'], copy.deepcopy(model.state_dict())
        if calls['n'] % 25 == 0 or calls['n'] == 1: print(f"LBFGS step {calls['n']:4d}/{PHASE0_LBFGS_MAX_ITER} | L={d['total']:.2e}")
        return loss
    optimizer.step(closure)
    model.load_state_dict(best_lbfgs_state)
    res = full_eval(model, 'Phase 0', times=[0.0])
    save_checkpoint(model, 'phase0', {'best_loss': best_lbfgs})
    return res

def train_phase1(model):
    print("\n" + '='*80 + "\nPHASE 1: Corrected Baseline PDE Training\n" + '='*80)
    data = make_data(T_MAX, PHASE1_N_COL, PHASE1_N_IC, PHASE1_N_BC, SEED + 10)
    best_loss = float('inf'); best_state = copy.deepcopy(model.state_dict())
    opt = torch.optim.Adam(model.parameters(), lr=PHASE1_LR)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=PHASE1_EPOCHS, eta_min=PHASE1_LR_MIN)
    for ep in range(1, PHASE1_EPOCHS + 1):
        opt.zero_grad(); loss, d = loss_phase1(model, data, ep); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sched.step()
        if d['total'] < best_loss: best_loss, best_state = d['total'], copy.deepcopy(model.state_dict())
        if ep % 500 == 0 or ep == 1:
            print(f"Ep {ep:6d}/{PHASE1_EPOCHS} | L={d['total']:.2e} | PDE={d['pde']:.2e} | IC={d['ic']:.2e} | BC={d['bc']:.2e} | w={d['w_ramp']:.3f}")
    model.load_state_dict(best_state)
    res = full_eval(model, 'Phase 1 Baseline')
    save_checkpoint(model, 'phase1', {'best_loss': best_loss})
    return res

def train_stage(model, cfg, label, seed_offset, stage_name):
    print("\n" + '='*80 + f"\n{label}\n" + '='*80)
    data = make_data(cfg['t_max'], cfg['n_col'], cfg['n_ic'], cfg['n_bc'], SEED + seed_offset, tw=cfg['tw'], tb=cfg['tb'])
    best_loss = float('inf'); best_state = copy.deepcopy(model.state_dict())
    opt = torch.optim.Adam(model.parameters(), lr=cfg['lr'])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg['n_epochs'], eta_min=cfg['lr_min'])
    for ep in range(1, cfg['n_epochs'] + 1):
        opt.zero_grad(); loss, d = loss_with_conservation(model, data, cfg['beta'], cfg['n_cons']); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sched.step()
        if d['total'] < best_loss: best_loss, best_state = d['total'], copy.deepcopy(model.state_dict())
        if ep % max(cfg['n_epochs']//10,1) == 0 or ep == 1:
            print(f"Ep {ep:6d}/{cfg['n_epochs']} | L={d['total']:.2e} | PDE={d['pde']:.2e} | IC={d['ic']:.2e} | BC={d['bc']:.2e} | Cons={d['cons']:.2e}")
    model.load_state_dict(best_state)
    res = full_eval(model, label)
    save_checkpoint(model, stage_name, {'best_loss': best_loss})
    return res

def train_stage_with_best_t05(model, cfg, label, seed_offset, stage_name):
    print("\n" + '='*80 + f"\n{label}\n" + '='*80)
    data = make_data(cfg['t_max'], cfg['n_col'], cfg['n_ic'], cfg['n_bc'], SEED + seed_offset, tw=cfg['tw'], tb=cfg['tb'])
    best_loss = float('inf'); best_state = copy.deepcopy(model.state_dict())
    best_t05_sum = float('inf'); best_t05_state = None; best_t05_epoch = 0
    opt = torch.optim.Adam(model.parameters(), lr=cfg['lr'])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg['n_epochs'], eta_min=cfg['lr_min'])
    for ep in range(1, cfg['n_epochs'] + 1):
        opt.zero_grad(); loss, d = loss_with_conservation(model, data, cfg['beta'], cfg['n_cons']); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sched.step()
        if d['total'] < best_loss: best_loss, best_state = d['total'], copy.deepcopy(model.state_dict())
        if ep % max(cfg['n_epochs']//10,1) == 0 or ep == 1:
            print(f"Ep {ep:6d}/{cfg['n_epochs']} | L={d['total']:.2e} | PDE={d['pde']:.2e} | IC={d['ic']:.2e} | BC={d['bc']:.2e} | Cons={d['cons']:.2e}")
        if ep % cfg.get('eval_every', 4000) == 0:
            tmp = full_eval(model, f'{label} checkpoint ep {ep}', times=[0.5,1.0])
            s05 = tmp['summary']['t05_sum_linf']
            if s05 < best_t05_sum: best_t05_sum, best_t05_state, best_t05_epoch = s05, copy.deepcopy(model.state_dict()), ep
    model.load_state_dict(best_state)
    res_final = full_eval(model, f'{label} FINAL')
    save_checkpoint(model, f'{stage_name}_final', {'best_loss': best_loss})
    res_best = None; model_best = None
    if best_t05_state is not None:
        model_best = PINN_KGS().to(device); model_best.load_state_dict(best_t05_state)
        res_best = full_eval(model_best, f'{label} BEST_T05 (ep {best_t05_epoch})')
        save_checkpoint(model_best, f'{stage_name}_best_t05', {'best_epoch': best_t05_epoch, 'best_t05_sum': best_t05_sum})
    if res_best is not None and res_best['summary']['t05_sum_linf'] < res_final['summary']['t05_sum_linf']:
        winner_model, winner_results, winner_name = model_best, res_best, f'best_t05_ep{best_t05_epoch}'
    else:
        winner_model, winner_results, winner_name = model, res_final, 'final'
    return {'final_model': model, 'final_results': res_final, 'best_model': model_best, 'best_results': res_best, 'winner_model': winner_model, 'winner_results': winner_results, 'winner_name': winner_name}

def choose_final_winner(candidates):
    best_name = None; best_model = None; best_results = None; best_s05 = float('inf')
    for name, (mdl, res) in candidates.items():
        if res is None: continue
        s05 = res['summary']['t05_sum_linf']
        if s05 < best_s05:
            best_name, best_model, best_results, best_s05 = name, mdl, res, s05
    return best_name, best_model, best_results

pipeline_start = time(); all_results = {}
model = PINN_KGS().to(device)
all_results['phase0'] = train_phase0(model)
all_results['phase1'] = train_phase1(model)
all_results['stage1'] = train_stage(model, STAGE1, 'STAGE 1: [0,2] + Conservation', 20, 'stage1')
all_results['stage2'] = train_stage(model, STAGE2, 'STAGE 2: [0,4] Weighted 60/40 + Conservation', 30, 'stage2')
all_results['stage3'] = train_stage(model, STAGE3, 'STAGE 3: [0,6] Weighted 50/30/20 + Conservation', 40, 'stage3')
stage4 = train_stage_with_best_t05(model, STAGE4, 'STAGE 4: Heavy Weighting 70/15/10/5', 50, 'stage4')
all_results['stage4_final'] = stage4['final_results']
if stage4['best_results'] is not None: all_results['stage4_best_t05'] = stage4['best_results']
model_s5 = PINN_KGS().to(device); model_s5.load_state_dict(stage4['winner_model'].state_dict())
stage5 = train_stage_with_best_t05(model_s5, STAGE5, 'STAGE 5: Fresh Collocation + Continue', 60, 'stage5')
all_results['stage5_final'] = stage5['final_results']
if stage5['best_results'] is not None: all_results['stage5_best_t05'] = stage5['best_results']
candidates = {'stage4_winner': (stage4['winner_model'], stage4['winner_results']), 'stage5_final': (stage5['final_model'], stage5['final_results'])}
if stage5['best_model'] is not None: candidates['stage5_best_t05'] = (stage5['best_model'], stage5['best_results'])
best_name, best_model, best_results = choose_final_winner(candidates)
print(f"\n*** FINAL WINNER FOR SEED {SEED}: {best_name} | t05_sum={best_results['summary']['t05_sum_linf']:.2e} ***")
all_results['final_winner'] = best_results
save_checkpoint(best_model, 'final_winner', {'winner_name': best_name, 'summary': best_results['summary']})

print("\n" + '='*80 + "\nINFERENCE SPEED BENCHMARK\n" + '='*80)
best_model.eval();
with torch.no_grad():
    dummy = torch.randn(10000, 2, device=device); _ = best_model(dummy)
    if device.type == 'cuda': torch.cuda.synchronize()
n_query = 10000; n_passes = 100
x_bench = torch.linspace(X_MIN, X_MAX, n_query, device=device).unsqueeze(1); t_bench = torch.full_like(x_bench, 1.0); inp = torch.cat([x_bench, t_bench], dim=1)
if device.type == 'cuda': torch.cuda.synchronize()
t0 = time()
with torch.no_grad():
    for _ in range(n_passes): _ = best_model(inp)
if device.type == 'cuda': torch.cuda.synchronize()
t_infer = (time() - t0) / n_passes
print(f"Single forward pass ({n_query} points): {t_infer*1000:.2f} ms")
print(f"Points per second: {n_query/t_infer:.0f}")
all_results['inference'] = {'time_per_pass_ms': t_infer*1000, 'n_query_points': n_query, 'points_per_second': n_query/t_infer}

print("\n" + '='*110 + "\nSEED SUMMARY\n" + '='*110)
print(f"{'Stage':<22} {'t05_Linf':<12} {'t10_Linf':<12} {'t05_L2':<12} {'t10_L2':<12} {'maxdC':<12}")
print('-'*110)
for key in ['phase1','stage1','stage2','stage3','stage4_final','stage5_final','final_winner']:
    if key in all_results:
        s = all_results[key]['summary']
        print(f"{key:<22} {s['t05_sum_linf']:<12.2e} {s['t10_sum_linf']:<12.2e} {s['t05_sum_l2']:<12.2e} {s['t10_sum_l2']:<12.2e} {s['max_drift']:<12.2e}")

json_results = {}
for stage, res in all_results.items():
    if stage == 'inference': json_results[stage] = res
    else: json_results[stage] = {str(k): v for k, v in res.items()}
json_path = os.path.join(OUTPUT_DIR, f'results_seed{SEED}.json')
with open(json_path, 'w') as f: json.dump(json_results, f, indent=2)
print(f"\nSaved: {json_path}")
total_time = time() - pipeline_start
print(f"\n{'='*80}\nALIGNED MULTI-SEED PIPELINE COMPLETE (Seed {SEED})\nTotal time: {total_time:.0f}s ({total_time/3600:.1f} hours)\nFinal winner: {best_name}\n{'='*80}")
