"""
================================================================================
COHORT C — CIFAR-10 + CNN generalisation test of the Latent Brittleness theory
Completes Zenodo 10.5281/zenodo.22120112, section 11 weakness 1.
================================================================================

Single self-contained script. Paste and run:

    pip install torch torchvision numpy scipy pillow
    python cohortC_all.py

Downloads CIFAR-10, trains the cohort, and prints the preregistered verdict.
Resumable: interrupt any time and re-run the same command; completed work is skipped.
CUDA auto-detected. ~30-60 min on one GPU; hours on CPU.

--------------------------------------------------------------------------------
PREREGISTERED QUESTIONS  (fixed before any data collected; do not restate after)

  Q1  Do models sharing architecture, recipe and baseline still differ in degradation?
      pass: within-cell share >= 15% of (architecture + model) variance, perm p < 0.01
  Q2  Does within-family stressor coherence exceed between-family?
      pass: gap >= 0.15, and survives residualisation on baseline accuracy
  Q3  Do fractional and fixed-absolute capacity damage load with OPPOSITE sign on width?
      pass: signs oppose and both |rho| > 0.3
  Q4  Do the first two functional-PCA modes explain >= 90% (magnitude + early/late tilt)?
      pass: M+T >= 90% and mode 1 is a monotone ramp
  Q5  Is curvature the most coordinate-sensitive of four curve features?
      pass: curvature has lowest rank-stability under eps -> eps^2

DECISION RULE
  5/5 or 4/5 -> theory generalises beyond MNIST/MLPs
  2-3/5      -> partial generalisation; failing items become boundary conditions
  0-1/5      -> the MNIST result is regime-specific
Any outcome is publishable. A failure is as informative as a pass. Report either way.
--------------------------------------------------------------------------------
"""

import io, json, itertools, random, sys, time, urllib.request, zipfile
from pathlib import Path

import numpy as np
import torch, torch.nn as nn, torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from scipy import stats

HERE = Path(__file__).parent if '__file__' in dir() else Path.cwd()
OUT = HERE / 'results'; OUT.mkdir(exist_ok=True)
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'

# ============================== CONFIG ========================================
SEEDS_PER_CELL = 6
MAXEP          = 30          # CIFAR seeds need time to reach a common plateau
P0_LO, P0_HI   = 0.44, 0.52  # baseline band -- widen if yield is poor (see note)
P0_TGT         = 0.48
N_TRAIN, N_TEST, EVAL_N = 8000, 2000, 1500
EPS  = np.array([k / 9 for k in range(10)])
REPS = 2                     # repeats for stochastic (model-side) damage

# NOTE ON THE BAND. CIFAR CNNs vary far more across seeds than MNIST MLPs, so a tight
# band rejects most models. If fewer than 24 survive, WIDEN P0_LO/P0_HI or raise
# SEEDS_PER_CELL and re-run (it resumes). Tune the band on YIELD ONLY -- never after
# looking at degradation curves. The analysis residualises baseline accuracy for
# exactly this reason.

CELLS = [(c, b, d) for c in (16, 32, 64) for b in (2, 3) for d in (0.1, 0.3)]

FAM = {'noise': 'input', 'blur': 'input', 'occlusion': 'input', 'contrast': 'input',
       'rotate': 'shift', 'translate': 'shift', 'scale': 'shift',
       'weight_drop': 'capacity', 'chan_ablate': 'capacity', 'mag_prune': 'capacity',
       'act_clip': 'capacity', 'abs_weight': 'capacity', 'abs_chan': 'capacity'}

# ============================== DATA ==========================================
def get_data():
    cache = HERE / 'cifar.pt'
    if cache.exists():
        print(f'Using cached {cache.name}')
        return torch.load(cache)
    src = HERE / 'CIFAR-10-images-master'
    if not src.exists():
        print('Downloading CIFAR-10 (~56 MB)...', flush=True)
        t = time.time()
        url = 'https://codeload.github.com/YoongiKim/CIFAR-10-images/zip/refs/heads/master'
        data = urllib.request.urlopen(url, timeout=600).read()
        print(f'  {len(data)/1e6:.0f} MB in {time.time()-t:.0f}s, extracting...', flush=True)
        zipfile.ZipFile(io.BytesIO(data)).extractall(HERE)
    from PIL import Image
    tr = sorted((src / 'train').glob('*/*.jpg'))
    te = sorted((src / 'test').glob('*/*.jpg'))
    classes = sorted({p.parent.name for p in tr})
    random.seed(0); random.shuffle(tr); random.shuffle(te)

    def load(paths):
        X = np.stack([np.array(Image.open(p)) for p in paths]).astype(np.float32) / 255.
        y = np.array([classes.index(p.parent.name) for p in paths])
        return torch.tensor(X).permute(0, 3, 1, 2), torch.tensor(y)

    print(f'Encoding {N_TRAIN} train / {N_TEST} test images...', flush=True)
    out = (*load(tr[:N_TRAIN]), *load(te[:N_TEST]))
    torch.save(out, cache)
    print(f'Cached -> {cache}')
    return out

# ============================== MODEL =========================================
class CNN(nn.Module):
    def __init__(s, ch, blocks, dr):
        super().__init__()
        s.convs = nn.ModuleList(); s.bns = nn.ModuleList()
        c_in = 3
        for b in range(blocks):
            c_out = ch * (2 ** b)
            s.convs.append(nn.Conv2d(c_in, c_out, 3, padding=1))
            s.bns.append(nn.BatchNorm2d(c_out)); c_in = c_out
        s.drop = nn.Dropout(dr); s.head = nn.Linear(c_in, 10)
        s.chan_mask = None; s.act_clip = None

    def forward(s, x):
        for i, (cv, bn) in enumerate(zip(s.convs, s.bns)):
            x = F.relu(bn(cv(x)))
            if s.act_clip is not None: x = torch.clamp(x, max=s.act_clip)
            if s.chan_mask is not None: x = x * s.chan_mask[i].view(1, -1, 1, 1)
            x = F.max_pool2d(x, 2)
        return s.head(s.drop(F.adaptive_avg_pool2d(x, 1).flatten(1)))

# ============================== STRESSORS =====================================
# Perturbation RNG is seeded INDEPENDENTLY of the training seed, so every model sees
# identical corrupted inputs. (This fixes an entanglement bug in the MNIST cohorts.)
def _pg(tag):
    return torch.Generator(device='cpu').manual_seed(abs(hash(tag)) % (2 ** 31))

def s_noise(X, e):
    if e == 0: return X
    n = torch.randn(X.shape, generator=_pg('noise')).to(X.device)
    return torch.clamp(X + n * (e * 0.35), 0, 1)

def s_blur(X, e):
    n = int(round(e * 8))
    if n < 1: return X
    k = (torch.ones(3, 1, 3, 3) / 9.).to(X.device); b = X.clone()
    for _ in range(n): b = F.conv2d(b, k, padding=1, groups=3)
    return torch.clamp(b, 0, 1)

def s_occlusion(X, e):
    sd = int(round(e * 18))
    if sd < 1: return X
    Z = X.clone(); g = _pg('occl')
    for i in range(len(Z)):
        r = torch.randint(0, max(1, 32 - sd), (2,), generator=g)
        Z[i, :, r[0]:r[0] + sd, r[1]:r[1] + sd] = 0
    return Z

def s_contrast(X, e):
    m = X.mean(dim=(1, 2, 3), keepdim=True)
    return torch.clamp((X - m) * (1 - 0.9 * e) + m, 0, 1)

def _aff(X, angle=0., tx=0., ty=0., sc=1.):
    th = torch.tensor(angle * np.pi / 180.)
    co, si = torch.cos(th) / sc, torch.sin(th) / sc
    M = torch.tensor([[co, -si, tx], [si, co, ty]], dtype=torch.float32)
    M = M.unsqueeze(0).repeat(len(X), 1, 1).to(X.device)
    grid = F.affine_grid(M, X.shape, align_corners=False)
    return F.grid_sample(X, grid, align_corners=False, padding_mode='zeros')

def s_rotate(X, e):    return X if e == 0 else _aff(X, angle=e * 40.)
def s_translate(X, e): return X if e == 0 else _aff(X, tx=e * 0.25, ty=e * 0.15)
def s_scale(X, e):     return X if e == 0 else _aff(X, sc=1 - 0.35 * e)

INPUT_S = dict(noise=s_noise, blur=s_blur, occlusion=s_occlusion, contrast=s_contrast,
               rotate=s_rotate, translate=s_translate, scale=s_scale)

class Perturb:
    """apply damage in place, restore on exit -- avoids deepcopy"""
    def __init__(s, m): s.m = m
    def __enter__(s):
        s.sv = [p.data.clone() for p in s.m.parameters()]; return s.m
    def __exit__(s, *a):
        for p, q in zip(s.m.parameters(), s.sv): p.data.copy_(q)
        s.m.chan_mask = None; s.m.act_clip = None

def _g(k): return torch.Generator().manual_seed(k)

# Q3 hinges on these four: fractional vs fixed-absolute, for weights and for channels.
def d_frac_weight(m, e, sd):
    for n_, p in m.named_parameters():
        if 'weight' in n_ and p.dim() > 1:
            mask = (torch.rand(p.shape, generator=_g(sd)) > e * 0.8).float()
            p.data *= mask.to(p.device)

def d_abs_weight(m, e, sd, keep=1500):
    """Absolute phi: fixed PER-LAYER retained-weight cap, NOT a global budget."""
    k0 = int(round(keep * (1 - e) + 100 * e))
    for n_, p in m.named_parameters():
        if 'weight' in n_ and p.dim() > 1:
            k = min(k0, p.numel())
            thr = p.data.abs().flatten().kthvalue(p.numel() - k + 1).values
            p.data *= (p.data.abs() >= thr).float()

def d_frac_chan(m, e, sd):
    m.chan_mask = [(torch.rand(c.out_channels, generator=_g(sd)) > e * 0.85).float().to(DEV)
                   for c in m.convs]

def d_abs_chan(m, e, sd, keep=12):
    """Absolute phi: fixed retained-channel cap PER CONV LAYER."""
    k0 = int(round(keep * (1 - e) + 1 * e)); ms = []
    for c in m.convs:
        w = c.out_channels; k = min(k0, w)
        mk = torch.zeros(w); mk[torch.randperm(w, generator=_g(sd))[:k]] = 1.
        ms.append(mk.to(DEV))
    m.chan_mask = ms

def d_mag_prune(m, e, sd):
    for n_, p in m.named_parameters():
        if 'weight' in n_ and p.dim() > 1:
            k = int(p.numel() * e * 0.95)
            if k > 0:
                thr = p.data.abs().flatten().kthvalue(k).values
                p.data *= (p.data.abs() > thr).float()

def d_act_clip(m, e, sd): m.act_clip = float(4.0 * (0.02 ** e))

MODEL_S = dict(weight_drop=d_frac_weight, chan_ablate=d_frac_chan, mag_prune=d_mag_prune,
               act_clip=d_act_clip, abs_weight=d_abs_weight, abs_chan=d_abs_chan)
STRESSORS = list(INPUT_S) + list(MODEL_S)

# ============================== RUN THE COHORT ================================
def run_cohort(EX, EY, TL):
    @torch.no_grad()
    def acc(m, X=None):
        m.eval(); X = EX if X is None else X; c = 0
        for i in range(0, len(X), 500):
            c += (m(X[i:i + 500]).argmax(1) == EY[i:i + 500]).sum().item()
        return c / len(X)

    def curve(m, s):
        if s in INPUT_S:
            f = INPUT_S[s]
            a = [acc(m, f(EX, float(e))) for e in EPS]
        else:
            f = MODEL_S[s]; a = []
            for e in EPS:
                if e == 0: a.append(acc(m)); continue
                vs = []
                for r in range(REPS):
                    with Perturb(m) as mm:
                        f(mm, float(e), r); vs.append(acc(mm))
                a.append(float(np.mean(vs)))
        a = np.array(a); return (a[0] - a).tolist()

    def train_matched(ch, bl, dr, seed):
        torch.manual_seed(seed)
        m = CNN(ch, bl, dr).to(DEV)
        opt = optim.Adam(m.parameters(), lr=2e-3)
        best = (9, None, 0)
        for ep in range(1, MAXEP + 1):
            m.train()
            for X, y in TL:
                X, y = X.to(DEV), y.to(DEV)
                opt.zero_grad(); nn.CrossEntropyLoss()(m(X), y).backward(); opt.step()
            p = acc(m)
            if abs(p - P0_TGT) < best[0]:
                best = (abs(p - P0_TGT), [q.data.clone() for q in m.parameters()], ep)
            if p > P0_HI + 0.04: break
        for q, s in zip(m.parameters(), best[1]): q.data.copy_(s)
        return m, acc(m), best[2]

    path = OUT / 'cohortC.json'
    recs = json.load(open(path)) if path.exists() else []
    done = {(r['ch'], r['bl'], r['dr'], r['seed']) for r in recs}
    t0 = time.time()
    print(f'\nDevice: {DEV} | {len(CELLS)} cells x {SEEDS_PER_CELL} seeds | '
          f'{len(STRESSORS)} stressors')
    if recs: print(f'Resuming: {len(recs)} models already done')
    for ch, bl, dr in CELLS:
        for sd in range(SEEDS_PER_CELL):
            if (ch, bl, dr, sd) in done: continue
            m, p0, ep = train_matched(ch, bl, dr, 7000 + 1000 * sd + ch + bl)
            ok = P0_LO <= p0 <= P0_HI
            print(f'  ch{ch:>3} b{bl} dr{dr} s{sd}  P0={p0:.4f} ep{ep:>2} '
                  f'{"OK" if ok else "--"}   [{time.time()-t0:.0f}s]', flush=True)
            if not ok: continue
            recs.append(dict(ch=ch, bl=bl, dr=dr, seed=sd, P0=float(p0), epochs=ep,
                             curves={s: curve(m, s) for s in STRESSORS}))
            json.dump(recs, open(path, 'w'))
    cells = {(r['ch'], r['bl'], r['dr']) for r in recs}
    print(f'\nCohort C: {len(recs)} survivors, {len(cells)} cells, '
          f'{len(recs)/max(len(cells),1):.1f} seeds/cell -> {path}')
    if len(recs) < 24:
        print('  WARNING: fewer than 24 survivors. Q1 needs >=4 seeds per cell across')
        print('  >=3 cells. Widen P0_LO/P0_HI or raise SEEDS_PER_CELL, then re-run.')
    return recs

# ============================== ANALYSE =======================================
def analyse(R):
    n = len(R)
    if n < 6:
        print('\nToo few models to analyse. Re-run with a wider band.'); return
    SS = list(R[0]['curves'])
    CH = np.array([r['ch'] for r in R]); P0 = np.array([r['P0'] for r in R])
    cell = np.array([f"{r['ch']}-{r['bl']}-{r['dr']}" for r in R]); cells = sorted(set(cell))
    CU = np.array([[r['curves'][s] for s in SS] for r in R])
    auc = lambda s: np.array([np.trapezoid(r['curves'][s], EPS) for r in R])
    Z = np.column_stack([(auc(s) - auc(s).mean()) / (auc(s).std() + 1e-12) for s in SS])

    print('\n' + '=' * 74)
    print(f'COHORT C — CIFAR-10 + CNN.  n={n}, {len(cells)} cells, '
          f'{n/max(len(cells),1):.1f} seeds/cell')
    print(f'baseline {P0.min():.4f}-{P0.max():.4f} (sd {P0.std():.4f})')
    print('=' * 74)
    if n < 12:
        print('\n*** n < 12: PILOT ONLY. Do not report as a validation cohort. ***')
    v = {}

    def resid(y, X):
        A = np.column_stack([X, np.ones(len(X))])
        return y - A @ np.linalg.lstsq(A, y, rcond=None)[0]

    # ---- Q1 ----
    print('\nQ1  SEED-LEVEL HETEROGENEITY within architecture cell')
    g = Z.mean(1); bet = np.array([g[cell == c].mean() for c in cell]); wit = g - bet
    share = wit.var() / (bet.var() + wit.var() + 1e-12)
    rng = np.random.default_rng(0); NP = 5000; obs = wit.var(); exc = 0
    for _ in range(NP):
        Zp = Z.copy()
        for c in cells:
            idx = np.where(cell == c)[0]
            for j in range(Z.shape[1]): Zp[idx, j] = Z[rng.permutation(idx), j]
        gp = Zp.mean(1); bp = np.array([gp[cell == x].mean() for x in cell])
        if (gp - bp).var() >= obs: exc += 1
    p = (exc + 1) / (NP + 1)
    print(f'    within-architecture share {100*share:.1f}%   permutation p = {p:.2e}')
    print(f'    (MNIST/MLP: input family strongly significant, z = +10.4)')
    v['Q1'] = share >= 0.15 and p < 0.01
    sizes = [int((cell == c).sum()) for c in cells]
    if min(sizes) < 3:
        print(f'    *** smallest cell has {min(sizes)} models — Q1 unreliable at this n')

    # ---- Q2 ----
    print('\nQ2  WITHIN-FAMILY COHERENCE exceeds between-family')
    def gap(mat):
        C = {(a, b): stats.spearmanr(mat[:, a], mat[:, b]).statistic
             for a, b in itertools.combinations(range(len(SS)), 2)}
        wi = [x for (a, b), x in C.items() if FAM[SS[a]] == FAM[SS[b]]]
        bw = [x for (a, b), x in C.items() if FAM[SS[a]] != FAM[SS[b]]]
        return np.mean(wi), np.mean(bw)
    w1, b1 = gap(Z)
    Zr = np.column_stack([resid(Z[:, j], P0[:, None]) for j in range(Z.shape[1])])
    w2, b2 = gap(Zr)
    print(f'    raw             within {w1:+.3f}  between {b1:+.3f}  gap {w1-b1:+.3f}')
    print(f'    P0-residualised within {w2:+.3f}  between {b2:+.3f}  gap {w2-b2:+.3f}')
    print(f'    (MNIST/MLP: gap +0.464 raw, +0.484 residualised)')
    v['Q2'] = (w1 - b1) >= 0.15 and (w2 - b2) >= 0.15

    # ---- Q3 ----
    print('\nQ3  PARAMETERISATION REVERSAL — fractional vs fixed-absolute damage')
    ok3 = []
    for f_, a_, lab in [('weight_drop', 'abs_weight', 'weights'),
                        ('chan_ablate', 'abs_chan', 'channels')]:
        rf = stats.spearmanr(CH, auc(f_)).statistic
        ra = stats.spearmanr(CH, auc(a_)).statistic
        good = rf * ra < 0 and abs(rf) > 0.3 and abs(ra) > 0.3
        ok3.append(good)
        print(f'    {lab:<10} fractional {rf:+.3f}   absolute {ra:+.3f}   '
              f'{"REVERSES" if good else "no clean reversal"}')
    print(f'    (MNIST/MLP: weights -0.81 vs +0.70; units -0.68 vs +0.37)')
    v['Q3'] = any(ok3)

    # ---- Q4 ----
    print('\nQ4  CURVE DIMENSIONALITY — magnitude + early/late tilt')
    flat = CU.reshape(-1, 10); flat = flat - flat.mean(0)
    _, sv, Vt = np.linalg.svd(flat, full_matrices=False); ev = sv ** 2 / (sv ** 2).sum()
    m1 = Vt[0] * np.sign(Vt[0].sum()); mono = bool((np.diff(m1) >= -0.05).all())
    print(f'    M {100*ev[0]:.1f}%   T {100*ev[1]:.1f}%   M+T {100*(ev[0]+ev[1]):.1f}%   '
          f'PC3 {100*ev[2]:.1f}%   mode1 monotone: {mono}')
    print(f'    (MNIST/MLP: 82.2 / 14.5 / 96.8 / 1.7)')
    print('    NOTE: two-dimensionality is near-tautological for monotone stressors;')
    print('    the informative quantity is T relative to a monotone null.')
    v['Q4'] = (ev[0] + ev[1]) >= 0.90 and mono

    # ---- Q5 ----
    print('\nQ5  CURVATURE most coordinate-sensitive of four features')
    IDX = [0, 3, 6, 9]
    def feats(D, e):
        h1, h2 = e[1] - e[0], e[2] - e[1]
        return dict(susceptibility=(D[1] - D[0]) / h1,
                    curvature=2 * (D[2] / h2 - D[1] * (1 / h1 + 1 / h2) + D[0] / h1) / (h1 + h2),
                    area=np.trapezoid(D, e), late_slope=(D[-1] - D[-2]) / (e[-1] - e[-2]))
    stab = {}
    for d in ['susceptibility', 'curvature', 'area', 'late_slope']:
        cs = []
        for s in SS:
            a = [feats(np.array(r['curves'][s])[IDX], EPS[IDX])[d] for r in R]
            b = [feats(np.array(r['curves'][s])[IDX], EPS[IDX] ** 2)[d] for r in R]
            c = stats.spearmanr(a, b).statistic
            if np.isfinite(c): cs.append(c)
        stab[d] = float(np.mean(cs)); print(f'    {d:<16}{stab[d]:>8.3f}')
    print(f'    (MNIST/MLP: 1.000 / 0.792 / 0.994 / 1.000 — curvature lowest)')
    v['Q5'] = min(stab, key=stab.get) == 'curvature'

    # ---- verdict ----
    print('\n' + '=' * 74)
    for k in ['Q1', 'Q2', 'Q3', 'Q4', 'Q5']:
        print(f'  {k}  {"PASS" if v[k] else "FAIL"}')
    npass = sum(v.values())
    print(f'\n  {npass}/5 passed')
    if npass >= 4:
        print('  -> The theory GENERALISES beyond MNIST/MLPs.')
    elif npass >= 2:
        print('  -> PARTIAL generalisation. Failing items become boundary conditions.')
    else:
        print('  -> The Cohort B result appears REGIME-SPECIFIC.')
    print('=' * 74)
    json.dump({k: bool(x) for k, x in v.items()},
              open(OUT / 'cohortC_verdict.json', 'w'), indent=1)
    print(f'\nWritten: {OUT/"cohortC.json"}  and  {OUT/"cohortC_verdict.json"}')
    print('Both belong in the next Zenodo version alongside PREREG_C.md.')

# ============================== MAIN ==========================================
if __name__ == '__main__':
    Xtr, ytr, Xte, yte = get_data()
    EX, EY = Xte[:EVAL_N].to(DEV), yte[:EVAL_N].to(DEV)
    TL = DataLoader(TensorDataset(Xtr, ytr), batch_size=256, shuffle=True)
    recs = run_cohort(EX, EY, TL)
    analyse(recs)
