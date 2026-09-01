"""
Predictor screen: do baseline internals predict (M,T) on HELD-OUT architecture cells?

Protocol (fixed before looking at results):
  - targets M,T = functional-PCA mode scores of degradation curves, per family
  - leave-3-cells-out grouped CV (4 folds over 12 architecture cells)
  - predictors residualised on P0 within training folds only
  - report R2_cv, rho_heldout, MAE
  - ridge with alpha chosen inside the training folds only
No single-predictor cherry-picking: every predictor is reported, and the multivariate
model is scored only out-of-fold.
"""
import json,numpy as np
from scipy import stats
from itertools import combinations

P=json.load(open('/home/claude/lb3/results/predictors.json'))
C=json.load(open('/home/claude/lb3/results/cohortB.json'))
FR=json.load(open('/home/claude/lb3/results/FROZEN_A.json'))
SA=FR['stressors']; EPS=np.array([k/9 for k in range(10)])
FAM={'noise':'input','blur':'input','occlusion':'input','contrast':'input',
     'rotate':'shift','translate':'shift','scale':'shift',
     'weight_drop':'capacity','neuron_ablate':'capacity','mag_prune':'capacity','act_clip':'capacity'}
INP=[s for s in SA if FAM[s]=='input']; CAP=[s for s in SA if FAM[s]=='capacity']

# ---- align the two files by (w,d,dr,seed) ----
key=lambda r:f"{r['w']}-{r['d']}-{r['dr']}-{r['seed']}"
C={key(r):r for r in C}
ids=[k for k in C if k in P]
print(f'matched {len(ids)} models across predictor and curve files')
cell=np.array(['-'.join(k.split('-')[:3]) for k in ids])
P0=np.array([C[k]['P0'] for k in ids])
W=np.array([float(k.split('-')[0]) for k in ids])
DE=np.array([float(k.split('-')[1]) for k in ids])
DR=np.array([float(k.split('-')[2]) for k in ids])

# ---- functional PCA modes on all curves, then per-family M,T scores ----
allc=np.array([[C[k]['curves'][s] for s in SA] for k in ids]).reshape(-1,10)
mu=allc.mean(0); Uc,sv,Vt=np.linalg.svd(allc-mu,full_matrices=False)
ev=sv**2/(sv**2).sum()
print(f'curve modes: M={100*ev[0]:.0f}%  T={100*ev[1]:.0f}%  (M+T={100*(ev[0]+ev[1]):.0f}%)')
if Vt[0].sum()<0: Vt[0]*=-1                 # M positive = more degradation
def score(k,ss,mode):
    return float(np.mean([ (np.array(C[k]['curves'][s])-mu)@Vt[mode] for s in ss]))
TGT={'M_input':np.array([score(k,INP,0) for k in ids]),
     'T_input':np.array([score(k,INP,1) for k in ids]),
     'M_capacity':np.array([score(k,CAP,0) for k in ids]),
     'T_capacity':np.array([score(k,CAP,1) for k in ids])}

PK=list(P[ids[0]].keys())
X=np.array([[P[k][p] for p in PK] for k in ids])
keep=[i for i in range(len(PK)) if X[:,i].std()>1e-9]
PK=[PK[i] for i in keep]; X=X[:,keep]
print(f'{len(PK)} predictors, {len(set(cell))} architecture cells\n')

# ---- sanity: does P0 predict anything? ----
print('--- baseline check: does P0 predict the targets? ---')
for t,y in TGT.items():
    print(f'  rho(P0, {t:<12}) = {stats.spearmanr(P0,y).statistic:+.3f}')

# ---- grouped CV ----
cells=sorted(set(cell)); rng=np.random.default_rng(0)
folds=[cells[i::4] for i in range(4)]
def resid_fit(Ytr,Ztr):
    A=np.column_stack([Ztr,np.ones(len(Ztr))])
    b=np.linalg.lstsq(A,Ytr,rcond=None)[0]; return b
def apply_res(Y,Z,b):
    A=np.column_stack([Z,np.ones(len(Z))]); return Y-A@b

def cv(Xm,y,ctrl,alphas=(0.1,1,10,100)):
    oof=np.full(len(y),np.nan)
    for f in folds:
        te=np.isin(cell,f); tr=~te
        # residualise on control (P0 [+arch]) using TRAIN fold only
        by=resid_fit(y[tr],ctrl[tr]); ytr=apply_res(y[tr],ctrl[tr],by); yte=apply_res(y[te],ctrl[te],by)
        Xtr,Xte=Xm[tr],Xm[te]
        mn,sd=Xtr.mean(0),Xtr.std(0)+1e-9
        Xtr=(Xtr-mn)/sd; Xte=(Xte-mn)/sd
        bx=resid_fit(Xtr,ctrl[tr]); Xtr=apply_res(Xtr,ctrl[tr],bx); Xte=apply_res(Xte,ctrl[te],bx)
        # inner CV for alpha
        best=(1e18,alphas[0])
        icells=sorted(set(cell[tr]))
        for a in alphas:
            errs=[]
            for g in [icells[i::3] for i in range(3)]:
                ite=np.isin(cell[tr],g); itr=~ite
                A=Xtr[itr]; b_=np.linalg.solve(A.T@A+a*np.eye(A.shape[1]),A.T@ytr[itr])
                errs.append(np.mean((ytr[ite]-Xtr[ite]@b_)**2))
            if np.mean(errs)<best[0]: best=(np.mean(errs),a)
        a=best[1]
        b_=np.linalg.solve(Xtr.T@Xtr+a*np.eye(Xtr.shape[1]),Xtr.T@ytr)
        oof[te]=Xte@b_
        if not hasattr(cv,'_ytrue'): cv._ytrue={}
    # rebuild residualised truth with same folds
    ytrue=np.full(len(y),np.nan)
    for f in folds:
        te=np.isin(cell,f); tr=~te
        by=resid_fit(y[tr],ctrl[tr]); ytrue[te]=apply_res(y[te],ctrl[te],by)
    ss=1-np.sum((ytrue-oof)**2)/np.sum((ytrue-ytrue.mean())**2)
    return ss,stats.spearmanr(ytrue,oof).statistic,np.mean(np.abs(ytrue-oof))

ctrlP0=P0[:,None]
ctrlFull=np.column_stack([P0,np.log(W),DE,DR])

print('\n--- MULTIVARIATE, leave-3-cells-out CV (all 22 predictors, ridge) ---')
print(f'{"target":<14}{"control":<16}{"R2_cv":>9}{"rho":>8}{"MAE":>10}')
for t,y in TGT.items():
    for nm,ct in [('P0',ctrlP0),('P0+arch',ctrlFull)]:
        r2,rh,mae=cv(X,y,ct)
        print(f'  {t:<12}{nm:<16}{r2:>9.3f}{rh:>8.3f}{mae:>10.4f}')

print('\n--- SINGLE PREDICTORS: held-out rank corr, residualised on P0+arch ---')
print('    (reported for ALL predictors; no selection)')
res={}
for t,y in TGT.items():
    rows=[]
    for i,p in enumerate(PK):
        r2,rh,_=cv(X[:,[i]],y,ctrlFull)
        rows.append((p,rh,r2))
    res[t]=rows
hdr=f'{"predictor":<16}'+''.join(f'{t:>14}' for t in TGT)
print(hdr); print('-'*len(hdr))
for i,p in enumerate(PK):
    print(f'  {p:<14}'+''.join(f'{res[t][i][1]:>+14.3f}' for t in TGT))

print('\n--- pre-committed hypotheses ---')
def get(t,p): return [r for r in res[t] if r[0]==p][0]
for p,t,lab in [('fisher_top','M_input','Fisher top eig -> input M'),
                ('jac_rms','M_input','Jacobian RMS -> input M'),
                ('margin_p10','M_input','logit margin p10 -> input M'),
                ('rel_sharp','M_capacity','relative sharpness -> capacity M'),
                ('hess_top','M_capacity','Hessian top eig -> capacity M'),
                ('pr_mean','M_capacity','participation ratio -> capacity M'),
                ('er_first','M_capacity','effective rank -> capacity M'),
                ('dead_frac','M_capacity','dead-unit fraction -> capacity M')]:
    if p in PK:
        _,rh,r2=get(t,p)
        v='SUPPORTED' if abs(rh)>0.3 else 'not supported'
        print(f'  {lab:<38} rho_heldout={rh:+.3f}  R2={r2:+.3f}   {v}')

print('\n--- cross-family specificity (does input predictor stay in its family?) ---')
for p in ['fisher_top','jac_rms','rel_sharp','hess_top','er_first']:
    if p in PK:
        a=get('M_input',p)[1]; b=get('M_capacity',p)[1]
        print(f'  {p:<14} input={a:+.3f}  capacity={b:+.3f}  '
              f'{"family-specific" if abs(abs(a)-abs(b))>0.25 else "general"}')
