"""Strictest test: within architecture cell, architecture is EXACTLY constant.
If a predictor still tracks M/T here, it is seed-level signal, not width leakage."""
import json,numpy as np
from scipy import stats
P=json.load(open('results/predictors.json')); C=json.load(open('results/cohortB.json'))
FR=json.load(open('results/FROZEN_A.json')); SA=FR['stressors']
FAM={'noise':'input','blur':'input','occlusion':'input','contrast':'input','rotate':'shift',
     'translate':'shift','scale':'shift','weight_drop':'capacity','neuron_ablate':'capacity',
     'mag_prune':'capacity','act_clip':'capacity'}
INP=[s for s in SA if FAM[s]=='input']; CAP=[s for s in SA if FAM[s]=='capacity']
key=lambda r:f"{r['w']}-{r['d']}-{r['dr']}-{r['seed']}"
C={key(r):r for r in C}; ids=[k for k in C if k in P]
cell=np.array(['-'.join(k.split('-')[:3]) for k in ids])
allc=np.array([[C[k]['curves'][s] for s in SA] for k in ids]).reshape(-1,10)
mu=allc.mean(0); _,sv,Vt=np.linalg.svd(allc-mu,full_matrices=False)
if Vt[0].sum()<0: Vt[0]*=-1
sc=lambda k,ss,m: float(np.mean([(np.array(C[k]['curves'][s])-mu)@Vt[m] for s in ss]))
TGT={'M_input':np.array([sc(k,INP,0) for k in ids]),'T_input':np.array([sc(k,INP,1) for k in ids]),
     'M_capacity':np.array([sc(k,CAP,0) for k in ids]),'T_capacity':np.array([sc(k,CAP,1) for k in ids])}
PK=[p for p in P[ids[0]] if np.std([P[k][p] for k in ids])>1e-9]
X=np.array([[P[k][p] for p in PK] for k in ids])
P0=np.array([C[k]['P0'] for k in ids])

def wcenter(v):
    o=v.copy().astype(float)
    for c in sorted(set(cell)):
        m=cell==c; o[m]=v[m]-v[m].mean()
    return o
Xw=np.column_stack([wcenter(X[:,i]) for i in range(X.shape[1])])
P0w=wcenter(P0)
# also strip residual P0 variation inside cells
def strip(y):
    A=np.column_stack([P0w,np.ones(len(P0w))]); return y-A@np.linalg.lstsq(A,y,rcond=None)[0]

print(f'n={len(ids)}, {len(set(cell))} cells, {len(ids)//len(set(cell))} seeds/cell')
print('WITHIN-CELL: architecture exactly constant, P0 stripped\n')
print(f'{"predictor":<16}'+''.join(f'{t:>13}' for t in TGT))
print('-'*(16+13*4))
out={}
for i,p in enumerate(PK):
    x=strip(Xw[:,i]); row=f'  {p:<14}'
    out[p]={}
    for t,y in TGT.items():
        r=stats.spearmanr(x,strip(wcenter(y)))
        out[p][t]=(r.statistic,r.pvalue)
        star='*' if r.pvalue<0.05 else ' '
        row+=f'{r.statistic:>+12.3f}{star}'
    print(row)
print('  (* p<0.05, uncorrected)')

print('\n--- Bonferroni across 22 predictors x 4 targets (88 tests, alpha=0.05/88) ---')
thr=0.05/88; any_=False
for p in PK:
    for t in TGT:
        r,pv=out[p][t]
        if pv<thr: print(f'  {p:<16} -> {t:<12} rho={r:+.3f}  p={pv:.2e}  SURVIVES'); any_=True
if not any_: print('  none survive Bonferroni correction')

print('\n--- how much of er_first is just width? ---')
W=np.array([float(k.split("-")[0]) for k in ids])
for p in ['er_first','pr_mean','hess_top','fisher_top','jac_rms']:
    if p in PK:
        i=PK.index(p)
        print(f'  {p:<12} rho(pred,width)={stats.spearmanr(X[:,i],W).statistic:+.3f}   '
              f'rho(M_capacity,width)={stats.spearmanr(TGT["M_capacity"],W).statistic:+.3f}')
