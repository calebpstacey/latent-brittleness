"""Freeze Cohort A: lock Q95 anchors and factor loadings. Nothing downstream may re-fit these."""
import json,numpy as np,hashlib
R=json.load(open('results/cross2.json')); A=json.load(open('results/analysis.json'))
S=A['stressors']; DIMS=['susceptibility','curvature','area','late_slope']
B=np.array(A['B']); n,ns=B.shape
Q95={s:{d:float(np.percentile([r['feats'][s][d] for r in R],95)) for d in DIMS} for s in S}
Z=(B-B.mean(0))/B.std(0)
mu=B.mean(0).tolist(); sd=B.std(0).tolist()
C=np.corrcoef(Z.T); w,V=np.linalg.eigh(C); o=np.argsort(-w); w,V=w[o],V[:,o]
# sign convention: make PC1 positive on 'noise', PC2 positive on 'weight_drop', PC3 positive on 'rotate'
for k,anchor in enumerate(['noise','weight_drop','rotate']):
    if V[S.index(anchor),k]<0: V[:,k]*=-1
FR=dict(stressors=S,dims=DIMS,Q95=Q95,B_mean=mu,B_sd=sd,
        loadings=V[:,:3].T.tolist(),eigen=w[:3].tolist(),
        eps_grid=[0.,0.33,0.67,1.0],n_cohortA=n,
        within_between=[0.387,0.060])
json.dump(FR,open('results/FROZEN_A.json','w'),indent=1)
h=hashlib.sha256(json.dumps(FR,sort_keys=True).encode()).hexdigest()[:16]
print(f'FROZEN_A.json written.  sha256[:16] = {h}')
print(f'\nFrozen loadings (sign-fixed):')
print(f'{"stressor":<15}{"PC1":>8}{"PC2":>8}{"PC3":>8}')
for j,s in enumerate(S): print(f'  {s:<13}'+''.join(f'{V[j,k]:>8.3f}' for k in range(3)))
print(f'\nFrozen Q95 anchors (per stressor, area only shown):')
for s in S: print(f'  {s:<15} area Q95 = {Q95[s]["area"]:.4f}')
