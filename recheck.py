import json,numpy as np
from scipy import stats
A=json.load(open('results/analysis.json')); R=json.load(open('results/cross2.json'))
B=np.array(A['B']); S=A['stressors']; G=np.array(A['G']); P0=np.array(A['P0'])
n,ns=B.shape; Z=(B-B.mean(0))/B.std(0); C=np.corrcoef(Z.T)
w=np.sort(np.linalg.eigvalsh(C))[::-1]
rng=np.random.default_rng(0)

print('='*66); print('1. PERMUTATION p-VALUE  (reporting floor)'); print('='*66)
obs=G.var(); ex=0; NP=20000
for _ in range(NP):
    Q=np.column_stack([rng.permutation(B[:,j]) for j in range(ns)])
    if (Q.mean(1)-Q.mean()).var()>=obs: ex+=1
print(f'  {ex}/{NP} permutations exceeded observed.  empirical p < {1/(NP+1):.2e}')
print(f'  (previous "p<0.0001" from 3000 perms was below the {1/3001:.5f} floor - was wrong)')

print(); print('='*66); print('2. PARALLEL ANALYSIS  (does PC3 survive?)'); print('='*66)
NR=2000; null=np.zeros((NR,ns))
for i in range(NR):
    Xr=rng.standard_normal((n,ns))
    null[i]=np.sort(np.linalg.eigvalsh(np.corrcoef(Xr.T)))[::-1]
p95=np.percentile(null,95,axis=0); p50=np.percentile(null,50,axis=0)
print(f'{"PC":<5}{"observed":>10}{"random p50":>12}{"random p95":>12}{"verdict":>12}')
for k in range(6):
    v='RETAIN' if w[k]>p95[k] else 'drop'
    print(f'  {k+1:<3}{w[k]:>10.2f}{p50[k]:>12.2f}{p95[k]:>12.2f}{v:>12}')
nret=int((w>p95).sum())
print(f'\n  Parallel analysis retains {nret} factors (Kaiser said 3).')

print(); print('='*66); print('3. BOOTSTRAP FACTOR STABILITY'); print('='*66)
wv,V=np.linalg.eigh(C); o=np.argsort(-wv); V0=V[:,o]
NB=1000; L1=np.zeros((NB,ns)); L2=np.zeros((NB,ns)); congr=np.zeros((NB,2))
for b in range(NB):
    idx=rng.integers(0,n,n); Zb=Z[idx]
    Zb=(Zb-Zb.mean(0))/(Zb.std(0)+1e-9)
    wb,Vb=np.linalg.eigh(np.corrcoef(Zb.T)); ob=np.argsort(-wb); Vb=Vb[:,ob]
    for k in range(2):
        v=Vb[:,k]
        if v@V0[:,k]<0: v=-v
        (L1 if k==0 else L2)[b]=v
        congr[b,k]=abs(v@V0[:,k])/np.sqrt((v@v)*(V0[:,k]@V0[:,k]))
print(f'  Tucker congruence with full-sample solution:')
print(f'    PC1: {congr[:,0].mean():.3f}  [{np.percentile(congr[:,0],5):.3f}, {np.percentile(congr[:,0],95):.3f}]')
print(f'    PC2: {congr[:,1].mean():.3f}  [{np.percentile(congr[:,1],5):.3f}, {np.percentile(congr[:,1],95):.3f}]')
print(f'  (>0.95 = equivalent, 0.85-0.95 = fair)')
print(f'\n  Loading 90% CIs:')
print(f'{"stressor":<15}{"PC1 loading":>22}{"PC2 loading":>22}')
for j,s in enumerate(S):
    a=np.percentile(L1[:,j],[5,95]); b2=np.percentile(L2[:,j],[5,95])
    f1='*' if a[0]*a[1]>0 else ' '; f2='*' if b2[0]*b2[1]>0 else ' '
    print(f'  {s:<13}{V0[j,0]:>8.2f} [{a[0]:+.2f},{a[1]:+.2f}]{f1}{V0[j,1]:>8.2f} [{b2[0]:+.2f},{b2[1]:+.2f}]{f2}')
print('  (* = 90% CI excludes zero)')
