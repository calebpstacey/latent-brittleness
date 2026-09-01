import json,numpy as np,itertools
from scipy import stats
A=json.load(open('results/analysis.json')); R=json.load(open('results/cross2.json'))
B=np.array(A['B']); S=A['stressors']; P0=np.array(A['P0']); n,ns=B.shape
Z=(B-B.mean(0))/B.std(0); C=np.corrcoef(Z.T)
wv,V=np.linalg.eigh(C); o=np.argsort(-wv); V0=V[:,o]; w0=wv[o]
rng=np.random.default_rng(0)

print('='*68); print('BOOTSTRAP with component MATCHING (fixes label switching)'); print('='*68)
NB=2000; K=3; best=np.zeros((NB,K)); Lb=np.zeros((NB,K,ns))
for b in range(NB):
    idx=rng.integers(0,n,n); Zb=Z[idx]; Zb=(Zb-Zb.mean(0))/(Zb.std(0)+1e-9)
    wb,Vb=np.linalg.eigh(np.corrcoef(Zb.T)); ob=np.argsort(-wb); Vb=Vb[:,ob][:,:K]
    # match each reference component to the bootstrap component with max |congruence|
    M=np.abs(V0[:,:K].T@Vb)
    perm=max(itertools.permutations(range(K)),key=lambda p:sum(M[k,p[k]] for k in range(K)))
    for k in range(K):
        v=Vb[:,perm[k]]
        if v@V0[:,k]<0: v=-v
        best[b,k]=abs(v@V0[:,k]); Lb[b,k]=v
print('  Tucker congruence (matched):')
for k in range(K):
    c=best[:,k]; print(f'    PC{k+1}: {c.mean():.3f}  [{np.percentile(c,5):.3f}, {np.percentile(c,95):.3f}]'
                       f'   frac>0.90: {(c>0.90).mean():.2f}')
print('\n  Loading stability (90% CI, * excludes 0):')
print(f'{"stressor":<15}{"PC1":>24}{"PC2":>24}')
for j,s in enumerate(S):
    a=np.percentile(Lb[:,0,j],[5,95]); b2=np.percentile(Lb[:,1,j],[5,95])
    print(f'  {s:<13}{V0[j,0]:>8.2f} [{a[0]:+.2f},{a[1]:+.2f}]{"*" if a[0]*a[1]>0 else " "}'
          f'{V0[j,1]:>8.2f} [{b2[0]:+.2f},{b2[1]:+.2f}]{"*" if b2[0]*b2[1]>0 else " "}')

print(); print('='*68); print('ROTATE vs OCCLUSION: is -0.68 just architecture?'); print('='*68)
ir,io=S.index('rotate'),S.index('occlusion')
arch=np.array([[r['cfg']['w'],r['cfg']['d'],r['cfg']['dr'],r['cfg']['lr'],r['cfg']['wd']] for r in R])
X=np.column_stack([arch,P0])
raw=stats.spearmanr(B[:,ir],B[:,io]).statistic
def resid(y,Xc):
    Xc=np.column_stack([Xc,np.ones(len(Xc))])
    return y-Xc@np.linalg.lstsq(Xc,y,rcond=None)[0]
rr,ro=resid(B[:,ir],X),resid(B[:,io],X)
part=stats.spearmanr(rr,ro).statistic
pr=stats.pearsonr(rr,ro)
print(f'  raw Spearman                     = {raw:+.3f}')
print(f'  partial (| width,depth,drop,lr,wd,P0) = {part:+.3f}   Pearson p={pr.pvalue:.4f}')
print(f'  -> {"SURVIVES architecture control" if abs(part)>0.4 else "largely explained by architecture"}')
print('\n  by depth:')
for d in [1,2,3]:
    m=arch[:,1]==d
    if m.sum()>4: print(f'    depth={d} (n={m.sum():2d}): rho={stats.spearmanr(B[m,ir],B[m,io]).statistic:+.3f}')

print(); print('='*68); print('MULTIVARIABLE: factors ~ arch + P0'); print('='*68)
Xd=np.column_stack([arch[:,0],arch[:,1],arch[:,2],arch[:,3],arch[:,4],P0])
Xs=(Xd-Xd.mean(0))/Xd.std(0); Xs=np.column_stack([Xs,np.ones(n)])
nm=['width','depth','dropout','lr','wd','P0']
print(f'{"":<10}'+''.join(f'{x:>9}' for x in nm)+f'{"R2":>8}')
for k in range(3):
    sc=Z@V0[:,k]; sc=(sc-sc.mean())/sc.std()
    beta,_,_,_=np.linalg.lstsq(Xs,sc,rcond=None)
    r2=1-((sc-Xs@beta)**2).sum()/((sc-sc.mean())**2).sum()
    print(f'  PC{k+1:<7}'+''.join(f'{b:>9.2f}' for b in beta[:6])+f'{r2:>8.2f}')
print('  (standardised betas)')
