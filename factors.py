import json,numpy as np
from scipy import stats
from scipy.cluster.hierarchy import linkage,fcluster
from scipy.spatial.distance import squareform
A=json.load(open('results/analysis.json'))
B=np.array(A['B']); S=A['stressors']; RH=np.array(A['rho']); n,ns=B.shape
Z=(B-B.mean(0))/B.std(0)

# eigen-decomposition of stressor correlation matrix
C=np.corrcoef(Z.T); w,V=np.linalg.eigh(C); o=np.argsort(-w); w,V=w[o],V[:,o]
print('── Eigenvalues of stressor correlation matrix ──────────────')
print('  ' + '  '.join(f'{x:.2f}' for x in w))
print(f'  variance explained: ' + '  '.join(f'PC{i+1}:{100*w[i]/w.sum():.0f}%' for i in range(4)))
print(f'  eigenvalues > 1 (Kaiser): {(w>1).sum()}  -> {(w>1).sum()} independent brittleness factors')

print('\n── Loadings ─────────────────────────────────────────────────')
print(f'{"stressor":<15}{"PC1":>8}{"PC2":>8}{"PC3":>8}')
for j,s in enumerate(S):
    print(f'  {s:<13}'+''.join(f'{V[j,k]*np.sqrt(w[k]):>8.2f}' for k in range(3)))

# hierarchical clustering of stressors
D=1-RH; np.fill_diagonal(D,0); D=(D+D.T)/2
L=linkage(squareform(D,checks=False),'average')
for k in [2,3,4]:
    cl=fcluster(L,k,'maxclust')
    groups={c:[S[i] for i in range(ns) if cl[i]==c] for c in sorted(set(cl))}
    print(f'\n  k={k}: ' + ' | '.join(','.join(g) for g in groups.values()))

# per-factor reliability: how well does each PC score replicate split-half?
rng=np.random.default_rng(1)
print('\n── Are factor scores reliable? (split-half over stressors) ──')
for k in range(3):
    ld=V[:,k]*np.sqrt(w[k]); sh=[]
    for _ in range(2000):
        p=rng.permutation(ns); h1,h2=p[:ns//2],p[ns//2:]
        s1=Z[:,h1]@ld[h1]; s2=Z[:,h2]@ld[h2]
        sh.append(stats.spearmanr(s1,s2).statistic)
    m=np.mean(sh); print(f'  PC{k+1}: split-half rho={m:+.3f}  Spearman-Brown={2*m/(1+m):+.3f}')

# what architecture drives PC1/PC2?
R=json.load(open('results/cross2.json'))
arch=np.array([[r['cfg']['w'],r['cfg']['d'],r['cfg']['dr'],r['cfg']['lr'],r['cfg']['wd']] for r in R])
names=['width','depth','dropout','lr','wd']
G=np.array(A['G'])
print('\n── What predicts each factor? (Spearman) ────────────────────')
print(f'{"":<10}{"G":>9}{"PC1":>9}{"PC2":>9}{"PC3":>9}')
for a,nm in enumerate(names):
    row=f'  {nm:<8}{stats.spearmanr(arch[:,a],G).statistic:>9.2f}'
    for k in range(3):
        sc=Z@(V[:,k]*np.sqrt(w[k]))
        row+=f'{stats.spearmanr(arch[:,a],sc).statistic:>9.2f}'
    print(row)
json.dump(dict(eig=w.tolist(),load=(V*np.sqrt(w)).tolist()),open('results/factors.json','w'))
