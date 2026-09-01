import json, numpy as np
from scipy import stats
from stress2 import STRESSORS, FAMILY, EPS

R=json.load(open('/home/claude/lb3/results/cross2.json'))
n=len(R); S=STRESSORS; ns=len(S)
DIMS=['susceptibility','curvature','area','late_slope']
P0=np.array([r['P0'] for r in R])

# ---- per-stressor Q95 calibration, then B-score ----
Q95={s:{d:np.percentile([r['feats'][s][d] for r in R],95) for d in DIMS} for s in S}
def arm(v,q): return min(1.5,max(0.,1.5*v/q)) if q>1e-12 else 0.
ARM=np.zeros((n,ns,4))
for i,r in enumerate(R):
    for j,s in enumerate(S):
        for k,d in enumerate(DIMS):
            ARM[i,j,k]=arm(r['feats'][s][d],Q95[s][d])
B=ARM.sum(2)                                    # (models x stressors)

print(f'n={n} models  x  {ns} stressors      P0: {P0.min():.3f}-{P0.max():.3f} (sd={P0.std():.4f})')
print('\n── B-score by stressor ──────────────────────────────────────')
print(f'{"stressor":<15}{"family":<10}{"mean":>7}{"sd":>7}{"min":>7}{"max":>7}')
for j,s in enumerate(S):
    print(f'  {s:<13}{FAMILY[s]:<10}{B[:,j].mean():>7.2f}{B[:,j].std():>7.2f}{B[:,j].min():>7.2f}{B[:,j].max():>7.2f}')

# ---- Q1: do rankings persist across stressors? ----
print('\n── Spearman rank correlation between stressors ──────────────')
RH=np.zeros((ns,ns))
for a in range(ns):
    for b in range(ns): RH[a,b]=stats.spearmanr(B[:,a],B[:,b]).statistic
hdr='               '+''.join(f'{s[:6]:>7}' for s in S); print(hdr)
for a,s in enumerate(S):
    print(f'  {s:<13}'+''.join(f'{RH[a,b]:>7.2f}' for b in range(ns)))

off=RH[np.triu_indices(ns,1)]
print(f'\n  mean off-diagonal rho = {off.mean():+.3f}   median = {np.median(off):+.3f}')
print(f'  positive pairs: {(off>0).sum()}/{len(off)}   |rho|>0.5: {(np.abs(off)>0.5).sum()}/{len(off)}')

# within vs between family
fam=np.array([FAMILY[s] for s in S]); wi=[];bw=[]
for a in range(ns):
    for b in range(a+1,ns): (wi if fam[a]==fam[b] else bw).append(RH[a,b])
print(f'  within-family  rho = {np.mean(wi):+.3f} (n={len(wi)})')
print(f'  between-family rho = {np.mean(bw):+.3f} (n={len(bw)})')

# ---- Q2: decomposition B[i,s] = mu + G_i + T_s + R[i,s] ----
mu=B.mean(); G=B.mean(1)-mu; T=B.mean(0)-mu
RES=B-mu-G[:,None]-T[None,:]
vG,vT,vR=(G**2).sum()*ns,(T**2).sum()*n,(RES**2).sum()
tot=vG+vT+vR
print('\n── Variance decomposition  B[i,s] = mu + G_i + T_s + R[i,s] ──')
print(f'  G  model (intrinsic)        {100*vG/tot:>5.1f}%')
print(f'  T  stressor (main effect)   {100*vT/tot:>5.1f}%')
print(f'  R  interaction (specific)   {100*vR/tot:>5.1f}%')

# variance after removing stressor main effect: how much of what's left is G?
share=vG/(vG+vR)
print(f'\n  Of model-attributable variance (G+R):  G = {100*share:.1f}%,  R = {100*(1-share):.1f}%')

# ---- reliability of G: split-half over stressors ----
rng=np.random.default_rng(0); Bz=(B-B.mean(0))/B.std(0)   # z per stressor
sh=[]
for _ in range(2000):
    p=rng.permutation(ns); h1,h2=p[:ns//2],p[ns//2:]
    sh.append(stats.spearmanr(Bz[:,h1].mean(1),Bz[:,h2].mean(1)).statistic)
sh=np.array(sh); sb=2*sh.mean()/(1+sh.mean())
print(f'\n── Reliability of G (split-half over stressors) ─────────────')
print(f'  mean split-half rho = {sh.mean():+.3f}   [{np.percentile(sh,5):+.3f}, {np.percentile(sh,95):+.3f}]')
print(f'  Spearman-Brown corrected (full 11-stressor G) = {sb:+.3f}')

# ---- permutation test on G ----
obs=G.var(); null=[]
Bp=B.copy()
for _ in range(3000):
    Q=np.column_stack([rng.permutation(Bp[:,j]) for j in range(ns)])
    null.append((Q.mean(1)-Q.mean()).var())
null=np.array(null); p=(null>=obs).mean()
print(f'\n  var(G) observed = {obs:.4f}   null mean = {null.mean():.4f}   p = {p:.4f}')

# ---- G extremes ----
o=np.argsort(-G)
print('\n── Most / least globally brittle models ─────────────────────')
print(f'{"":>4}{"P0":>8}{"G":>8}{"Bbar":>8}   width depth drop')
for i in list(o[:3])+list(o[-3:]):
    c=R[i]['cfg']; print(f'  {i:>2}{P0[i]:>8.3f}{G[i]:>+8.3f}{B[i].mean():>8.2f}   {c["w"]:>5}{c["d"]:>6}{c["dr"]:>5}')

# does P0 explain G?
print(f'\n  corr(G, P0) = {np.corrcoef(G,P0)[0,1]:+.3f}   (G should NOT be baseline performance)')

json.dump(dict(B=B.tolist(),G=G.tolist(),T=T.tolist(),P0=P0.tolist(),
               stressors=S,rho=RH.tolist(),
               varG=float(vG/tot),varT=float(vT/tot),varR=float(vR/tot),
               share=float(share),sb=float(sb),p=float(p)),
          open('/home/claude/lb3/results/analysis.json','w'))
