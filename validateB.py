import json,numpy as np,itertools
from scipy import stats
FR=json.load(open('/home/claude/lb3/results/FROZEN_A.json'))
R=json.load(open('/home/claude/lb3/results/cohortB.json'))
SA=FR['stressors']; DIMS=FR['dims']; L_A=np.array(FR['loadings'])   # 3 x 11
Q95=FR['Q95']; EPS=np.array([k/9 for k in range(10)]); IDX_A=[0,3,6,9]
FAM={'noise':'input','blur':'input','occlusion':'input','contrast':'input','rotate':'shift',
     'translate':'shift','scale':'shift','weight_drop':'capacity','neuron_ablate':'capacity',
     'mag_prune':'capacity','act_clip':'capacity','abs_weight':'capacity','abs_neuron':'capacity'}
n=len(R); P0=np.array([r['P0'] for r in R])
W=np.array([r['w'] for r in R]); D=np.array([r['d'] for r in R]); DR=np.array([r['dr'] for r in R])
CELL=np.array([f"{r['w']}-{r['d']}-{r['dr']}" for r in R])

def feats(D_,eps):
    h1,h2=eps[1]-eps[0],eps[2]-eps[1]
    return dict(susceptibility=float((D_[1]-D_[0])/h1),
                curvature=float(2*(D_[2]/h2-D_[1]*(1/h1+1/h2)+D_[0]/h1)/(h1+h2)),
                area=float(np.trapezoid(D_,eps)),
                late_slope=float((D_[-1]-D_[-2])/(eps[-1]-eps[-2])))
def bmat(eps_map):
    """B-scores on Cohort A's 11 stressors using FROZEN Q95 anchors."""
    e=eps_map(EPS[IDX_A]); Bm=np.zeros((n,len(SA)))
    for i,r in enumerate(R):
        for j,s in enumerate(SA):
            c=np.array(r['curves'][s])[IDX_A]; f=feats(c,e)
            Bm[i,j]=sum(min(1.5,max(0.,1.5*f[d]/Q95[s][d])) for d in DIMS)
    return Bm

print('='*72); print(f'COHORT B  n={n}   P0 {P0.min():.4f}-{P0.max():.4f}  sd={P0.std():.4f}')
print(f'  (Cohort A was 0.821-0.896, sd=0.0212 -> band is {0.0212/P0.std():.1f}x tighter)')
print(f'  cells: {len(set(CELL))}, seeds/cell: {n//len(set(CELL))}'); print('='*72)

lin=lambda e:e; quad=lambda e:e**2
B=bmat(lin); Bq=bmat(quad)
Z=(B-B.mean(0))/B.std(0)
def resid(Y,x):
    X=np.column_stack([x,np.ones(len(x))])
    return Y-X@np.linalg.lstsq(X,Y,rcond=None)[0]
Zr=resid(Z,P0); Zr=(Zr-Zr.mean(0))/Zr.std(0)
Zq=(Bq-Bq.mean(0))/Bq.std(0)

# ---------- C1 loading coherence ----------
print('\n--- C1  LOADING COHERENCE (frozen A loadings applied to B) ---')
inp=[j for j,s in enumerate(SA) if FAM[s]=='input']; cap=[j for j,s in enumerate(SA) if FAM[s]=='capacity']
def coh(Zx):
    C=np.corrcoef(Zx.T); w_,V=np.linalg.eigh(C); o=np.argsort(-w_); V=V[:,o]
    for k,a in enumerate(['noise','weight_drop','rotate']):
        if V[SA.index(a),k]<0: V[:,k]*=-1
    return (V[inp,0].mean()-V[cap,0].mean(), V[cap,1].mean()-V[inp,1].mean(), V)
d1,d2,V_B=coh(Z)
rng=np.random.default_rng(0); bs=np.zeros((2000,2))
for b in range(2000):
    ix=rng.integers(0,n,n); Zb=Z[ix]; Zb=(Zb-Zb.mean(0))/(Zb.std(0)+1e-9)
    bs[b]=coh(Zb)[:2]
ci1=np.percentile(bs[:,0],[5,95]); ci2=np.percentile(bs[:,1],[5,95])
print(f'  PC1 input-minus-capacity loading gap = {d1:+.3f}  90%CI [{ci1[0]:+.3f},{ci1[1]:+.3f}]  {"PASS" if ci1[0]>0 else "FAIL"}')
print(f'  PC2 capacity-minus-input loading gap = {d2:+.3f}  90%CI [{ci2[0]:+.3f},{ci2[1]:+.3f}]  {"PASS" if ci2[0]>0 else "FAIL"}')
C1 = ci1[0]>0 and ci2[0]>0

# ---------- C2 congruence with frozen A ----------
print('\n--- C2  BOOTSTRAP CONGRUENCE vs FROZEN COHORT A (>=0.85) ---')
def tuck(a,b): return abs(a@b)/np.sqrt((a@a)*(b@b))
obs=[tuck(V_B[:,k],L_A[k]) for k in range(3)]
cb=np.zeros((2000,3))
for b in range(2000):
    ix=rng.integers(0,n,n); Zb=Z[ix]; Zb=(Zb-Zb.mean(0))/(Zb.std(0)+1e-9)
    Vb=coh(Zb)[2]
    M=np.abs(L_A[:3]@Vb[:,:3])
    pm=max(itertools.permutations(range(3)),key=lambda p:sum(M[k,p[k]] for k in range(3)))
    for k in range(3): cb[b,k]=tuck(Vb[:,pm[k]],L_A[k])
for k in range(3):
    lo=np.percentile(cb[:,k],5)
    print(f'  PC{k+1}: congruence={obs[k]:.3f}  bootstrap mean={cb[:,k].mean():.3f} [5%={lo:.3f}]  '
          f'{"PASS" if cb[:,k].mean()>=0.85 else "FAIL"}')
C2 = cb[:,0].mean()>=0.85 and cb[:,1].mean()>=0.85

# ---------- C3/C4/C5 family separation ----------
def sep(Zx):
    C=np.zeros((len(SA),len(SA)))
    for a in range(len(SA)):
        for b in range(len(SA)): C[a,b]=stats.spearmanr(Zx[:,a],Zx[:,b]).statistic
    wi=[C[a,b] for a in range(len(SA)) for b in range(a+1,len(SA)) if FAM[SA[a]]==FAM[SA[b]]]
    bw=[C[a,b] for a in range(len(SA)) for b in range(a+1,len(SA)) if FAM[SA[a]]!=FAM[SA[b]]]
    return np.mean(wi),np.mean(bw)
print('\n--- C3/C4/C5  FAMILY SEPARATION (gap >= 0.15) ---')
res={}
for nm,Zx in [('C3 raw',Z),('C4 residualised on P0',Zr),('C5 eps -> eps^2',Zq)]:
    wi,bw=sep(Zx); g=wi-bw; res[nm]=g
    print(f'  {nm:<24} within={wi:+.3f}  between={bw:+.3f}  gap={g:+.3f}  {"PASS" if g>=0.15 else "FAIL"}')
C3,C4,C5=[res[k]>=0.15 for k in res]

print('\n'+'='*72)
allp=[C1,C2,C3,C4,C5]
for nm,v in zip(['C1 loading coherence','C2 congruence>=.85','C3 family sep',
                 'C4 survives P0 residualisation','C5 survives eps^2'],allp):
    print(f'  {nm:<34}{"PASS" if v else "FAIL"}')
print(f'\n  VERDICT: {"VALIDATED" if all(allp) else ("KILLED" if not(C1 and C2) else "DOWNGRADED")}')
print('='*72)

# ---------- within-cell latent variance (separately reported) ----------
print('\n--- WITHIN-ARCHITECTURE LATENT VARIANCE (not pass/fail) ---')
Gm=Z.mean(1)
cells=sorted(set(CELL)); cm={c:Gm[CELL==c].mean() for c in cells}
between=np.array([cm[c] for c in CELL]); within=Gm-between
vb,vw=between.var(),within.var()
print(f'  var(architecture) = {vb:.4f}   var(model|architecture) = {vw:.4f}')
print(f'  within-architecture share = {100*vw/(vb+vw):.1f}%')
null=[]
for _ in range(5000):
    p=rng.permutation(n); Gp=Gm[p]
    bm={c:Gp[CELL==c].mean() for c in cells}
    null.append((Gp-np.array([bm[c] for c in CELL])).var())
print(f'  permutation p (within-var != chance) = {(np.array(null)<=vw).mean():.4f}')
print(f'  -> same architecture, same P0, still different degradation: '
      f'{"YES" if vw/(vb+vw)>0.15 else "marginal"}')

# ---------- coordinate sensitivity per feature ----------
print('\n--- PER-FEATURE COORDINATE SENSITIVITY (eps -> eps^2) ---')
print(f'{"feature":<16}{"mean rank corr":>16}{"verdict":>12}')
for d in DIMS:
    cs=[]
    for s in SA:
        a=[feats(np.array(r['curves'][s])[IDX_A],EPS[IDX_A])[d] for r in R]
        b=[feats(np.array(r['curves'][s])[IDX_A],EPS[IDX_A]**2)[d] for r in R]
        cs.append(stats.spearmanr(a,b).statistic)
    mc=np.nanmean(cs)
    print(f'  {d:<14}{mc:>16.3f}{("stable" if mc>0.8 else "SENSITIVE"):>12}')

# ---------- fractional vs absolute damage ----------
print('\n--- FRACTIONAL vs ABSOLUTE CAPACITY DAMAGE (width) ---')
for s in ['weight_drop','abs_weight','neuron_ablate','abs_neuron']:
    auc=np.array([np.trapezoid(r['curves'][s],EPS) for r in R])
    print(f'  {s:<15} rho(width)={stats.spearmanr(W,auc).statistic:+.3f}')

# ---------- functional PCA on full curves ----------
print('\n--- FUNCTIONAL PCA ON 10-POINT CURVES (do S,C,A,L match the data modes?) ---')
allc=np.array([[r['curves'][s] for s in SA] for r in R]).reshape(-1,10)
allc=allc-allc.mean(0)
U,sv,Vt=np.linalg.svd(allc,full_matrices=False)
ev=sv**2/ (sv**2).sum()
print('  variance explained: '+'  '.join(f'M{i+1}:{100*ev[i]:.0f}%' for i in range(4)))
for i in range(3):
    print(f'    mode{i+1} shape: '+' '.join(f'{v:+.2f}' for v in Vt[i]))
print('  (mode1 ~ overall magnitude; mode2 ~ early-vs-late tilt; mode3 ~ curvature)')
