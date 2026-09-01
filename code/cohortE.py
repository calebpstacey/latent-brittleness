"""
================================================================================
COHORT E — clean two-regime test.  MNIST+MLP and CIFAR-10+CNN, one protocol.
Supersedes Cohort C. Preregistered in PREREG_E.md BEFORE this was run.
================================================================================

    pip install torch torchvision numpy scipy pillow
    python cohortE.py mnist      # arm 1   (~20 min on a T4)
    python cohortE.py cifar      # arm 2   (~35 min on a T4)
    python cohortE.py analyse    # both arms, preregistered verdict

Resumable: re-run the same command after any interruption.

THREE DESIGN FIXES vs Cohort B/C, all specified in PREREG_E.md before running:
  1. FIXED training budget, identical for every model  -> no epoch confound
  2. NO baseline band, every model retained            -> 100% yield
  3. THREE-WAY split: train / match / stress, disjoint -> no test-set reuse
Baseline equivalence is enforced statistically (residualise P0, block on architecture),
not by selection.
"""
import io, json, itertools, random, sys, time, urllib.request, zipfile
from pathlib import Path
import numpy as np
import torch, torch.nn as nn, torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from scipy import stats

HERE = Path.cwd(); OUT = HERE/'results'; OUT.mkdir(exist_ok=True)
DEV = ('cuda' if torch.cuda.is_available()
       else 'mps' if getattr(torch.backends,'mps',None) and torch.backends.mps.is_available()
       else 'cpu')

# ============================ CONFIG (preregistered) ==========================
SEEDS_PER_CELL = 6
EPOCHS         = 25          # FIXED for every model. No scanning, no selection.
N_TRAIN        = 30000
N_MATCH        = 2000        # reports P0 only
N_STRESS       = 2000        # touched ONLY by the stress sweep
EPS  = np.array([k/9 for k in range(10)])
REPS = 2

CELLS_MNIST = [(w,d,dr) for w in (64,128,256) for d in (1,2) for dr in (0.1,0.3)]
CELLS_CIFAR = [(c,b,dr) for c in (16,32,64)   for b in (2,3) for dr in (0.1,0.3)]

FAM = {'noise':'input','blur':'input','occlusion':'input','contrast':'input',
       'rotate':'shift','translate':'shift','scale':'shift',
       'weight_drop':'capacity','unit_ablate':'capacity','mag_prune':'capacity',
       'act_clip':'capacity','abs_weight':'capacity','abs_unit':'capacity'}

# ============================ DATA ============================================
def _fetch_mnist_raw():
    """torchvision's MNIST mirrors 403 intermittently; fall back to a GitHub mirror."""
    import gzip, shutil
    raw = HERE/'data'/'MNIST'/'raw'; raw.mkdir(parents=True, exist_ok=True)
    files = ['train-images-idx3-ubyte','train-labels-idx1-ubyte',
             't10k-images-idx3-ubyte','t10k-labels-idx1-ubyte']
    if all((raw/f).exists() for f in files): return
    base = 'https://raw.githubusercontent.com/fgnt/mnist/master/'
    for f in files:
        if (raw/f).exists(): continue
        print(f'  fetching {f}...', flush=True)
        urllib.request.urlretrieve(base+f+'.gz', raw/(f+'.gz'))
        with gzip.open(raw/(f+'.gz'),'rb') as i, open(raw/f,'wb') as o:
            shutil.copyfileobj(i,o)

def data_mnist():
    from torchvision import datasets, transforms
    t = transforms.ToTensor()
    try:
        tr = datasets.MNIST(HERE/'data', train=True,  download=True, transform=t)
        te = datasets.MNIST(HERE/'data', train=False, download=True, transform=t)
    except Exception:
        print('torchvision MNIST download failed, using mirror...', flush=True)
        _fetch_mnist_raw()
        tr = datasets.MNIST(HERE/'data', train=True,  download=False, transform=t)
        te = datasets.MNIST(HERE/'data', train=False, download=False, transform=t)
    X = torch.stack([tr[i][0] for i in range(N_TRAIN)])
    y = torch.tensor([tr[i][1] for i in range(N_TRAIN)])
    Xe = torch.stack([te[i][0] for i in range(len(te))])
    ye = torch.tensor([te[i][1] for i in range(len(te))])
    return X, y, Xe, ye

def data_cifar():
    cache = HERE/'cifarE.pt'
    if cache.exists(): return torch.load(cache)
    src = HERE/'CIFAR-10-images-master'
    if not src.exists():
        print('Downloading CIFAR-10...', flush=True)
        u='https://codeload.github.com/YoongiKim/CIFAR-10-images/zip/refs/heads/master'
        zipfile.ZipFile(io.BytesIO(urllib.request.urlopen(u,timeout=900).read())).extractall(HERE)
    from PIL import Image
    tr=sorted((src/'train').glob('*/*.jpg')); te=sorted((src/'test').glob('*/*.jpg'))
    cls=sorted({p.parent.name for p in tr})
    random.seed(0); random.shuffle(tr); random.shuffle(te)
    def load(ps):
        X=np.stack([np.array(Image.open(p)) for p in ps]).astype(np.float32)/255.
        return torch.tensor(X).permute(0,3,1,2), torch.tensor([cls.index(p.parent.name) for p in ps])
    print(f'Encoding {N_TRAIN} train + {len(te)} test...', flush=True)
    out=(*load(tr[:N_TRAIN]), *load(te))
    torch.save(out,cache); return out

# ============================ MODELS ==========================================
class MLP(nn.Module):
    def __init__(s,w,d,dr):
        super().__init__()
        s.layers=nn.ModuleList(); s.drops=nn.ModuleList()
        dims=[784]+[w]*d
        for i in range(d):
            s.layers.append(nn.Linear(dims[i],dims[i+1])); s.drops.append(nn.Dropout(dr))
        s.head=nn.Linear(w,10); s.relu=nn.ReLU(); s.unit_mask=None; s.act_clip=None
    def forward(s,x):
        x=x.view(x.size(0),-1)
        for i,(l,dr) in enumerate(zip(s.layers,s.drops)):
            x=s.relu(l(x))
            if s.act_clip is not None: x=torch.clamp(x,max=s.act_clip)
            if s.unit_mask is not None: x=x*s.unit_mask[i]
            x=dr(x)
        return s.head(x)
    def units(s): return [l.out_features for l in s.layers]

class CNN(nn.Module):
    def __init__(s,ch,blocks,dr):
        super().__init__()
        s.convs=nn.ModuleList(); s.bns=nn.ModuleList(); c=3
        for b in range(blocks):
            o=ch*(2**b); s.convs.append(nn.Conv2d(c,o,3,padding=1))
            s.bns.append(nn.BatchNorm2d(o)); c=o
        s.drop=nn.Dropout(dr); s.head=nn.Linear(c,10)
        s.unit_mask=None; s.act_clip=None
    def forward(s,x):
        for i,(cv,bn) in enumerate(zip(s.convs,s.bns)):
            x=F.relu(bn(cv(x)))
            if s.act_clip is not None: x=torch.clamp(x,max=s.act_clip)
            if s.unit_mask is not None: x=x*s.unit_mask[i].view(1,-1,1,1)
            x=F.max_pool2d(x,2)
        return s.head(s.drop(F.adaptive_avg_pool2d(x,1).flatten(1)))
    def units(s): return [c.out_channels for c in s.convs]

# ============================ STRESSORS =======================================
def _pg(t): return torch.Generator(device='cpu').manual_seed(abs(hash(t))%(2**31))
def s_noise(X,e,sc):
    if e==0: return X
    return torch.clamp(X+torch.randn(X.shape,generator=_pg('noise')).to(X.device)*(e*sc['noise']),0,1)
def s_blur(X,e,sc):
    n=int(round(e*sc['blur']))
    if n<1: return X
    C=X.shape[1]; k=(torch.ones(C,1,3,3)/9.).to(X.device); b=X.clone()
    for _ in range(n): b=F.conv2d(b,k,padding=1,groups=C)
    return torch.clamp(b,0,1)
def s_occlusion(X,e,sc):
    sd=int(round(e*sc['occl']))
    if sd<1: return X
    Z=X.clone(); g=_pg('occl'); S=X.shape[-1]
    for i in range(len(Z)):
        r=torch.randint(0,max(1,S-sd),(2,),generator=g); Z[i,:,r[0]:r[0]+sd,r[1]:r[1]+sd]=0
    return Z
def s_contrast(X,e,sc):
    m=X.mean(dim=(1,2,3),keepdim=True); return torch.clamp((X-m)*(1-sc['contrast']*e)+m,0,1)
def _aff(X,angle=0.,tx=0.,ty=0.,s_=1.):
    th=torch.tensor(angle*np.pi/180.); co,si=torch.cos(th)/s_,torch.sin(th)/s_
    M=torch.tensor([[co,-si,tx],[si,co,ty]],dtype=torch.float32).unsqueeze(0).repeat(len(X),1,1).to(X.device)
    return F.grid_sample(X,F.affine_grid(M,X.shape,align_corners=False),align_corners=False,padding_mode='zeros')
def s_rotate(X,e,sc):    return X if e==0 else _aff(X,angle=e*sc['rot'])
def s_translate(X,e,sc): return X if e==0 else _aff(X,tx=e*sc['tr'],ty=e*sc['tr']*0.6)
def s_scale(X,e,sc):     return X if e==0 else _aff(X,s_=1-sc['sc']*e)
INPUT_S=dict(noise=s_noise,blur=s_blur,occlusion=s_occlusion,contrast=s_contrast,
             rotate=s_rotate,translate=s_translate,scale=s_scale)
SCALE={'mnist':dict(noise=0.6,blur=20,occl=16,contrast=0.8,rot=30,tr=0.12,sc=0.25),
       'cifar':dict(noise=0.35,blur=8,occl=18,contrast=0.9,rot=40,tr=0.25,sc=0.35)}

class Perturb:
    def __init__(s,m): s.m=m
    def __enter__(s): s.sv=[p.data.clone() for p in s.m.parameters()]; return s.m
    def __exit__(s,*a):
        for p,q in zip(s.m.parameters(),s.sv): p.data.copy_(q)
        s.m.unit_mask=None; s.m.act_clip=None
def _g(k): return torch.Generator().manual_seed(k)

def d_frac_weight(m,e,sd):
    for n_,p in m.named_parameters():
        if 'weight' in n_ and p.dim()>1:
            p.data*=(torch.rand(p.shape,generator=_g(sd))>e*0.8).float().to(p.device)
def d_abs_weight(m,e,sd,keep=3000):
    """Absolute phi: fixed PER-LAYER retained-weight cap, NOT a global budget."""
    k0=int(round(keep*(1-e)+150*e))
    for n_,p in m.named_parameters():
        if 'weight' in n_ and p.dim()>1:
            k=min(k0,p.numel())
            thr=p.data.abs().flatten().kthvalue(p.numel()-k+1).values
            p.data*=(p.data.abs()>=thr).float()
def d_frac_unit(m,e,sd):
    m.unit_mask=[(torch.rand(u,generator=_g(sd))>e*0.85).float().to(DEV) for u in m.units()]
def d_abs_unit(m,e,sd,keep=16):
    """Absolute phi: fixed retained-unit cap PER LAYER."""
    k0=int(round(keep*(1-e)+1*e)); ms=[]
    for u in m.units():
        k=min(k0,u); mk=torch.zeros(u); mk[torch.randperm(u,generator=_g(sd))[:k]]=1.
        ms.append(mk.to(DEV))
    m.unit_mask=ms
def d_mag_prune(m,e,sd):
    for n_,p in m.named_parameters():
        if 'weight' in n_ and p.dim()>1:
            k=int(p.numel()*e*0.95)
            if k>0:
                thr=p.data.abs().flatten().kthvalue(k).values
                p.data*=(p.data.abs()>thr).float()
def d_act_clip(m,e,sd): m.act_clip=float(4.0*(0.02**e))
MODEL_S=dict(weight_drop=d_frac_weight,unit_ablate=d_frac_unit,mag_prune=d_mag_prune,
             act_clip=d_act_clip,abs_weight=d_abs_weight,abs_unit=d_abs_unit)
STRESSORS=list(INPUT_S)+list(MODEL_S)

# ============================ RUN ONE ARM =====================================
def run(arm):
    Xtr,ytr,Xte,yte = data_mnist() if arm=='mnist' else data_cifar()
    # THREE-WAY SPLIT: match and stress sets are disjoint and never overlap.
    XM,YM = Xte[:N_MATCH].to(DEV), yte[:N_MATCH].to(DEV)
    XS,YS = Xte[N_MATCH:N_MATCH+N_STRESS].to(DEV), yte[N_MATCH:N_MATCH+N_STRESS].to(DEV)
    TL = DataLoader(TensorDataset(Xtr,ytr),batch_size=256,shuffle=True)
    CELLS = CELLS_MNIST if arm=='mnist' else CELLS_CIFAR
    Net   = MLP if arm=='mnist' else CNN
    sc    = SCALE[arm]

    @torch.no_grad()
    def acc(m,X,Y):
        m.eval(); c=0
        for i in range(0,len(X),500): c+=(m(X[i:i+500]).argmax(1)==Y[i:i+500]).sum().item()
        return c/len(X)
    def curve(m,s):
        if s in INPUT_S:
            f=INPUT_S[s]; a=[acc(m,f(XS,float(e),sc),YS) for e in EPS]
        else:
            f=MODEL_S[s]; a=[]
            for e in EPS:
                if e==0: a.append(acc(m,XS,YS)); continue
                v=[]
                for r in range(REPS):
                    with Perturb(m) as mm: f(mm,float(e),r); v.append(acc(mm,XS,YS))
                a.append(float(np.mean(v)))
        a=np.array(a); return (a[0]-a).tolist()

    path=OUT/f'cohortE_{arm}.json'
    recs=json.load(open(path)) if path.exists() else []
    done={(r['a'],r['b'],r['dr'],r['seed']) for r in recs}
    t0=time.time()
    print(f'\n=== ARM: {arm} | device {DEV} | {len(CELLS)} cells x {SEEDS_PER_CELL} seeds')
    print(f'    FIXED {EPOCHS} epochs, no band, all models retained')
    print(f'    split: {N_TRAIN} train / {N_MATCH} match / {N_STRESS} stress (disjoint)')
    if recs: print(f'    resuming: {len(recs)} done')
    for A,B,dr in CELLS:
        for sd in range(SEEDS_PER_CELL):
            if (A,B,dr,sd) in done: continue
            torch.manual_seed(9000+1000*sd+A+B)
            m=Net(A,B,dr).to(DEV)
            opt=optim.Adam(m.parameters(),lr=1e-3 if arm=='mnist' else 2e-3)
            for _ in range(EPOCHS):                    # FIXED budget. No early stop.
                m.train()
                for X,y in TL:
                    X,y=X.to(DEV),y.to(DEV)
                    opt.zero_grad(); nn.CrossEntropyLoss()(m(X),y).backward(); opt.step()
            P0=acc(m,XM,YM)                            # reported on MATCH set only
            recs.append(dict(a=A,b=B,dr=dr,seed=sd,P0=float(P0),arm=arm,
                             curves={s:curve(m,s) for s in STRESSORS}))
            json.dump(recs,open(path,'w'))
            print(f'  {A:>3}/{B}/{dr}  s{sd}  P0={P0:.4f}   [{time.time()-t0:.0f}s]',flush=True)
    P=np.array([r['P0'] for r in recs])
    print(f'\n{arm}: {len(recs)} models (100% retained), P0 {P.min():.3f}-{P.max():.3f} sd {P.std():.4f}')
    return recs

# ============================ ANALYSE =========================================
def analyse():
    arms={}
    for a in ['mnist','cifar']:
        p=OUT/f'cohortE_{a}.json'
        if p.exists(): arms[a]=json.load(open(p))
    if not arms: print('No data. Run: python cohortE.py mnist'); return
    v={q:{} for q in ['Q1','Q2','Q3','Q4','Q5']}
    for arm,R in arms.items():
        n=len(R); SS=list(R[0]['curves'])
        WD=np.array([r['a'] for r in R]); P0=np.array([r['P0'] for r in R])
        cell=np.array([f"{r['a']}-{r['b']}-{r['dr']}" for r in R]); cells=sorted(set(cell))
        CU=np.array([[r['curves'][s] for s in SS] for r in R])
        auc=lambda s: np.array([np.trapezoid(r['curves'][s],EPS) for r in R])
        Z=np.column_stack([(auc(s)-auc(s).mean())/(auc(s).std()+1e-12) for s in SS])
        def rs(y,X):
            A=np.column_stack([X,np.ones(len(X))]); return y-A@np.linalg.lstsq(A,y,rcond=None)[0]
        Zr=np.column_stack([rs(Z[:,j],P0[:,None]) for j in range(Z.shape[1])])
        sizes=[int((cell==c).sum()) for c in cells]
        evaluable = min(sizes)>=4 and len(cells)>=3
        print('\n'+'='*74)
        print(f'ARM {arm.upper()}  n={n}, {len(cells)} cells, min cell {min(sizes)}, '
              f'P0 {P0.min():.3f}-{P0.max():.3f} (sd {P0.std():.4f})')
        print('='*74)
        if not evaluable: print('*** underpowered: Q1/Q2 -> NOT EVALUABLE (prereg rule) ***')

        def wc(M):
            g=M.mean(1); return g-np.array([g[cell==c].mean() for c in cell])
        # Q1
        obs=wc(Zr).var(); rng=np.random.default_rng(0); NP=5000; exc=0
        for _ in range(NP):
            Zp=Zr.copy()
            for c in cells:
                ix=np.where(cell==c)[0]
                for j in range(Zr.shape[1]): Zp[ix,j]=Zr[rng.permutation(ix),j]
            if wc(Zp).var()>=obs: exc+=1
        p1=(exc+1)/(NP+1)
        v['Q1'][arm]= (p1<0.01) if evaluable else None
        print(f'Q1 persistence      p={p1:.2e}   {"PASS" if v["Q1"][arm] else ("NOT EVALUABLE" if not evaluable else "fail")}')
        # Q2
        def gap(M):
            C={(a_,b_):stats.spearmanr(M[:,a_],M[:,b_]).statistic
               for a_,b_ in itertools.combinations(range(len(SS)),2)}
            wi=[x for (a_,b_),x in C.items() if FAM[SS[a_]]==FAM[SS[b_]]]
            bw=[x for (a_,b_),x in C.items() if FAM[SS[a_]]!=FAM[SS[b_]]]
            return np.mean(wi)-np.mean(bw)
        g2=gap(Zr); v['Q2'][arm]=(g2>=0.15) if evaluable else None
        print(f'Q2 family gap       {g2:+.3f}   {"PASS" if v["Q2"][arm] else ("NOT EVALUABLE" if not evaluable else "fail")}')
        # Q3
        ok=[]
        for f_,a_,lab in [('weight_drop','abs_weight','weights'),('unit_ablate','abs_unit','units')]:
            rf=stats.spearmanr(WD,auc(f_)).statistic; ra=stats.spearmanr(WD,auc(a_)).statistic
            good=rf*ra<0 and abs(rf)>0.3 and abs(ra)>0.3; ok.append(good)
            print(f'Q3 {lab:<10} frac {rf:+.3f}  abs {ra:+.3f}   {"REVERSES" if good else "no"}')
        v['Q3'][arm]=any(ok)
        # Q4 + monotone null
        flat=CU.reshape(-1,10); flat=flat-flat.mean(0)
        _,sv,Vt=np.linalg.svd(flat,full_matrices=False); ev=sv**2/(sv**2).sum()
        g=np.random.default_rng(1); nulls=[]
        for _ in range(20):
            Nl=[]
            for r_ in range(len(flat)):
                inc=g.random(9)+0.05; c_=np.concatenate([[0],np.cumsum(inc)]); Nl.append(c_/c_[-1])
            Nl=np.array(Nl); Nl=Nl-Nl.mean(0)
            _,s2,_=np.linalg.svd(Nl,full_matrices=False); nulls.append((s2**2/(s2**2).sum())[1])
        m1=Vt[0]*np.sign(Vt[0].sum()); mono=bool((np.diff(m1)>=-0.05).all())
        v['Q4'][arm]=(ev[0]+ev[1])>=0.90 and mono
        print(f'Q4 M {100*ev[0]:.1f}% T {100*ev[1]:.1f}% (null T {100*np.mean(nulls):.1f}%) '
              f'M+T {100*(ev[0]+ev[1]):.1f}%  {"PASS" if v["Q4"][arm] else "fail"}')
        # Q5
        IDX=[0,3,6,9]
        def ft(D,e):
            h1,h2=e[1]-e[0],e[2]-e[1]
            return dict(susceptibility=(D[1]-D[0])/h1,
                        curvature=2*(D[2]/h2-D[1]*(1/h1+1/h2)+D[0]/h1)/(h1+h2),
                        area=np.trapezoid(D,e),late_slope=(D[-1]-D[-2])/(e[-1]-e[-2]))
        st={}
        for d in ['susceptibility','curvature','area','late_slope']:
            cs=[]
            for s in SS:
                A_=[ft(np.array(r['curves'][s])[IDX],EPS[IDX])[d] for r in R]
                B_=[ft(np.array(r['curves'][s])[IDX],EPS[IDX]**2)[d] for r in R]
                c_=stats.spearmanr(A_,B_).statistic
                if np.isfinite(c_): cs.append(c_)
            st[d]=float(np.mean(cs))
        order=sorted(st,key=st.get); margin=st[order[1]]-st[order[0]]
        v['Q5'][arm]= order[0]=='curvature' and margin>0.02
        print(f'Q5 lowest={order[0]} ({st[order[0]]:.3f}), margin {margin:.3f}  '
              f'{"PASS" if v["Q5"][arm] else "fail"}')

    print('\n'+'='*74+'\nPREREGISTERED VERDICT\n'+'='*74)
    npass=0; nev=0
    for q in ['Q1','Q2','Q3','Q4','Q5']:
        vals=list(v[q].values())
        if any(x is None for x in vals): st='NOT EVALUABLE'; nev+=1
        elif q=='Q3': st='PASS' if any(vals) else 'FAIL'
        else: st='PASS' if all(vals) else 'FAIL'
        if st=='PASS': npass+=1
        print(f'  {q}  {st:<14} {v[q]}')
    tot=5-nev
    print(f'\n  {npass}/{tot} evaluable questions passed ({nev} not evaluable)')
    if tot>0 and npass/tot>=0.8: print('  -> theory GENERALISES across regimes')
    elif tot>0 and npass/tot>=0.4: print('  -> PARTIAL; failures become boundary conditions')
    elif tot>0: print('  -> effects appear specific to the exploratory regime')
    json.dump({q:{k:(None if x is None else bool(x)) for k,x in v[q].items()}
               for q in v}, open(OUT/'cohortE_verdict.json','w'), indent=1)
    print(f'\nWritten: results/cohortE_verdict.json')

if __name__=='__main__':
    cmd=sys.argv[1] if len(sys.argv)>1 else 'analyse'
    if cmd in ('mnist','cifar'): run(cmd)
    elif cmd=='both': run('mnist'); run('cifar'); analyse()
    else: analyse()
