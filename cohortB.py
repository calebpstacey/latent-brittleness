"""
Cohort B — blocked validation cohort.
- 12 architecture cells x >=4 seeds, baseline-matched by epoch snapshot
- 10-point eps grid k/9 (contains Cohort A's 0,.33,.67,1 exactly)
- capacity damage in BOTH fractional and fixed-absolute parameterisation
- full curves retained
"""
import json,sys,copy,numpy as np,torch,torch.nn as nn,torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets,transforms
from torch.utils.data import DataLoader,Subset
from pathlib import Path

ROOT=Path('/home/claude/lb3'); OUT=ROOT/'results'
TRAIN_N=5000; EVAL_N=800; MAXEP=14
P0_LO,P0_HI,P0_TGT=0.855,0.885,0.870
EPS=np.array([k/9 for k in range(10)])
IDX_A=[0,3,6,9]                                  # Cohort-A subgrid inside EPS

tfm=transforms.Compose([transforms.ToTensor()])
_tr=datasets.MNIST(ROOT/'data',train=True,download=False,transform=tfm)
_te=datasets.MNIST(ROOT/'data',train=False,download=False,transform=tfm)
TL=DataLoader(Subset(_tr,range(TRAIN_N)),batch_size=256,shuffle=True)
_e=DataLoader(Subset(_te,range(EVAL_N)),batch_size=400)
EX=torch.cat([x for x,_ in _e]); EY=torch.cat([y for _,y in _e])

class MLP(nn.Module):
    def __init__(s,w,dr,d):
        super().__init__()
        s.layers=nn.ModuleList(); s.drops=nn.ModuleList()
        dims=[784]+[w]*d
        for i in range(d):
            s.layers.append(nn.Linear(dims[i],dims[i+1])); s.drops.append(nn.Dropout(dr))
        s.head=nn.Linear(w,10); s.relu=nn.ReLU(); s.act_clip=None; s.neuron_mask=None
    def forward(s,x):
        x=x.view(x.size(0),-1)
        for li,(l,dr) in enumerate(zip(s.layers,s.drops)):
            x=s.relu(l(x))
            if s.act_clip is not None: x=torch.clamp(x,max=s.act_clip)
            if s.neuron_mask is not None: x=x*s.neuron_mask[li]
            x=dr(x)
        return s.head(x)

@torch.no_grad()
def acc(m,X=None):
    m.eval(); X=EX if X is None else X; c=0
    for i in range(0,len(X),400): c+=(m(X[i:i+400]).argmax(1)==EY[i:i+400]).sum().item()
    return c/len(X)

def _aff(X,angle=0.,tx=0.,ty=0.,sc=1.):
    th=torch.tensor(angle*np.pi/180.); co,si=torch.cos(th)/sc,torch.sin(th)/sc
    M=torch.tensor([[co,-si,tx],[si,co,ty]],dtype=torch.float32).unsqueeze(0).repeat(len(X),1,1)
    return F.grid_sample(X,F.affine_grid(M,X.shape,align_corners=False),align_corners=False,padding_mode='zeros')

# ---- input-side (identical magnitudes to Cohort A) ----
def s_noise(X,e):   return torch.clamp(X+torch.randn_like(X)*(e*0.6),0,1)
def s_blur(X,e):
    n=int(round(e*20))
    if n<1: return X
    k=torch.ones(1,1,3,3)/9.; b=X.clone()
    for _ in range(n): b=F.conv2d(b,k,padding=1)
    return torch.clamp(b,0,1)
def s_occlusion(X,e):
    sd=int(round(e*16))
    if sd<1: return X
    Z=X.clone(); g=torch.Generator().manual_seed(0)
    for i in range(len(Z)):
        r=torch.randint(0,max(1,28-sd),(2,),generator=g); Z[i,:,r[0]:r[0]+sd,r[1]:r[1]+sd]=0
    return Z
def s_contrast(X,e):
    m=X.mean(dim=(1,2,3),keepdim=True); return torch.clamp((X-m)*(1-0.8*e)+m,0,1)
def s_rotate(X,e):    return X if e==0 else _aff(X,angle=e*30.)
def s_translate(X,e): return X if e==0 else _aff(X,tx=e*0.12,ty=e*0.072)
def s_scale(X,e):     return X if e==0 else _aff(X,sc=1-0.25*e)
INPUT_S=dict(noise=s_noise,blur=s_blur,occlusion=s_occlusion,contrast=s_contrast,
             rotate=s_rotate,translate=s_translate,scale=s_scale)

# ---- model-side: in-place perturb + restore (no deepcopy) ----
class Perturb:
    def __init__(s,m): s.m=m; s.saved=None
    def __enter__(s):
        s.saved=[p.data.clone() for p in s.m.parameters()]; return s.m
    def __exit__(s,*a):
        for p,q in zip(s.m.parameters(),s.saved): p.data.copy_(q)
        s.m.act_clip=None; s.m.neuron_mask=None

def _g(seed): return torch.Generator().manual_seed(seed)
def d_frac_weight(m,e,sd):
    for n_,p in m.named_parameters():
        if 'weight' in n_: p.data*=(torch.rand(p.shape,generator=_g(sd))>e*0.7).float()
def d_abs_weight(m,e,sd,keep=3000):
    k0=int(round(keep*(1-e)+200*e))
    for n_,p in m.named_parameters():
        if 'weight' in n_ and p.dim()==2:
            k=min(k0,p.numel())
            thr=p.data.abs().flatten().kthvalue(p.numel()-k+1).values
            p.data*=(p.data.abs()>=thr).float()
def d_frac_neuron(m,e,sd):
    m.neuron_mask=[(torch.rand(l.out_features,generator=_g(sd))>e*0.8).float() for l in m.layers]
def d_abs_neuron(m,e,sd,keep=24):
    k0=int(round(keep*(1-e)+2*e)); ms=[]
    for l in m.layers:
        w=l.out_features; k=min(k0,w)
        mk=torch.zeros(w); mk[torch.randperm(w,generator=_g(sd))[:k]]=1.; ms.append(mk)
    m.neuron_mask=ms
def d_mag_prune(m,e,sd):
    for n_,p in m.named_parameters():
        if 'weight' in n_ and p.dim()==2:
            k=int(p.numel()*e*0.9)
            if k>0:
                thr=p.data.abs().flatten().kthvalue(k).values
                p.data*=(p.data.abs()>thr).float()
def d_act_clip(m,e,sd): m.act_clip=float(3.0*(0.04**e))
MODEL_S=dict(weight_drop=d_frac_weight,neuron_ablate=d_frac_neuron,
             mag_prune=d_mag_prune,act_clip=d_act_clip,
             abs_weight=d_abs_weight,abs_neuron=d_abs_neuron)
STRESSORS=list(INPUT_S)+list(MODEL_S)

def curve(m,s,reps=2):
    if s in INPUT_S:
        f=INPUT_S[s]; a=[acc(m,f(EX,float(e))) for e in EPS]
    else:
        f=MODEL_S[s]; a=[]
        for e in EPS:
            if e==0: a.append(acc(m)); continue
            vs=[]
            for r in range(reps):
                with Perturb(m) as mm: f(mm,float(e),r); vs.append(acc(mm))
            a.append(float(np.mean(vs)))
    a=np.array(a); return (a[0]-a).tolist()

CELLS=[(w,d,dr) for w in (64,128,256) for d in (1,2) for dr in (0.1,0.3)]

def train_matched(w,d,dr,seed):
    """Train, snapshot at the epoch whose P0 is closest to target."""
    torch.manual_seed(seed)
    m=MLP(w,dr,d); opt=optim.Adam(m.parameters(),lr=1e-3)
    best=(9,None,0)
    for ep in range(1,MAXEP+1):
        m.train()
        for X,y in TL:
            opt.zero_grad(); nn.CrossEntropyLoss()(m(X),y).backward(); opt.step()
        p=acc(m)
        if abs(p-P0_TGT)<best[0]: best=(abs(p-P0_TGT),[q.data.clone() for q in m.parameters()],ep)
        if p>P0_HI+0.02: break
    for q,s in zip(m.parameters(),best[1]): q.data.copy_(s)
    return m,acc(m),best[2]

def run(c0,c1,seeds=5):
    path=OUT/'cohortB.json'
    recs=json.load(open(path)) if path.exists() else []
    done={(r['w'],r['d'],r['dr'],r['seed']) for r in recs}
    for w,d,dr in CELLS[c0:c1]:
        for sd in range(seeds):
            if (w,d,dr,sd) in done: continue
            m,p0,ep=train_matched(w,d,dr,1000*sd+w+d)
            ok=P0_LO<=p0<=P0_HI
            print(f'  w{w:>3} d{d} dr{dr} s{sd}  P0={p0:.4f} ep{ep} {"OK" if ok else "--"}',flush=True)
            if not ok: continue
            recs.append(dict(w=w,d=d,dr=dr,seed=sd,P0=float(p0),epochs=ep,
                             curves={s:curve(m,s) for s in STRESSORS}))
            json.dump(recs,open(path,'w'))
    print(f'cohort B n={len(recs)}')

if __name__=='__main__': run(int(sys.argv[1]),int(sys.argv[2]))
