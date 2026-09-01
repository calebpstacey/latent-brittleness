"""
Cross-stress latent brittleness — calibrated.
All 11 stressors tuned so mean degradation at eps=1.0 lands near 0.40 (no floor saturation),
making B-scores comparable across stress types.
"""
import json, copy, sys, numpy as np, torch, torch.nn as nn, torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from pathlib import Path

DEV='cpu'; EPOCHS=3; TRAIN_N=5000; EVAL_N=1000
P0_LOW,P0_HIGH=0.82,0.93
ROOT=Path('/home/claude/lb3'); OUT=ROOT/'results'; OUT.mkdir(exist_ok=True)

tfm=transforms.Compose([transforms.ToTensor()])
_tr=datasets.MNIST(ROOT/'data',train=True,download=False,transform=tfm)
_te=datasets.MNIST(ROOT/'data',train=False,download=False,transform=tfm)
TL=DataLoader(Subset(_tr,range(TRAIN_N)),batch_size=256,shuffle=True)
_EL=DataLoader(Subset(_te,range(EVAL_N)),batch_size=250)
EVAL_X=torch.cat([x for x,_ in _EL]); EVAL_Y=torch.cat([y for _,y in _EL])

class MLP(nn.Module):
    def __init__(s,w,dr,d):
        super().__init__()
        s.layers=nn.ModuleList(); s.drops=nn.ModuleList()
        dims=[784]+[w]*d
        for i in range(d):
            s.layers.append(nn.Linear(dims[i],dims[i+1])); s.drops.append(nn.Dropout(dr))
        s.head=nn.Linear(w,10); s.relu=nn.ReLU()
        s.act_clip=None; s.neuron_mask=None
    def forward(s,x):
        x=x.view(x.size(0),-1)
        for li,(l,dr) in enumerate(zip(s.layers,s.drops)):
            x=s.relu(l(x))
            if s.act_clip is not None: x=torch.clamp(x,max=s.act_clip)
            if s.neuron_mask is not None: x=x*s.neuron_mask[li]
            x=dr(x)
        return s.head(x)

@torch.no_grad()
def acc_on(model,X,Y):
    model.eval(); c=0
    for i in range(0,len(X),250):
        c+=(model(X[i:i+250]).argmax(1)==Y[i:i+250]).sum().item()
    return c/len(X)

def _affine(X,angle=0.,tx=0.,ty=0.,scale=1.):
    th=torch.tensor(angle*np.pi/180.)
    cos,sin=torch.cos(th)/scale,torch.sin(th)/scale
    M=torch.tensor([[cos,-sin,tx],[sin,cos,ty]],dtype=torch.float32).unsqueeze(0).repeat(len(X),1,1)
    return F.grid_sample(X,F.affine_grid(M,X.shape,align_corners=False),
                         align_corners=False,padding_mode='zeros')

# ---- input-side stressors: f(X, e) with e in [0,1] ----
def s_noise(X,e):    return torch.clamp(X+torch.randn_like(X)*(e*0.6),0,1)
def s_blur(X,e):
    n=int(round(e*20))
    if n<1: return X
    k=torch.ones(1,1,3,3)/9.; b=X.clone()
    for _ in range(n): b=F.conv2d(b,k,padding=1)
    return torch.clamp(b,0,1)
def s_occlusion(X,e):
    side=int(round(e*16))
    if side<1: return X
    Z=X.clone(); g=torch.Generator().manual_seed(0)
    for i in range(len(Z)):
        r=torch.randint(0,max(1,28-side),(2,),generator=g)
        Z[i,:,r[0]:r[0]+side,r[1]:r[1]+side]=0
    return Z
def s_contrast(X,e):
    m=X.mean(dim=(1,2,3),keepdim=True); return torch.clamp((X-m)*(1-0.8*e)+m,0,1)
def s_rotate(X,e):    return X if e==0 else _affine(X,angle=e*30.)
def s_translate(X,e): return X if e==0 else _affine(X,tx=e*0.12,ty=e*0.072)
def s_scale(X,e):     return X if e==0 else _affine(X,scale=1-0.25*e)

# ---- model-side stressors: f(model, e, seed) ----
@torch.no_grad()
def m_weight_drop(m,e,seed=0):
    q=copy.deepcopy(m)
    if e==0: return q
    g=torch.Generator().manual_seed(seed)
    for n,p in q.named_parameters():
        if 'weight' in n: p.data*=(torch.rand(p.shape,generator=g)>e*0.7).float()
    return q
@torch.no_grad()
def m_neuron_ablate(m,e,seed=0):
    q=copy.deepcopy(m)
    if e==0: return q
    g=torch.Generator().manual_seed(seed)
    q.neuron_mask=[(torch.rand(l.out_features,generator=g)>e*0.8).float() for l in q.layers]
    return q
@torch.no_grad()
def m_mag_prune(m,e,seed=0):
    q=copy.deepcopy(m)
    if e==0: return q
    for n,p in q.named_parameters():
        if 'weight' in n and p.dim()==2:
            k=int(p.numel()*e*0.9)
            if k>0:
                thr=p.data.abs().flatten().kthvalue(k).values
                p.data*=(p.data.abs()>thr).float()
    return q
@torch.no_grad()
def m_act_clip(m,e,seed=0):
    q=copy.deepcopy(m)
    if e==0: return q
    q.act_clip=float(3.0*(0.04**e))     # 3.0 -> 0.12
    return q

INPUT_S={'noise':s_noise,'blur':s_blur,'occlusion':s_occlusion,'contrast':s_contrast,
         'rotate':s_rotate,'translate':s_translate,'scale':s_scale}
MODEL_S={'weight_drop':m_weight_drop,'neuron_ablate':m_neuron_ablate,
         'mag_prune':m_mag_prune,'act_clip':m_act_clip}
FAMILY={'noise':'input','blur':'input','occlusion':'input','contrast':'input',
        'rotate':'shift','translate':'shift','scale':'shift',
        'weight_drop':'internal','neuron_ablate':'internal',
        'mag_prune':'resource','act_clip':'resource'}
STRESSORS=list(INPUT_S)+list(MODEL_S)
EPS=np.array([0.0,0.33,0.67,1.0])

def curve_features(D,eps=EPS):
    h1,h2=eps[1]-eps[0],eps[2]-eps[1]
    return dict(susceptibility=float((D[1]-D[0])/h1),
                curvature=float(2*(D[2]/h2-D[1]*(1/h1+1/h2)+D[0]/h1)/(h1+h2)),
                area=float(np.trapezoid(D,eps)),
                late_slope=float((D[-1]-D[-2])/(eps[-1]-eps[-2])))

def degradation_curve(model,s):
    if s in INPUT_S:
        f=INPUT_S[s]; a=[acc_on(model,f(EVAL_X,float(e)),EVAL_Y) for e in EPS]
    else:
        f=MODEL_S[s]; a=[]
        for e in EPS:
            reps=1 if e==0 else 3
            a.append(float(np.mean([acc_on(f(model,float(e),seed=k),EVAL_X,EVAL_Y) for k in range(reps)])))
    a=np.array(a); return a[0]-a

def train_one(cfg):
    torch.manual_seed(cfg['seed'])
    m=MLP(cfg['w'],cfg['dr'],cfg['d'])
    opt=optim.Adam(m.parameters(),lr=cfg['lr'],weight_decay=cfg['wd'])
    for _ in range(EPOCHS):
        m.train()
        for X,y in TL:
            opt.zero_grad(); nn.CrossEntropyLoss()(m(X),y).backward(); opt.step()
    return m

def make_configs(n,seed=11):
    rng=np.random.default_rng(seed)
    return [dict(seed=int(rng.integers(0,99999)),w=int(rng.choice([32,64,128,256])),
                 dr=float(rng.choice([0.0,0.1,0.3,0.5])),d=int(rng.choice([1,2,3])),
                 lr=float(rng.choice([5e-4,1e-3,2e-3])),wd=float(rng.choice([0.0,1e-4,1e-3])))
            for _ in range(n)]

def run(lo,hi,n_total=120):
    cfgs=make_configs(n_total)
    path=OUT/'cross2.json'
    recs=json.load(open(path)) if path.exists() else []
    for i in range(lo,min(hi,n_total)):
        m=train_one(cfgs[i]); P0=acc_on(m,EVAL_X,EVAL_Y)
        if not (P0_LOW<=P0<=P0_HIGH):
            print(f'[{i+1:03d}] P0={P0:.4f} --',flush=True); continue
        curves={s:degradation_curve(m,s).tolist() for s in STRESSORS}
        recs.append(dict(idx=i,P0=float(P0),cfg=cfgs[i],curves=curves,
                         feats={s:curve_features(np.array(c)) for s,c in curves.items()}))
        json.dump(recs,open(path,'w'))
        print(f'[{i+1:03d}] P0={P0:.4f} OK  (n={len(recs)})',flush=True)
    print(f'survivors: {len(recs)}')

if __name__=='__main__':
    run(int(sys.argv[1]),int(sys.argv[2]))
