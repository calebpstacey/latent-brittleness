"""
Confound test: fractional vs absolute capacity damage.

If you remove 70% of weights, a width-256 net keeps 77 units and a width-32 net keeps 10.
The width<->capacity-robustness link may be definitional rather than structural.

Test: apply damage as (a) fixed FRACTION and (b) fixed ABSOLUTE count.
If width predicts robustness under BOTH, the finding is real.
"""
import json,copy,sys,numpy as np,torch,torch.nn as nn
from scipy import stats
from stress2 import (MLP,train_one,make_configs,acc_on,EVAL_X,EVAL_Y,P0_LOW,P0_HIGH,OUT)

EPS=np.array([0.,0.33,0.67,1.])

@torch.no_grad()
def frac_neuron(m,e,seed):
    q=copy.deepcopy(m)
    if e==0: return q
    g=torch.Generator().manual_seed(seed)
    q.neuron_mask=[(torch.rand(l.out_features,generator=g)>e*0.8).float() for l in q.layers]
    return q

@torch.no_grad()
def abs_neuron(m,e,seed,keep_max=24):
    """Keep a fixed ABSOLUTE number of neurons: keep_max -> 2, independent of width."""
    q=copy.deepcopy(m)
    if e==0: return q
    g=torch.Generator().manual_seed(seed); masks=[]
    for l in q.layers:
        w=l.out_features
        keep=int(round(keep_max*(1-e)+2*e))       # keep_max at e=0 -> 2 at e=1
        keep=min(keep,w)
        idx=torch.randperm(w,generator=g)[:keep]
        mk=torch.zeros(w); mk[idx]=1.; masks.append(mk)
    q.neuron_mask=masks
    return q

@torch.no_grad()
def frac_weight(m,e,seed):
    q=copy.deepcopy(m)
    if e==0: return q
    g=torch.Generator().manual_seed(seed)
    for n_,p in q.named_parameters():
        if 'weight' in n_: p.data*=(torch.rand(p.shape,generator=g)>e*0.7).float()
    return q

@torch.no_grad()
def abs_weight(m,e,seed,keep_n=3000):
    """Keep a fixed ABSOLUTE number of weights (largest-magnitude), independent of width."""
    q=copy.deepcopy(m)
    if e==0: return q
    keep=int(round(keep_n*(1-e)+200*e))
    for n_,p in q.named_parameters():
        if 'weight' in n_ and p.dim()==2:
            k=min(keep,p.numel())
            thr=p.data.abs().flatten().kthvalue(p.numel()-k+1).values
            p.data*=(p.data.abs()>=thr).float()
    return q

VARIANTS={'frac_neuron':frac_neuron,'abs_neuron':abs_neuron,
          'frac_weight':frac_weight,'abs_weight':abs_weight}

def curve(m,f):
    a=[]
    for e in EPS:
        reps=1 if e==0 else 3
        a.append(float(np.mean([acc_on(f(m,float(e),s),EVAL_X,EVAL_Y) for s in range(reps)])))
    a=np.array(a); return (a[0]-a).tolist()

def run(lo,hi,N=120):
    cfgs=make_configs(N); path=OUT/'confound.json'
    recs=json.load(open(path)) if path.exists() else []
    for i in range(lo,min(hi,N)):
        m=train_one(cfgs[i]); P0=acc_on(m,EVAL_X,EVAL_Y)
        if not(P0_LOW<=P0<=P0_HIGH): continue
        recs.append(dict(idx=i,P0=float(P0),cfg=cfgs[i],
                         curves={k:curve(m,f) for k,f in VARIANTS.items()}))
        json.dump(recs,open(path,'w'))
        print(f'[{i+1:03d}] P0={P0:.3f} w={cfgs[i]["w"]:>3} (n={len(recs)})',flush=True)

if __name__=='__main__': run(int(sys.argv[1]),int(sys.argv[2]))
