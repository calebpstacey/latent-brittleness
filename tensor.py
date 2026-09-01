import json,numpy as np
from scipy import stats
R=json.load(open('results/cross2.json')); A=json.load(open('results/analysis.json'))
S=A['stressors']; DIMS=['susceptibility','curvature','area','late_slope']
n=len(R); ns=len(S)
FAM={'noise':'input','blur':'input','occlusion':'input','contrast':'input','rotate':'shift',
     'translate':'shift','scale':'shift','weight_drop':'internal','neuron_ablate':'internal',
     'mag_prune':'resource','act_clip':'resource'}
# tensor model x stressor x metric  (z-scored within stressor x metric)
T=np.zeros((n,ns,4))
for i,r in enumerate(R):
    for j,s in enumerate(S):
        for k,d in enumerate(DIMS): T[i,j,k]=r['feats'][s][d]
Tz=(T-T.mean(0))/ (T.std(0)+1e-9)

print('='*70); print('TENSOR: is "3 kinds of brittleness" really "3 kinds of COLLAPSE GEOMETRY"?'); print('='*70)
print('\nPer-metric: mean within- vs between-family rank correlation across stressors')
print(f'{"metric":<16}{"within-fam":>12}{"between-fam":>13}{"gap":>8}{"PC1 var":>10}')
for k,d in enumerate(DIMS):
    M=Tz[:,:,k]; C=np.zeros((ns,ns))
    for a in range(ns):
        for b in range(ns): C[a,b]=stats.spearmanr(M[:,a],M[:,b]).statistic
    wi=[C[a,b] for a in range(ns) for b in range(a+1,ns) if FAM[S[a]]==FAM[S[b]]]
    bw=[C[a,b] for a in range(ns) for b in range(a+1,ns) if FAM[S[a]]!=FAM[S[b]]]
    ev=np.sort(np.linalg.eigvalsh(np.corrcoef(M.T)))[::-1]
    print(f'  {d:<14}{np.mean(wi):>12.3f}{np.mean(bw):>13.3f}{np.mean(wi)-np.mean(bw):>8.3f}{100*ev[0]/ev.sum():>9.0f}%')

print('\nWhich metric separates families best? -> that is the axis the factors live on')
print('\nPer-family: which collapse metric is most consistent across its stressors?')
fams=sorted(set(FAM.values()))
print(f'{"family":<12}'+''.join(f'{d[:9]:>11}' for d in DIMS))
for f in fams:
    js=[j for j,s in enumerate(S) if FAM[s]==f]
    if len(js)<2: continue
    row=f'  {f:<10}'
    for k in range(4):
        cs=[stats.spearmanr(Tz[:,a,k],Tz[:,b,k]).statistic for ii,a in enumerate(js) for b in js[ii+1:]]
        row+=f'{np.mean(cs):>11.2f}'
    print(row)
