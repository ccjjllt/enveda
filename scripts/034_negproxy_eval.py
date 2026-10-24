import json,numpy as np
T=json.load(open('truth.json'))
import sys
P=sys.argv[1]
for x in ['negproxy','posctrl']:
    o=json.load(open(f'{P}_{x}.json')); rr=[];rnd=[]
    for k,s in o.items():
        v={a:b for a,b in s.items() if b is not None}; t=v[T[k]]
        r=1+sum(b>t for a,b in v.items() if a!=T[k]); rr.append(1/r); rnd.append(np.mean(1/np.arange(1,len(v)+1)))
    print(x,'MRR %.3f top1 %d random %.3f n %d'%(np.mean(rr),sum(np.array(rr)==1),np.mean(rnd),len(rr)))
