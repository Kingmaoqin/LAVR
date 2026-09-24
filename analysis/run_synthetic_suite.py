"""Required synthetic A-F validation for the rescue probe families."""
import json
import sys
from pathlib import Path

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'analysis'))
import probe_suite as R


def split_docs(doc,seed=0):
    u=np.unique(doc); np.random.default_rng(seed).shuffle(u)
    a,b=int(.6*len(u)),int(.75*len(u))
    return tuple(np.where(np.isin(doc,x))[0] for x in (u[:a],u[a:b],u[b:]))


def ridge_pred(x,y,tr,va,te):
    m=R.fit_ridge(x[tr],y[tr],x[va],y[va]); return R.predict_sklearn(m,x[te])


def main():
    rng=np.random.default_rng(31); ndoc,nstate,ncand,d=180,3,6,16
    n=ndoc*nstate*ncand
    doc=np.repeat(np.arange(ndoc),nstate*ncand)
    state=np.repeat(np.arange(ndoc*nstate),ncand)
    hi=rng.normal(size=(n,d)).astype(np.float32)
    hg=np.repeat(rng.normal(size=(ndoc*nstate,d)),ncand,0).astype(np.float32)
    prev=rng.normal(size=(n,d)).astype(np.float32)
    ht=(prev+0.35*rng.normal(size=(n,d))).astype(np.float32)
    cheap=rng.normal(size=(n,6)).astype(np.float32)
    tr,va,te=split_docs(doc)
    cfg=R.SuiteConfig(seed=0,pca_dim=d,epochs=500,patience=60,batch_size=len(tr),
                      lr=3e-3,device='cpu')
    w=rng.normal(size=d)/np.sqrt(d); u=rng.normal(size=d)/np.sqrt(d); v=rng.normal(size=d)/np.sqrt(d)
    W=np.outer(u,v)
    state_scale=np.repeat(rng.lognormal(0,1.2,size=ndoc*nstate),ncand)
    state_offset=np.repeat(rng.normal(0,8,size=ndoc*nstate),ncand)
    noise=lambda sd:rng.normal(0,sd,n)
    targets={
      'A_linear':hi@w+noise(.25),
      'B_bilinear':np.einsum('ni,ij,nj->n',hi,W,hg)+noise(.2),
      'C_temporal':(ht-prev)@w+noise(.08),
      'D_ranking_only':state_offset+state_scale*(hi@w)+noise(.15),
      'E_nonlinear':(hi@u)**2+(hg@v)**2+noise(.1),
      'F_null':noise(1.0),
    }
    out={}
    for name,y in targets.items():
        y=y.astype(np.float32); row={}
        add=ridge_pred(np.c_[cheap,hi,hg],y,tr,va,te)
        row['additive_linear']=R.decision_metrics(y[te],add,state[te])
        if name=='A_linear':
            p=ridge_pred(hi,y,tr,va,te); row['linear']=R.decision_metrics(y[te],p,state[te])
        if name=='B_bilinear':
            p,hp=R.fit_torch_score('bilinear',cheap,hi,hg,y,state,tr,va,te,cfg,rank=4)
            row['bilinear']=R.decision_metrics(y[te],p,state[te]); row['bilinear_hp']=hp
            p,hp=R.fit_torch_score('relational',cheap,hi,hg,y,state,tr,va,te,cfg)
            row['relational_mlp']=R.decision_metrics(y[te],p,state[te])
            sh=hg.copy(); rng.shuffle(sh)
            p,_=R.fit_torch_score('bilinear',cheap,hi,sh,y,state,tr,va,te,cfg,rank=4)
            row['shuffled_hg']=R.decision_metrics(y[te],p,state[te])
        if name=='C_temporal':
            ps=ridge_pred(ht,y,tr,va,te); pd=ridge_pred(ht-prev,y,tr,va,te)
            row['static']=R.decision_metrics(y[te],ps,state[te]); row['delta']=R.decision_metrics(y[te],pd,state[te])
        if name=='D_ranking_only':
            pp,hp=R.pairwise_logistic(hi,y,state,tr,va,te)
            row['pairwise']=R.decision_metrics(y[te],pp,state[te])
            pl,_=R.fit_torch_score('relational',cheap,hi,hg,y,state,tr,va,te,cfg,objective='listwise')
            row['listwise']=R.decision_metrics(y[te],pl,state[te])
        if name=='E_nonlinear':
            rel=np.c_[hi,hg,hi-hg,np.abs(hi-hg),hi*hg]
            pk,_=R.kernel_probe(rel,y,tr,va,te,'poly2',0,1024)
            row['poly2']=R.decision_metrics(y[te],pk,state[te])
        if name=='F_null':
            p,_=R.fit_torch_score('bilinear',cheap,hi,hg,y,state,tr,va,te,cfg,rank=4)
            row['bilinear']=R.decision_metrics(y[te],p,state[te])
            pk,_=R.kernel_probe(np.c_[hi,hg],y,tr,va,te,'rbf',0,512)
            row['rbf']=R.decision_metrics(y[te],pk,state[te])
        out[name]=row

    checks={
      'A_linear_recovers':out['A_linear']['linear']['r2']>.7,
      'B_additive_fails':out['B_bilinear']['additive_linear']['r2']<.15,
      'B_bilinear_recovers':out['B_bilinear']['bilinear']['r2']>.5,
      'B_relational_recovers':out['B_bilinear']['relational_mlp']['r2']>.35,
      'B_shuffle_fails':out['B_bilinear']['shuffled_hg']['r2']<.15,
      'C_delta_beats_static':out['C_temporal']['delta']['r2']>out['C_temporal']['static']['r2']+.3,
      'D_pairwise_recovers':out['D_ranking_only']['pairwise']['pairwise_concordance']>.8,
      'E_kernel_beats_linear':out['E_nonlinear']['poly2']['r2']>out['E_nonlinear']['additive_linear']['r2']+.1,
      'F_linear_null':abs(out['F_null']['additive_linear']['r2'])<.1,
      'F_bilinear_null':out['F_null']['bilinear']['r2']<.1,
    }
    result={'status':'VALIDATED' if all(checks.values()) else 'FAILED','checks':checks,'results':out}
    dst=ROOT/'analysis/results/synthetic/synthetic_suite.json'; dst.parent.mkdir(parents=True,exist_ok=True)
    dst.write_text(json.dumps(result,indent=2,default=float)); print(json.dumps({'status':result['status'],'checks':checks},indent=2))
    if not all(checks.values()): raise SystemExit(1)


if __name__=='__main__': main()
