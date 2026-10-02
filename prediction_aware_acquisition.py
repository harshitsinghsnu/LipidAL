"""Prediction-aware Thompson via batch-conditional integrated variance reduction.

An empirical combination of established ingredients, not a novelty claim.
No proxy labels or outer-test inputs/labels enter this selector.
"""
import numpy as np
from scipy.linalg import cholesky
from scipy.stats import rankdata
from grid_gp import select as baseline_select

POLICIES=('random','uncertainty','ucb_balanced','thompson','prediction_aware_thompson','variance_reduction')
LAMBDA=.5

def rank01(values):
    values=np.asarray(values)
    return (rankdata(values,method='average')-1)/max(len(values)-1,1)

def variance_gain(covariance,n_pool,noise):
    cross=covariance[n_pool:,:n_pool]
    return np.mean(cross**2,axis=0)/np.maximum(np.diag(covariance)[:n_pool]+noise,1e-12)

def select(model,pool,proxy,policy,batch,rng):
    pool=np.asarray(pool,dtype=int);proxy=np.asarray(proxy,dtype=int)
    if set(pool)&set(proxy) or set(model.train)&set(proxy):raise ValueError('Proxy must be reserved from candidate/training rows')
    if not len(proxy) or not 0<=batch<=len(pool):raise ValueError('Invalid pool, proxy or batch')
    if policy in ('random','uncertainty','ucb_balanced','thompson'):
        phase={'random':'R','uncertainty':'E','ucb_balanced':'B','thompson':'T'}[policy]
        return baseline_select(model,pool,phase,batch,rng),[]
    if policy not in POLICIES:raise ValueError(policy)
    weight=LAMBDA if policy=='prediction_aware_thompson' else 1.
    indices=np.r_[pool,proxy];mean,cov=model.predict(indices,joint=True);n=len(pool)
    noise=max(model.noise*model.scale**2,1e-12)
    draw=np.zeros(n)
    if weight<1:
        # Same latent, correlated draw convention as the existing Thompson baseline.
        block=cov[:n,:n];jitter=max(float(np.diag(block).max()),1.)*1e-9
        for attempt in range(8):
            try:
                draw=mean[:n]+cholesky(block+jitter*np.eye(n),lower=True)@rng.standard_normal(n);break
            except np.linalg.LinAlgError:jitter*=10
        else:raise np.linalg.LinAlgError('Thompson covariance Cholesky failed')
    remaining=np.ones(n,dtype=bool);chosen=[];audit=[]
    for step in range(batch):
        gain=variance_gain(cov,n,noise);available=np.flatnonzero(remaining)
        score=(1-weight)*rank01(draw[available])+weight*rank01(gain[available])
        q=int(available[np.argmax(score)]);chosen.append(q);remaining[q]=False
        before=float(np.diag(cov)[n:].mean())
        col=cov[:,q].copy();cov-=np.outer(col,col)/max(cov[q,q]+noise,1e-12);cov=(cov+cov.T)*.5
        after=float(np.diag(cov)[n:].mean())
        if after>before+1e-8:raise AssertionError('Conditioning increased proxy variance')
        audit.append(dict(row_index=int(pool[q]),sampled_affinity=float(draw[q]) if weight<1 else None,
                          predicted_proxy_variance_reduction=float(gain[q]),actual_conditional_reduction=before-after,
                          proxy_variance_before=before,proxy_variance_after=after,weight=weight))
    return pool[chosen],audit
