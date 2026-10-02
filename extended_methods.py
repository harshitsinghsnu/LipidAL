"""Additional policies; original paper engine and schedules remain unchanged."""
import numpy as np
from scipy.special import ndtr
from grid_gp import KERNELS, PROTOCOLS as ORIGINAL_PROTOCOLS, SCHEDULES as ORIGINAL_SCHEDULES, select as original_select

STATIC_PHASES = dict(random='R', thompson='T', ucb_balanced='B', greedy='X',
                     uncertainty='E', ei='I', pi='P', thompson_diverse='D')
PROTOCOLS = ORIGINAL_PROTOCOLS + ('greedy', 'uncertainty', 'ei', 'pi', 'thompson_diverse')
SCHEDULES = {**ORIGINAL_SCHEDULES, **{k:v*10 for k,v in STATIC_PHASES.items()}}
XI = 0.01  # improvement threshold in pK units, not tuned against test labels
DIVERSITY_WEIGHT = 0.25

def improvement_scores(mean, sd, incumbent):
    improvement = np.asarray(mean)-incumbent-XI
    sd = np.asarray(sd)
    z = improvement/np.maximum(sd, 1e-15)
    pi = np.where(sd>1e-15, ndtr(z), (improvement>0).astype(float))
    ei = np.where(sd>1e-15, improvement*ndtr(z)+sd*np.exp(-.5*z*z)/np.sqrt(2*np.pi), np.maximum(improvement,0))
    return np.maximum(ei,0), pi

def select(model, pool, phase, batch, rng, incumbent=None):
    pool = np.asarray(pool, dtype=int)
    if not 0 <= batch <= len(pool): raise ValueError('Invalid batch size')
    if batch == 0: return pool[:0]
    if phase not in ('I','P','D'): return original_select(model,pool,phase,batch,rng)
    if phase in ('I','P'):
        if incumbent is None or not np.isfinite(incumbent): raise ValueError('EI/PI require the best observed training pK')
        mean,sd,_ = model.predict(pool)
        ei,pi = improvement_scores(mean,sd,incumbent)
        return pool[np.argsort(-(ei if phase=='I' else pi),kind='stable')[:batch]]
    # One joint posterior draw + greedy diversity bonus. This is an explicit
    # heuristic, not independent marginal TS, qEI, or an information-gain rule.
    draw = model.thompson(pool,rng)
    score = (draw-draw.mean())/max(float(draw.std()),1e-12)
    distance = model.g.lipid_d2[np.ix_(pool,pool)]
    chosen=[];remaining=np.ones(len(pool),dtype=bool)
    for _ in range(batch):
        bonus=np.zeros(len(pool)) if not chosen else 1-np.exp(-.5*distance[:,chosen].min(axis=1))
        candidate=int(np.argmax(np.where(remaining,score+DIVERSITY_WEIGHT*bonus,-np.inf)))
        chosen.append(candidate);remaining[candidate]=False
    return pool[chosen]
