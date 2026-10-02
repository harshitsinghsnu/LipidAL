"""Exact GP with analytic marginal-likelihood gradients for the full paper grid.

The lipid kernel is one of the paper's five choices. For pooled protein-lipid
data it is multiplied by a learned Matern-3/2 protein kernel; on a single target
the protein factor equals one. This product factor is a domain adaptation.
"""
import numpy as np
from scipy.linalg import cho_factor, cho_solve, cholesky
from scipy.optimize import minimize
from sklearn.preprocessing import StandardScaler

KERNELS=('tanimoto','linear','rbf','rq','matern32')
PROTOCOLS=('random','ucb_balanced','ucb_alternate','ucb_sandwich',
           'ucb_explore_heavy','ucb_exploit_heavy','ucb_gradual','thompson')
SCHEDULES={
 'random':'R'*10,'ucb_balanced':'B'*10,'ucb_alternate':'EX'*5,
 'ucb_sandwich':'EE'+'X'*6+'EE','ucb_explore_heavy':'E'*7+'X'*3,
 'ucb_exploit_heavy':'X'*7+'E'*3,'ucb_gradual':'EEE'+'B'*4+'XXX',
 'thompson':'T'*10,
}

def sqdist(x):
    norm=np.einsum('ij,ij->i',x,x)
    d=np.maximum(norm[:,None]+norm[None,:]-2*x@x.T,0)
    np.fill_diagonal(d,0)
    return d

class Geometry:
    def __init__(self,lipid,protein,initial):
        # Scaling fitted only to initial labelled inputs; frozen for this campaign.
        x=np.asarray(lipid,float)
        dot=x@x.T;norm=np.diag(dot)
        self.tanimoto=dot/np.maximum(norm[:,None]+norm[None,:]-dot,1e-12)
        both_zero=(norm[:,None]+norm[None,:])<1e-12
        self.tanimoto[both_zero]=1.0
        self.linear=dot/max(float(np.mean(norm[initial])),1e-12)
        d=sqdist(x);ref=d[np.ix_(initial,initial)];positive=ref[ref>1e-12]
        self.lipid_d2=d/(float(np.median(positive)) if len(positive) else 1.0)
        p=StandardScaler().fit(protein[initial]).transform(protein)
        pd=sqdist(p);ref=pd[np.ix_(initial,initial)];positive=ref[ref>1e-12]
        self.protein_d2=pd/(float(np.median(positive)) if len(positive) else 1.0)
        self.multi_target=bool(np.max(pd)>1e-12)

class ExactPairGP:
    def __init__(self,geometry,kernel,maxiter=60):
        if kernel not in KERNELS:raise ValueError(kernel)
        self.g=geometry;self.kernel=kernel;self.maxiter=maxiter
        self.names=['log_signal','log_noise','mean']
        self.bounds=[(-7,5),(-11,2),(-5,5)]
        self.theta=np.array([0.,np.log(.05),0.])
        if kernel in ('rbf','rq','matern32'):
            self.names+=['log_lipid_length'];self.bounds+=[(-5,5)];self.theta=np.r_[self.theta,0.]
        if kernel=='rq':
            self.names+=['log_rq_alpha'];self.bounds+=[(-5,5)];self.theta=np.r_[self.theta,0.]
        if geometry.multi_target:
            self.names+=['log_protein_length'];self.bounds+=[(-5,5)];self.theta=np.r_[self.theta,0.]

    def cov(self,a,b,theta=None,derivatives=False):
        theta=self.theta if theta is None else theta
        par=dict(zip(self.names,theta));ix=np.ix_(a,b)
        ds={};signal=np.exp(par['log_signal'])
        if self.kernel=='tanimoto':kl=self.g.tanimoto[ix]
        elif self.kernel=='linear':kl=self.g.linear[ix]
        else:
            d2=self.g.lipid_d2[ix]/np.exp(2*par['log_lipid_length'])
            if self.kernel=='rbf':
                kl=np.exp(-.5*d2);ds['log_lipid_length']=kl*d2
            elif self.kernel=='matern32':
                r=np.sqrt(3*d2);e=np.exp(-r);kl=(1+r)*e
                ds['log_lipid_length']=r*r*e
            else:
                alpha=np.exp(par['log_rq_alpha']);t=d2/(2*alpha)
                kl=np.exp(-alpha*np.log1p(t))
                ds['log_lipid_length']=kl*2*alpha*t/(1+t)
                ds['log_rq_alpha']=kl*alpha*(-np.log1p(t)+t/(1+t))
        if self.g.multi_target:
            r=np.sqrt(3*self.g.protein_d2[ix])/np.exp(par['log_protein_length'])
            e=np.exp(-r);kp=(1+r)*e;dp=r*r*e
        else:kp=1.
        k=signal*kl*kp
        if not derivatives:return k
        deriv={n:signal*v*kp for n,v in ds.items()}
        deriv['log_signal']=k
        if self.g.multi_target:deriv['log_protein_length']=signal*kl*dp
        return k,deriv

    def objective(self,theta):
        k,deriv=self.cov(self.train,self.train,theta,True)
        par=dict(zip(self.names,theta));noise=np.exp(par['log_noise']);n=len(self.train)
        k=k+(noise+1e-8)*np.eye(n)
        cf=cho_factor(k,lower=True,check_finite=False)
        residual=self.y-par['mean'];alpha=cho_solve(cf,residual,check_finite=False)
        loss=.5*residual@alpha+np.log(np.diag(cf[0])).sum()+n*.5*np.log(2*np.pi)
        w=cho_solve(cf,np.eye(n),check_finite=False)-np.outer(alpha,alpha)
        grad=[]
        for name in self.names:
            if name=='mean':grad.append(-alpha.sum())
            elif name=='log_noise':grad.append(.5*noise*np.trace(w))
            else:grad.append(.5*np.sum(w*deriv[name]))
        return float(loss),np.asarray(grad)

    def fit(self,indices,y):
        self.train=np.asarray(indices)
        values=np.asarray(y,float)
        self.center=float(np.mean(values));self.scale=max(float(np.std(values)),1e-6)
        self.y=(values-self.center)/self.scale
        before=self.objective(self.theta)[0]
        result=minimize(self.objective,self.theta,jac=True,method='L-BFGS-B',bounds=self.bounds,
                        options=dict(maxiter=self.maxiter,ftol=1e-8,gtol=1e-5,maxls=30))
        if np.isfinite(result.fun) and result.fun<=before+1e-7:self.theta=result.x
        par=dict(zip(self.names,self.theta));self.mean=par['mean'];self.noise=np.exp(par['log_noise'])
        k=self.cov(self.train,self.train)+(self.noise+1e-8)*np.eye(len(self.train))
        self.cf=cho_factor(k,lower=True,check_finite=False)
        self.alpha=cho_solve(self.cf,self.y-self.mean,check_finite=False)
        self.diagnostics=dict(nll=self.objective(self.theta)[0],initial_nll=before,
            success=bool(result.success),message=str(result.message),iterations=int(result.nit),
            params=par)
        return self

    def predict(self,indices,joint=False):
        idx=np.asarray(indices);cross=self.cov(idx,self.train)
        mean=(self.mean+cross@self.alpha)*self.scale+self.center
        solved=cho_solve(self.cf,cross.T,check_finite=False)
        if joint:
            cov=self.cov(idx,idx)-cross@solved;cov=(cov+cov.T)*.5*self.scale**2
            return mean,cov
        # Self covariance for normalized stationary kernels is signal; linear varies.
        diag=np.exp(self.theta[0])* (self.g.linear.diagonal()[idx] if self.kernel=='linear' else np.ones(len(idx)))
        latent=np.maximum(diag-np.einsum('ij,ji->i',cross,solved),1e-12)*self.scale**2
        return mean,np.sqrt(latent),np.sqrt(latent+self.noise*self.scale**2)

    def thompson(self,indices,rng):
        mean,cov=self.predict(indices,joint=True)
        jitter=max(float(np.max(np.diag(cov))),1.)*1e-9
        for _ in range(8):
            try:return mean+cholesky(cov+jitter*np.eye(len(mean)),lower=True)@rng.standard_normal(len(mean))
            except np.linalg.LinAlgError:jitter*=10
        raise np.linalg.LinAlgError('Posterior covariance failed jittered Cholesky')

def select(model,pool,phase,batch,rng):
    if batch==0:return np.array([],dtype=int)
    if phase=='R':return rng.choice(pool,batch,replace=False)
    if phase=='T':score=model.thompson(pool,rng)
    else:
        mean,latent_sd,obs_sd=model.predict(pool)
        # Homoskedastic predictive observation uncertainty matches canonical author code.
        if phase=='E':score=obs_sd
        elif phase=='X':score=mean
        elif phase=='B':score=.5*mean+.5*obs_sd
        else:raise ValueError(phase)
    return np.asarray(pool)[np.argsort(-score,kind='stable')[:batch]]
