'Exact-distribution categorical rounding acceleration; no Gaussian approximation.'
import numpy as np
from scipy import sparse
from paper_suite.algorithms.sample_sdp_core import stochastic_round

def prepare(X,y,clip,precision,rng,numeric_dim,low,high,chunk=256):
    N,d=X.shape;r=int(numeric_dim);q=d-r
    if y.shape!=(N,) or not 0<=r<=d or clip<=0 or precision<1:raise ValueError('invalid statistics request')
    low,high=np.asarray(low,float),np.asarray(high,float)
    if low.shape!=(q,) or high.shape!=(q,) or np.any(high<=low):raise ValueError('invalid public binary schema')
    ui,uj=np.triu_indices(d);numeric=np.flatnonzero((ui<r)&(uj<r));catpair=np.flatnonzero(ui>=r)
    mixed_pos=np.r_[numeric,np.arange(len(ui),len(ui)+r)]
    other_pos=np.setdiff1d(np.arange(len(ui)+d),mixed_pos)
    upper_other=other_pos[other_pos<len(ui)];xy_other=other_pos[other_pos>=len(ui)]-len(ui)
    sums=np.zeros(len(ui)+d,dtype=np.int64)
    binary_parts=[];ordinary_ids=[];unclipped=0;clipped=0;norm_sample=[]
    for start in range(0,N,chunk):
        xb=np.asarray(X[start:start+chunk],float);yb=np.asarray(y[start:start+chunk],float)
        if not np.isfinite(xb).all() or not np.isfinite(yb).all():raise ValueError('finite data required')
        sq=xb*xb;s2=sq.sum(1)
        norms=np.sqrt((s2*s2+(sq*sq).sum(1))/2+s2*yb*yb)
        factor=np.minimum(1.,clip/np.maximum(norms,1e-12));ordinary=factor==1
        if len(norm_sample)<5000:norm_sample.extend(norms[:5000-len(norm_sample)].tolist())
        z=np.concatenate([xb[:,ui[numeric]]*xb[:,uj[numeric]],xb[:,:r]*yb[:,None]],axis=1)
        z*=factor[:,None]*precision
        sums[mixed_pos]+=stochastic_round(z,rng).sum(0,dtype=np.int64)
        cat=xb[:,r:]
        bits=cat>(low+high)[None,:]/2
        expected=np.where(bits,high[None,:],low[None,:])
        if not np.allclose(cat,expected,rtol=0.,atol=1e-12):raise ValueError('categorical schema mismatch: refuse approximate grouped rounding')
        binary_parts.append(sparse.csr_matrix(bits[ordinary],dtype=np.int64))
        ordinary_ids.extend((start+np.flatnonzero(ordinary)).tolist())
        unclipped+=int(ordinary.sum());clipped+=int((~ordinary).sum())
        bad=np.flatnonzero(~ordinary)
        for i in range(0,len(bad),16):
            ix=bad[i:i+16]
            z=np.concatenate([xb[ix][:,ui[upper_other]]*xb[ix][:,uj[upper_other]],xb[ix][:,xy_other]*yb[ix,None]],axis=1)
            z*=factor[ix,None]*precision
            sums[other_pos]+=stochastic_round(z,rng).sum(0,dtype=np.int64)
    B=sparse.vstack(binary_parts,format='csr');del binary_parts
    counts=np.asarray(B.sum(0)).ravel();cooc=(B.T@B).toarray()
    ordinary_ids=np.asarray(ordinary_ids,dtype=np.int64)


    for column in range(r+1):
        scalar=np.asarray(X[ordinary_ids,column],float) if column<r else np.asarray(y[ordinary_ids],float)
        values,inverse,tally=np.unique(scalar,return_inverse=True,return_counts=True)
        A=sparse.csr_matrix((np.ones(unclipped,np.int64),(np.arange(unclipped),inverse)),shape=(unclipped,len(values)))
        active=(A.T@B).tocsr()
        positions=np.flatnonzero((ui==column)&(uj>=r)) if column<r else len(ui)+np.arange(r,d)
        for start in range(0,len(values),128):
            end=min(start+128,len(values));c=active[start:end].toarray()
            for counts0,base in ((tally[start:end,None]-c,low),(c,high)):
                z=precision*values[start:end,None]*base[None,:];floors=np.floor(z)
                sums[positions]+=(counts0*floors.astype(np.int64)+rng.binomial(counts0,z-floors)).sum(0,dtype=np.int64)
    ii,jj=ui[catpair]-r,uj[catpair]-r
    both=cooc[ii,jj];a=counts[ii];b=counts[jj]
    groups=[(unclipped-a-b+both,low[ii]*low[jj]),(a-both,high[ii]*low[jj]),
            (b-both,low[ii]*high[jj]),(both,high[ii]*high[jj])]
    assert all(np.all(c>=0) for c,_ in groups)
    assert np.all(sum(c for c,_ in groups)==unclipped)
    for c,v in groups:
        scaled=precision*v;floors=np.floor(scaled)
        sums[catpair]+=c*floors.astype(np.int64)+rng.binomial(c,scaled-floors)
    return dict(rounded=sums,precision=int(precision),clip=float(clip),meta=dict(n=N,d=d,D=len(sums),
        clip_rate=clipped/N,local_statistic_totalization_count=0,raw_norm_percentiles=np.percentile(norm_sample,[50,90,95,99]).tolist(),
        preparation='exact distribution v2: individual numeric/clipped rounding + binomial convolution of exactly equal scalar/category Bernoulli probabilities',
        individual_seed_mapping_preserved=False))
