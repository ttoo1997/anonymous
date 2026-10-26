'Reconstruct the public-fitted transformation without fitting new statistics.'
import numpy as np
import pandas as pd

def transform(values,time,station,names,meta):
    physical=[i for i,n in enumerate(names) if n not in meta['excluded_physical']]
    labels=[names[i] for i in physical]
    X=values[:,physical]
    hour=(time%86400)/3600;doy=pd.to_datetime(time,unit='s',utc=True).dayofyear.to_numpy()
    X=np.column_stack([X,np.sin(hour*2*np.pi/24),np.cos(hour*2*np.pi/24),np.sin(doy*2*np.pi/365.25),np.cos(doy*2*np.pi/365.25)])
    labels+=['hour_sin','hour_cos','annual_sin','annual_cos']
    missing=~np.isfinite(X);X=np.where(missing,np.asarray(meta['imputation_medians']),X)
    d=meta['physical_dim'];keep=[labels.index(n) for n in meta['features'][:d]]
    normalized=(X[:,keep]-np.asarray(meta['feature_means']))/np.asarray(meta['feature_std'])
    indicators=[]
    for name in meta['features'][d:]:
        if name.startswith('missing_'):indicators.append(missing[:,labels.index(name[len('missing_'):])])
        elif name.startswith('turbine_'):indicators.append(station==int(name.rsplit('_',1)[1]))
        else:raise ValueError(name)
    if indicators:normalized=np.column_stack([normalized,np.column_stack(indicators)-np.asarray(meta['indicator_means'])])
    return normalized.astype(np.float32)
