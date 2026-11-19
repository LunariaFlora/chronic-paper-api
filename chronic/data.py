from pathlib import Path
import hashlib
import warnings
import numpy as np
import pandas as pd
from .contracts import LOOKBACK

ROOT=Path(__file__).resolve().parents[1]

def risk_targets(future,steps):
    from .contracts import LOW,HIGH
    f=np.asarray(future)[:,:steps]
    return {'low':f.min(axis=1)<LOW,'high':f.max(axis=1)>HIGH}

def features(x):
    """Only past/current inputs. Never accept target arrays here."""
    x=np.asarray(x,dtype=float)
    if x.ndim==1:x=x.reshape(1,-1)
    assert x.shape[1]==LOOKBACK
    return np.column_stack([x,np.diff(x[:,-4:],axis=1),
        *[a for n in (3,6,12) for a in (x[:,-n:].mean(axis=1),x[:,-n:].std(axis=1),x[:,-n:].min(axis=1),x[:,-n:].max(axis=1))],
        (x[:,-1]-x[:,-4])/45])

def clock_features(history_features, minutes):
    """Add the circular time of day in the patient's configured local time zone."""
    phase = 2 * np.pi * np.asarray(minutes, dtype=float) / 1440
    return np.column_stack([history_features, np.sin(phase), np.cos(phase)])

def forecast_values(entry, history_features, current, minutes):
    model = entry['model']
    if model is None:
        return current
    x = clock_features(history_features, minutes) if entry.get('clock') else history_features
    prediction = model.predict(x)
    return current + prediction if entry.get('target') == 'change' else prediction

def windows_from_frame(d,patient,file):
    """Current at i; future labels use i+1..i+4, strictly after every input."""
    c=next(c for c in d.columns if 'CGM' in str(c))
    v=pd.to_numeric(d[c],errors='coerce').to_numpy(dtype=float)
    t=pd.to_datetime(d['Date'],errors='coerce')
    delta=t.diff().dt.total_seconds().to_numpy()
    x=[];future=[];meta=[]
    for i in range(LOOKBACK-1,len(d)-4):
        a=v[i-LOOKBACK+1:i+5]
        if not np.isfinite(a).all() or not ((a>=20)&(a<=600)).all():continue
        if not np.isfinite(delta[i-LOOKBACK+2:i+5]).all() or not np.all(np.abs(delta[i-LOOKBACK+2:i+5]-900)<=1):continue
        x.append(a[:LOOKBACK]);future.append(a[LOOKBACK:])
        meta.append({'patient':str(patient),'file':file,'row':i,'timestamp':t.iloc[i].isoformat()})
    return x,future,meta

def load_glucose(cache=True):
    dest=ROOT/'work/glucose_v2_windows.npz'
    files=sorted(p for p in (ROOT/'datasets/shanghai_cgm/Shanghai_T2DM').iterdir() if p.suffix.lower() in ('.xls','.xlsx'))
    fingerprint=hashlib.sha256(b'window-v2-12lags-4future-15min-20to600'+b''.join(hashlib.sha256(p.read_bytes()).digest() for p in files)).hexdigest()
    if cache and dest.exists():
        with np.load(dest,allow_pickle=False) as z:
            if 'fingerprint' in z and str(z['fingerprint'])==fingerprint:
                return z['x'],z['future'],pd.DataFrame({k:z['source_file' if k=='file' else k] for k in ['patient','file','row','timestamp']})
    xs=[];ys=[];ms=[]
    for p in files:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            d=pd.read_excel(p)
        x,y,m=windows_from_frame(d,p.stem.split('_')[0],p.name)
        xs.extend(x);ys.extend(y);ms.extend(m)
    x=np.asarray(xs);y=np.asarray(ys);m=pd.DataFrame(ms)
    np.savez_compressed(dest,x=x,future=y,fingerprint=fingerprint,**{('source_file' if k=='file' else k):m[k].to_numpy(dtype=str if k!='row' else int) for k in m})
    return x,y,m
