import numpy as np
from sklearn.metrics import (precision_recall_curve,confusion_matrix,roc_auc_score,
    average_precision_score,brier_score_loss,precision_score,recall_score,f1_score)

def threshold_on_validation(y,p,beta=1):
    if np.unique(y).size<2:return .5
    precision,recall,threshold=precision_recall_curve(y,p)
    f=(1+beta**2)*precision[:-1]*recall[:-1]/np.maximum(beta**2*precision[:-1]+recall[:-1],1e-12)
    # Equal F1 -> higher threshold, less noise; no test-set tuning.
    return float(threshold[np.flatnonzero(f==f.max())[-1]])

def binary_metrics(y,p,threshold):
    y=np.asarray(y,dtype=int);p=np.asarray(p);threshold=np.asarray(threshold);pred=p>=threshold
    tn,fp,fn,tp=confusion_matrix(y,pred,labels=[0,1]).ravel()
    return {'n':len(y),'positive':int(y.sum()),'threshold':float(threshold) if threshold.ndim==0 else 'state_dependent_validation_thresholds',
        'precision':float(precision_score(y,pred,zero_division=0)),
        'recall':float(recall_score(y,pred,zero_division=0)),
        'f1':float(f1_score(y,pred,zero_division=0)),
        'false_positive_rate':float(fp/(tn+fp)) if tn+fp else None,
        'auc':float(roc_auc_score(y,p)) if np.unique(y).size==2 else None,
        'average_precision':float(average_precision_score(y,p)) if y.sum() else None,
        'brier':float(brier_score_loss(y,p)),
        'counts':{'tn':int(tn),'fp':int(fp),'fn':int(fn),'tp':int(tp)}}

def regression(y,p):
    a=np.abs(np.asarray(y)-p)
    return {'mae':float(a.mean()),'rmse':float(np.sqrt(np.mean(a*a))),'p95_absolute_error':float(np.quantile(a,.95))}

def bootstrap_patient_mean(values,patients,repeats=500):
    per=np.asarray([np.mean(np.asarray(values)[patients==k]) for k in np.unique(patients)])
    rng=np.random.default_rng(20261024)
    means=np.mean(rng.choice(per,(repeats,len(per)),replace=True),axis=1)
    return {'patient_mean':float(per.mean()),'bootstrap_95_interval':np.quantile(means,[.025,.975]).tolist()}
