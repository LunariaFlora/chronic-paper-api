"""Reproducible offline training and evaluation. No thresholds tuned on test data."""
from pathlib import Path
import argparse,hashlib,json,platform,time
import joblib,numpy as np,pandas as pd,sklearn
from sklearn.ensemble import HistGradientBoostingClassifier,HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression,Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import classification_report,confusion_matrix,log_loss
from threadpoolctl import threadpool_limits
from .data import ROOT,features,load_glucose,risk_targets
from .contracts import VERSION,SURVEY,LOW,HIGH,RULE_SOURCE
from .metrics import binary_metrics,threshold_on_validation,regression,bootstrap_patient_mean

OUT=ROOT/'artifacts/v2'
SEED=42

def dump(name,obj):
    (OUT/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf8')

def partition(groups):
    """Reuse legacy test/validation groups, split original train into fit/calibration."""
    ix=np.arange(len(groups));g=np.asarray(groups)
    tv,te=next(GroupShuffleSplit(n_splits=1,test_size=.2,random_state=SEED).split(ix,groups=g))
    tr,va=next(GroupShuffleSplit(n_splits=1,test_size=.25,random_state=SEED).split(tv,groups=g[tv]))
    train=tv[tr];va=tv[va]
    fit,cal=next(GroupShuffleSplit(n_splits=1,test_size=1/6,random_state=SEED+1).split(train,groups=g[train]))
    result={'fit':train[fit],'calibration':train[cal],'validation':va,'test':te}
    keys=list(result)
    for i,a in enumerate(keys):
        for b in keys[i+1:]:assert not set(g[result[a]])&set(g[result[b]])
    return result

def survey_metrics(y,p,weights=None):
    pred=(p*(np.asarray(weights) if weights is not None else 1)).argmax(axis=1)
    return {'n':len(y),'log_loss':float(log_loss(y,p,labels=[0,1,2])),
            'confusion_matrix':confusion_matrix(y,pred,labels=[0,1,2]).tolist(),
            'report':classification_report(y,pred,labels=[0,1,2],output_dict=True,zero_division=0)}

def train_survey():
    d=pd.read_csv(ROOT/'datasets/cdc_diabetes/diabetes_012_health_indicators_BRFSS2015.csv')
    x=d[list(SURVEY)];y=d.Diabetes_012.astype(int).to_numpy()
    g=pd.util.hash_pandas_object(x,index=False).to_numpy();s=partition(g)
    models={
        'logistic':make_pipeline(StandardScaler(),LogisticRegression(max_iter=700,random_state=SEED)),
        'gradient_boosting':HistGradientBoostingClassifier(max_iter=180,max_leaf_nodes=15,l2_regularization=5,learning_rate=.07,early_stopping=False,random_state=SEED)}
    scores={};calibrated={}
    for name,m in models.items():
        m.fit(x.iloc[s['fit']],y[s['fit']])
        c=CalibratedClassifierCV(FrozenEstimator(m),method='sigmoid')
        c.fit(x.iloc[s['calibration']],y[s['calibration']])
        scores[name]=survey_metrics(y[s['validation']],c.predict_proba(x.iloc[s['validation']]))
        calibrated[name]=c
    best=min(scores,key=lambda k:scores[k]['log_loss']);model=calibrated[best]
    # Calibrated probabilities alone suppress the rare prediabetes class under argmax.
    # Tune decision costs on validation only; keep probability estimates unchanged.
    valp=model.predict_proba(x.iloc[s['validation']]);valy=y[s['validation']]
    decision_weights=[1.,1.,1.];best_macro=-1
    for w1 in [1.,2.,3.,4.,6.,8.,10.,15.,20.]:
        for w2 in [1.,1.5,2.,3.,4.]:
            pred=(valp*np.array([1,w1,w2])).argmax(axis=1)
            cm=np.bincount(valy*3+pred,minlength=9).reshape(3,3)
            f1=2*np.diag(cm)/np.maximum(cm.sum(axis=0)+cm.sum(axis=1),1)
            score=float(f1.mean())
            if score>best_macro:best_macro=score;decision_weights=[1.,w1,w2]
    # Separate a diabetes-positive alert from the three-way state classification.
    vprob=model.predict_proba(x.iloc[s['validation']])[:,2]
    threshold=threshold_on_validation(y[s['validation']]==2,vprob)
    p=model.predict_proba(x.iloc[s['test']])
    report={'version':VERSION,'task':'Three-class survey diabetes status',
        'labels':{'0':'no_diabetes_or_gestational_only','1':'prediabetes','2':'diabetes'},
        'features':list(SURVEY),'selected_model':best,'selection_metric':'validation multiclass log loss',
        'calibration':'sigmoid on separate feature-profile groups',
        'positive_threshold_policy':'max validation F1, not a clinical decision threshold',
        'positive_threshold':threshold,'validation':scores,
        'decision_weights':decision_weights,'decision_selection':'maximum validation macro F1 over fixed 45 cost settings',
        'validation_decision_macro_f1':best_macro,
        'test':survey_metrics(y[s['test']],p,decision_weights),
        'test_unweighted_argmax':survey_metrics(y[s['test']],p),
        'test_diabetes_flag':binary_metrics(y[s['test']]==2,p[:,2],threshold),
        'baseline_all_majority_accuracy':float(np.mean(y[s['test']]==0)),
        'split':{k:{'rows':len(v),'groups':len(set(g[v]))} for k,v in s.items()},
        'source':'https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset',
        'evaluation_note':'Existing internal holdout previously reported for v1; no fresh clinical external validation'}
    joblib.dump({'version':VERSION,'model':model,'features':list(SURVEY),'threshold':threshold,'decision_weights':decision_weights},OUT/'survey.joblib')
    dump('survey_metrics.json',report)
    sample=x.iloc[s['test'][0]].to_dict();dump('survey_input.json',sample)
    dump('survey_truth.json',{'class':int(y[s['test'][0]])})
    split=pd.DataFrame({'source_row':np.arange(len(d)),'feature_group':g.astype(str),'split':''})
    for k,v in s.items():split.loc[v,'split']=k
    split.to_csv(OUT/'survey_split.csv',index=False)
    print('SURVEY',best,report['test_diabetes_flag'],flush=True)

def calibrate_binary(model,x,y,ix):
    if np.unique(y[ix]).size<2:
        return model,'not_calibrated_single_class_calibration_data'
    c=CalibratedClassifierCV(FrozenEstimator(model),method='sigmoid');c.fit(x[ix],y[ix])
    return c,'sigmoid_on_disjoint_calibration_patients'

def train_glucose():
    raw,future,meta=load_glucose();x=features(raw);g=meta.patient.to_numpy();s=partition(g)
    forecasters={};events={};reports={};rows=[]
    for steps in (2,4):
        horizon=str(steps*15);y=future[:,steps-1]
        candidates={
            'ridge':make_pipeline(StandardScaler(),Ridge(alpha=20)),
            'gradient_boosting':HistGradientBoostingRegressor(max_iter=220,max_leaf_nodes=15,l2_regularization=5,learning_rate=.06,early_stopping=False,random_state=SEED)}
        vs={'persistence':regression(y[s['validation']],raw[s['validation'],-1])}
        for name,m in candidates.items():
            m.fit(x[s['fit']],y[s['fit']]);vs[name]=regression(y[s['validation']],m.predict(x[s['validation']]))
        best=min(vs,key=lambda k:vs[k]['mae'])
        model=candidates.get(best)
        def preds(ix):return model.predict(x[ix]) if model is not None else raw[ix,-1]
        residual=np.abs(y[s['calibration']]-preds(s['calibration']))
        q=float(np.quantile(residual,min(1,np.ceil((len(residual)+1)*.9)/len(residual)),method='higher'))
        ix=s['test'];p=preds(ix);actual=y[ix]
        per=[]
        for patient in np.unique(g[ix]):
            mask=g[ix]==patient
            per.append({'patient':patient,'mae':float(np.abs(actual[mask]-p[mask]).mean()),
                'baseline_mae':float(np.abs(actual[mask]-raw[ix,-1][mask]).mean()),'n':int(mask.sum())})
        report={'selected_model':best,'validation':vs,'test':regression(actual,p),
            'test_persistence':regression(actual,raw[ix,-1]),'per_patient':per,
            'patient_bootstrap':bootstrap_patient_mean(np.abs(actual-p),g[ix]),
            'interval':{'method':'pooled absolute calibration residual, descriptive only under clustered windows',
                'nominal_coverage':.9,'half_width_mg_dl':q,
                'test_empirical_coverage':float(np.mean(np.abs(actual-p)<=q))}}
        forecasters[horizon]={'model':model,'interval_half_width':q}
        for i,z in enumerate(ix):rows.append({'patient':g[z],'file':meta.iloc[z]['file'],'row':int(meta.iloc[z]['row']),
            'horizon':int(horizon),'actual':float(actual[i]),'prediction':float(p[i]),'baseline':float(raw[z,-1])})
        ev={}
        for event in ['low','high']:
            target=risk_targets(future,steps)[event]
            c=HistGradientBoostingClassifier(max_iter=160,max_leaf_nodes=15,l2_regularization=5,
                learning_rate=.06,early_stopping=False,random_state=SEED,class_weight='balanced')
            if np.unique(target[s['fit']]).size<2:raise RuntimeError(f'Insufficient training classes for {horizon}/{event}')
            c.fit(x[s['fit']],target[s['fit']])
            c,calnote=calibrate_binary(c,x,target,s['calibration'])
            pv=c.predict_proba(x[s['validation']])[:,1]
            th=threshold_on_validation(target[s['validation']],pv)
            pt=c.predict_proba(x[ix])[:,1]
            current=(raw[:, -1]<LOW) if event=='low' else (raw[:, -1]>HIGH)
            stable=(raw[:,-1]>=LOW)&(raw[:,-1]<=HIGH)
            vm=stable[s['validation']]
            early_th=threshold_on_validation(target[s['validation']][vm],pv[vm],beta=2)
            test_th=np.where(stable[ix],early_th,th)
            # A genuine predictive baseline: continue the past 45-minute slope.
            slope=(raw[:,-1]-raw[:,-4])/3
            extrap=raw[:,-1,None]+slope[:,None]*np.arange(1,steps+1)
            linear=(extrap.min(axis=1)<LOW) if event=='low' else (extrap.max(axis=1)>HIGH)
            et={'calibration':calnote,'threshold':th,'early_warning_threshold':early_th,
                'threshold_policy':'max validation F1 generally; max validation F2 when currently in range, prioritizing recall',
                'test':binary_metrics(target[ix],pt,test_th),
                'test_current_state_baseline':binary_metrics(target[ix],current[ix].astype(float),.5),
                'test_linear_extrapolation_baseline':binary_metrics(target[ix],linear[ix].astype(float),.5),
                'test_new_onset_from_in_range':binary_metrics(target[ix][stable[ix]],pt[stable[ix]],early_th),
                'validation_new_onset':binary_metrics(target[s['validation']][vm],pv[vm],early_th),
                'positive_counts':{k:int(target[v].sum()) for k,v in s.items()}}
            ev[event]=et;events[f'{horizon}_{event}']={'model':c,'threshold':th,'early_warning_threshold':early_th,'calibration':calnote}
            for i,z in enumerate(ix):rows.append({'patient':g[z],'file':meta.iloc[z]['file'],'row':int(meta.iloc[z]['row']),
                'horizon':int(horizon),'event':event,'actual_event':int(target[z]),'event_score':float(pt[i]),
                'event_flag':bool(pt[i]>=test_th[i]),'currently_in_range':bool(stable[z])})
        report['events']=ev;reports[horizon]=report
        print('GLUCOSE',horizon,report['test'],'events', {k:v['test'] for k,v in ev.items()},flush=True)
    report={'version':VERSION,'horizons':reports,'source':'https://figshare.com/articles/dataset/diabetes_datasets_zip/21600933',
        'cohort':'100 ShanghaiT2DM patients; 15 minute CGM',
        'split':{k:{'n':len(v),'patients':sorted(set(g[v]))} for k,v in s.items()},
        'risk_definition':'Any sampled glucose <70 or >180 mg/dL in next (0,30] or (0,60] minutes; includes persistent events',
        'risk_source':RULE_SOURCE,'selection':'Validation only; calibration patients separate from fit/validation/test',
        'limitations':['15 minute samples may miss shorter events','Internal holdout was previously used in v1 reports',
                       'No prospective clinical validation','Pregnancy/children/personal treatment targets not supported']}
    joblib.dump({'version':VERSION,'forecasters':forecasters,'events':events,
        'training_min':float(raw[s['fit']].min()),'training_max':float(raw[s['fit']].max())},OUT/'glucose.joblib')
    dump('glucose_metrics.json',report)
    split=meta.copy();split['split']=''
    for k,v in s.items():split.loc[v,'split']=k
    split.to_csv(OUT/'glucose_split.csv',index=False)
    pd.DataFrame(rows).to_csv(OUT/'glucose_test_predictions.csv',index=False)
    # Save actual held-out examples for normal, high, low and incident risks when available.
    examples={}
    for name,mask in [('in_range',(raw[:,-1]>=LOW)&(raw[:,-1]<=HIGH)),('high',raw[:,-1]>HIGH),('low',raw[:,-1]<LOW),
                      ('future_high',(raw[:,-1]<=HIGH)&(future.max(axis=1)>HIGH)),
                      ('future_low',(raw[:,-1]>=LOW)&(future.min(axis=1)<LOW))]:
        candidates=s['test'][mask[s['test']]]
        if not len(candidates):continue
        i=int(candidates[0]);t=pd.Timestamp(meta.iloc[i].timestamp).tz_localize('Asia/Shanghai')
        measurements=[{'timestamp':(t-pd.Timedelta(minutes=15*(11-j))).isoformat(),'value':float(v)} for j,v in enumerate(raw[i])]
        payload={'unit':'mg/dL','measurements':measurements,'as_of':t.isoformat()}
        dump(f'glucose_{name}_input.json',payload)
        examples[name]={'patient':g[i],'file':meta.iloc[i]['file'],'row':int(meta.iloc[i]['row']),
                        'actual_future_15_30_45_60':future[i].tolist(),'input':f'glucose_{name}_input.json'}
    dump('glucose_example_truth.json',examples)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--task',choices=['all','survey','glucose'],default='all');args=parser.parse_args()
    OUT.mkdir(exist_ok=True,parents=True);start=time.time()
    with threadpool_limits(limits=6):
        if args.task in ['all','survey']:train_survey()
        if args.task in ['all','glucose']:train_glucose()
    dump('environment.json',{'version':VERSION,'python':platform.python_version(),'sklearn':sklearn.__version__,'seed':SEED,'training_seconds':time.time()-start})
    inputs=[ROOT/'datasets/cdc_diabetes/diabetes_012_health_indicators_BRFSS2015.csv',*sorted((ROOT/'datasets/shanghai_cgm/Shanghai_T2DM').iterdir())]
    dump('sources_sha256.json',{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs if p.is_file()})

if __name__=='__main__':main()
