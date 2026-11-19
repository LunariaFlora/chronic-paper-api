"""Freeze and independently recompute final results, including a new-cohort stress test."""
from pathlib import Path
import hashlib,json,warnings
import numpy as np,pandas as pd,joblib
from threadpoolctl import threadpool_limits
from chronic.data import ROOT,load_glucose,windows_from_frame,features,risk_targets
from chronic.metrics import regression,binary_metrics

ART=ROOT/'artifacts/v2';OUT=ROOT/'outputs/completion'

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    b=joblib.load(ART/'glucose.joblib')
    xs=[];fs=[];ids=[]
    for p in sorted((ROOT/'datasets/shanghai_cgm/Shanghai_T1DM').iterdir()):
        if p.suffix.lower() not in ('.xlsx','.xls'):continue
        with warnings.catch_warnings():warnings.simplefilter('ignore');d=pd.read_excel(p)
        x,f,m=windows_from_frame(d,p.stem.split('_')[0],p.name)
        xs.extend(x);fs.extend(f);ids.extend([p.stem.split('_')[0]]*len(x))
    raw=np.asarray(xs);future=np.asarray(fs)
    keep=((raw>=b['training_min'])&(raw<=b['training_max'])).all(axis=1)
    x=features(raw[keep]);future=future[keep];current=raw[keep,-1]
    stress={'cohort':'ShanghaiT1DM, unseen type-1 cohort from same publisher, not external clinical validation',
        'patients':len(set(ids)),'windows_total':len(raw),'windows_supported':int(keep.sum()),
        'abstained_windows':int((~keep).sum()),'horizons':{}}
    with threadpool_limits(limits=4):
        for h,steps in [('30',2),('60',4)]:
            m=b['forecasters'][h]['model'];pred=m.predict(x) if m else current
            events={}
            for event in ['low','high']:
                e=b['events'][h+'_'+event];p=e['model'].predict_proba(x)[:,1]
                thresholds=np.where((current>=70)&(current<=180),e['early_warning_threshold'],e['threshold'])
                events[event]=binary_metrics(risk_targets(future,steps)[event],p,thresholds)
            stress['horizons'][h]={'prediction':regression(future[:,steps-1],pred),
                'persistence':regression(future[:,steps-1],current),'events':events}
    (OUT/'cohort_stress_results.json').write_text(json.dumps(stress,ensure_ascii=False,indent=2),encoding='utf8')
    s=json.loads((ART/'survey_metrics.json').read_text(encoding='utf8'))
    g=json.loads((ART/'glucose_metrics.json').read_text(encoding='utf8'))
    t=json.loads((OUT/'test_results.json').read_text(encoding='utf8'))
    assert t['passed'],'Tests must pass before publishing final results'
    summary={'version':'2.0.0','tests':t,'survey':s,'glucose':g,'cohort_stress':stress,
             'model_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in ART.glob('*.joblib')},
             'scope':'Local prediction algorithms and offline risk output; HTTP API and recommendation generation excluded by user scope',
             'acceptance':'Engineering functionality verified; no clinical acceptance or guaranteed predictive accuracy'}
    (OUT/'results.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf8')
    print(json.dumps({'stress':{h:v['prediction'] for h,v in stress['horizons'].items()},'tests':t['tests_run']},ensure_ascii=False))

if __name__=='__main__':main()
