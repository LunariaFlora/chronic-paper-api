"""Train and reuse a classifier from an explicit table schema, including new targets."""
from pathlib import Path
import hashlib,json
import joblib,numpy as np,pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import classification_report,confusion_matrix,roc_auc_score
from threadpoolctl import threadpool_limits
from .table_io import read_table,mapped

def prepare(d,schema):
    d=mapped(d,schema.get('mapping',{}))
    cols=schema['numeric']+schema.get('categorical',[])
    if len(set(cols))!=len(cols):raise ValueError('特征列重复')
    missing=set(cols)-set(d)
    if missing:raise ValueError(f'缺少训练所需字段：{sorted(missing)}')
    x=d[cols].copy();errors=pd.Series('',index=d.index,dtype=object)
    optional=set(schema.get('allow_missing',[]))
    for c in cols:
        empty=x[c].isna()|x[c].astype(str).str.strip().isin(['','?','NA','NaN','nan'])
        if c in schema['numeric']:
            v=pd.to_numeric(x[c].where(~empty),errors='coerce')
            bad=(~empty)&(~np.isfinite(v))
            if c in schema.get('ranges',{}):
                lo,hi=schema['ranges'][c];bad|=(~empty)&(~v.between(lo,hi))
            if c in schema.get('enums',{}):bad|=(~empty)&(~v.isin(schema['enums'][c]))
            x[c]=v.astype(float)
        else:
            x[c]=x[c].where(~empty,np.nan).map(lambda v:str(v).strip() if pd.notna(v) else np.nan)
            bad=pd.Series(False,index=d.index)
            if c in schema.get('categorical_numeric',[]):
                numeric=pd.to_numeric(x[c],errors='coerce');bad|=(~empty)&(~np.isfinite(numeric))
                x[c]=numeric.map(lambda v:str(float(v)) if pd.notna(v) else np.nan)
            if c in schema.get('enums',{}):bad|=(~empty)&(~x[c].isin(schema['enums'][c]))
        if c not in optional:bad|=empty
        errors.loc[bad]+=c+'缺失、非法或不在约定范围；'
    return x,errors

def train_table(input_path,schema_path,output,algorithm='auto'):
    if algorithm not in ('auto','logistic','random_forest'):raise ValueError('不支持的分类训练方式')
    schema=json.loads(Path(schema_path).read_text(encoding='utf-8-sig'))
    out=Path(output)
    if out.exists():raise FileExistsError('模型目录已存在，请使用新目录保留旧模型')
    d=read_table(input_path,schema);x,errors=prepare(d,schema)
    target=schema['target']
    if target in x:raise ValueError('标签不能同时作为输入特征')
    if schema.get('group_column') in x:raise ValueError('患者分组ID不能作为预测特征')
    if target not in d:raise ValueError(f'训练表缺少标签列：{target}')
    y=d[target].astype(str).str.strip()
    if schema.get('label_map'):
        # Numeric raw labels can be written as 1 or 1.0 in Excel.
        y=y.map(lambda v:str(int(float(v))) if v.replace('.','',1).isdigit() and float(v).is_integer() else v)
        y=y.map(schema['label_map'])
    missing=y.isna()|y.astype(str).str.strip().isin(['','nan','?'])
    errors.loc[missing]+='缺少或不支持的训练标签；'
    good=errors.eq('');xx=x[good];yy=y[good].astype(str)
    labels=sorted(yy.unique().tolist())
    encoded=pd.Series([labels.index(v) for v in yy],index=yy.index,dtype=int)
    groups=pd.util.hash_pandas_object(xx,index=False).to_numpy()
    if schema.get('group_column'):
        # Prevent both same-person and identical-feature leakage via union-find.
        ids=d.loc[good,schema['group_column']]
        if ids.isna().any() or ids.astype(str).str.strip().eq('').any():raise ValueError('训练分组ID不可缺失')
        parent=list(range(len(xx)))
        def find(i):
            while parent[i]!=i:parent[i]=parent[parent[i]];i=parent[i]
            return i
        for keys in [groups,ids.astype(str).to_numpy()]:
            seen={}
            for i,key in enumerate(keys):
                if key in seen:parent[find(i)]=find(seen[key])
                else:seen[key]=i
        groups=np.array([find(i) for i in range(len(xx))])
    counts=pd.DataFrame({'y':yy.to_numpy(),'g':groups}).groupby('y').g.nunique()
    if len(counts)<2 or counts.min()<10:raise ValueError('每类至少需要10组独立特征/人员记录，且至少两类；数据不足时不伪造评估')
    folds=list(StratifiedGroupKFold(5,shuffle=True,random_state=20261025).split(xx,yy,groups))
    test=folds[0][1];val=folds[1][1];fit=np.setdiff1d(np.arange(len(xx)),np.r_[test,val])
    expected=set(yy)
    for name,ix in [('fit',fit),('validation',val),('test',test)]:
        if set(yy.iloc[ix])!=expected:raise ValueError(f'{name}分组后缺少类别，需要更多独立样本')
    transformers=[('numeric',make_pipeline(SimpleImputer(strategy='median'),StandardScaler()),schema['numeric'])]
    if schema.get('categorical'):
        transformers.append(('categorical',make_pipeline(SimpleImputer(strategy='most_frequent'),OneHotEncoder(handle_unknown='ignore',sparse_output=False)),schema['categorical']))
    def pipeline(model):return make_pipeline(ColumnTransformer(transformers),model)
    candidates={'logistic':pipeline(LogisticRegression(max_iter=1000,class_weight='balanced',random_state=42)),
        'random_forest':pipeline(RandomForestClassifier(n_estimators=200,min_samples_leaf=3,class_weight='balanced',random_state=42,n_jobs=4))}
    if algorithm!='auto':candidates={algorithm:candidates[algorithm]}
    scores={}
    with threadpool_limits(limits=4):
        for name,model in candidates.items():
            model.fit(xx.iloc[fit],encoded.iloc[fit].to_numpy(copy=True))
            valpred=np.asarray(labels)[model.predict(xx.iloc[val]).astype(int)]
            scores[name]=classification_report(yy.iloc[val],valpred,output_dict=True,zero_division=0)
        best=max(scores,key=lambda k:scores[k]['macro avg']['f1-score']);model=candidates[best]
        pred=np.asarray(labels)[model.predict(xx.iloc[test]).astype(int)];probs=model.predict_proba(xx.iloc[test])
    out.mkdir(parents=True)
    report={'task':schema['task'],'source':schema.get('source'),'rows':len(d),'valid_rows':len(xx),
        'discarded_training_rows':int((~good).sum()),'selected_model':best,'training_strategy':algorithm,'validation':scores,
        'test':classification_report(yy.iloc[test],pred,output_dict=True,zero_division=0),
        'test_confusion_matrix':confusion_matrix(yy.iloc[test],pred,labels=labels).tolist(),
        'classes':labels,'partitions':{n:len(ix) for n,ix in [('fit',fit),('validation',val),('test',test)]},
        'interpretation':schema['interpretation'],'evaluation':'Internal grouped holdout; no clinical validation; no refit on test.',
        'majority_baseline_accuracy':float((yy.iloc[test]==yy.iloc[fit].value_counts().idxmax()).mean()),
        'source_sha256':hashlib.sha256(Path(input_path).read_bytes()).hexdigest()}
    if len(labels)==2:report['test_auc']=float(roc_auc_score(yy.iloc[test]==labels[1],probs[:,1]))
    bundle={'format_version':1,'schema':schema,'model':model,'classes':labels}
    joblib.dump(bundle,out/'table_model.joblib')
    (out/'metrics.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf8')
    split=pd.DataFrame({'source_row':np.arange(len(d))+1,'split':'invalid','reason':errors,'feature_group':''})
    original=np.flatnonzero(good)
    split.loc[original,'feature_group']=groups.astype(str)
    for n,ix in [('fit',fit),('validation',val),('test',test)]:split.loc[original[ix],'split']=n
    split.to_csv(out/'split.csv',index=False,encoding='utf-8-sig')
    checks=pd.DataFrame({'source_row':original[test]+1,'actual':yy.iloc[test].to_numpy(),'predicted':pred,
        'correct':yy.iloc[test].to_numpy()==pred})
    for j,label in enumerate(labels):checks['score_'+label]=probs[:,j]
    checks.to_csv(out/'test_predictions.csv',index=False,encoding='utf-8-sig')
    return report
