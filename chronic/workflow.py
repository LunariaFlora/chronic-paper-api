"""A full local classification experiment, followed by optional separate cloud advice."""
from contextlib import contextmanager
from pathlib import Path
import hashlib,json,socket,time
import pandas as pd
from .discovery import discover
from .table_models import train_table
from .table_io import read_table
from .batch import run_batch

@contextmanager
def local_only():
    old_connect=socket.socket.connect;old_create=socket.create_connection
    def deny(*args,**kwargs):raise RuntimeError('本地机器学习阶段禁止联网')
    socket.socket.connect=deny;socket.create_connection=deny
    try:yield
    finally:socket.socket.connect=old_connect;socket.create_connection=old_create

def run_workflow(input_path,output,target=None,schema_path=None,recommendations=False,advice_limit=3,feature_columns=None,verbose=True,algorithm='auto'):
    out=Path(output)
    if out.exists():raise FileExistsError('输出目录已存在，请选新目录保存本次全过程')
    out.mkdir(parents=True);started=time.time()
    with local_only():
        print('[1/4] 本地读取原始整表，识别字段和目标列',flush=True)
        if schema_path:
            schema=json.loads(Path(schema_path).read_text(encoding='utf-8-sig'))
            inspection={'recognition':'user_schema','target':schema['target'],'note':'使用明确的字段配置'}
        else:schema,inspection=discover(input_path,target)
        from .selection import select_features
        schema=select_features(schema,feature_columns)
        inspection['selected_input_columns']=schema['numeric']+schema.get('categorical',[])
        (out/'columns.json').write_text(json.dumps(inspection,ensure_ascii=False,indent=2),encoding='utf8')
        (out/'schema.json').write_text(json.dumps(schema,ensure_ascii=False,indent=2),encoding='utf8')
        print(f"[2/4] 本地整理并训练：目标={schema['target']}，分训练/验证/测试组",flush=True)
        report=train_table(input_path,out/'schema.json',out/'model',algorithm=algorithm)
        print('[3/4] 用本次新训练的模型逐行预测，导出结果',flush=True)
        batch=run_batch(input_path,out/'predictions',model=out/'model/table_model.joblib',include_rule_advice=False)
        # Independently recompute the held-out accuracy from the exported labels.
        test=pd.read_csv(out/'model/test_predictions.csv',dtype={'actual':str,'predicted':str})
        observed=float((test.actual==test.predicted).mean())
        if abs(observed-report['test']['accuracy'])>1e-12:raise RuntimeError('导出测试结果与评估指标不一致')
        split=pd.read_csv(out/'model/split.csv',dtype={'feature_group':str},keep_default_na=False)
        valid=split[split.split!='invalid']
        if valid.groupby('feature_group')['split'].nunique().max()!=1:raise RuntimeError('分组跨训练/测试集')
        core={'input':str(Path(input_path).resolve()),'target':schema['target'],'model':report['selected_model'],'training_strategy':algorithm,
            'rows':batch['input_rows'],'status_counts':batch['status_counts'],'test_rows':len(test),
            'test_accuracy':observed,'test_macro_f1':report['test']['macro avg']['f1-score'],
            'majority_baseline_accuracy':report['majority_baseline_accuracy'],
            'test_accuracy_recomputed_from_csv':True,'feature_groups_disjoint':True,
            'local_ml_network_blocked':True,'llm_used_for_prediction':False,
            'core_seconds':round(time.time()-started,2),'recommendations':'not_requested'}
        immutable=[out/'model/table_model.joblib',out/'model/metrics.json',out/'model/test_predictions.csv',out/'predictions/results.csv']
        before={str(p.relative_to(out)):hashlib.sha256(p.read_bytes()).hexdigest() for p in immutable}
        core['prediction_sha256']=before
        (out/'run_summary.json').write_text(json.dumps(core,ensure_ascii=False,indent=2),encoding='utf8')
    if recommendations:
        print(f'[4/4] 本地预测已保存；单独调用大模型为前{advice_limit}条有效测试记录生成建议',flush=True)
        from .llm_advice import append_llm_advice,packet_from_row
        d=read_table(input_path,schema);records=[]
        for _,r in test.head(advice_limit).iterrows():
            i=int(r.source_row)-1
            scores={c[6:]:float(r[c]) for c in test if c.startswith('score_')}
            packet=packet_from_row(d.iloc[i].to_dict(),schema,{'predicted_class':r.predicted,'class_scores':scores,
                'interpretation':'本地模型的状态分类分数，不是诊断或未来发病概率'})
            records.append({'source_row':i+1,'packet':packet})
        recs=append_llm_advice(records,out/'advice',advice_limit)
        core['recommendations']={'requested':len(records),'successful':sum(x['status']=='ok' for x in recs),
            'failed':sum(x['status']!='ok' for x in recs),'provider':'Kimi',
            'models':sorted({x['response']['model'] for x in recs if x['status']=='ok' and 'model' in x['response']})}
    else:print('[4/4] 已跳过大模型建议；全过程可离线完成',flush=True)
    core['prediction_unchanged_after_advice']=all(hashlib.sha256((out/p).read_bytes()).hexdigest()==v for p,v in before.items())
    if not core['prediction_unchanged_after_advice']:raise RuntimeError('预测产物被意外修改')
    (out/'run_summary.json').write_text(json.dumps(core,ensure_ascii=False,indent=2),encoding='utf8')
    if verbose:print(json.dumps(core,ensure_ascii=False,indent=2),flush=True)
    else:print(f"完成：{core['rows']}行；独立测试{core['test_rows']}条，准确率{core['test_accuracy']:.2%}，多数类基线{core['majority_baseline_accuracy']:.2%}。",flush=True)
    return core
