"""Optional advice uses already saved predictions; never mutates prediction files."""
import hashlib
from pathlib import Path
import pandas as pd
from .model_match import schema_for
from .llm_advice import packet_from_row,append_llm_advice
from .table_io import read_table
from .data import ROOT


def saved_records(selected,folder,limit):
    """Reconnect saved holdout rows to original inputs; never send their true labels."""
    test=next((p for p in [folder/'测试集_真实值与预测.csv',folder/'model/test_predictions.csv'] if p.exists()),None)
    if test is None:return [],None
    app=ROOT.parent if ROOT.name=='程序' else ROOT
    imports=app/'导入数据'
    checks=pd.read_csv(test)
    records=[]
    if selected['kind']=='glucose':
        chosen=checks.loc[(checks.horizon==30)&checks.prediction.notna()].head(limit)
        for _,r in chosen.iterrows():
            filename=str(r['file'])
            path=imports/'连续血糖'/filename
            if not path.exists():path=ROOT/'datasets/shanghai_cgm/Shanghai_T2DM'/filename
            if not path.exists():return [],test
            raw=pd.read_excel(path)
            c=next(c for c in raw if 'CGM' in str(c))
            current=float(raw.iloc[int(r['row'])][c])
            future60=checks.loc[(checks['file']==filename)&(checks['row']==r['row'])&(checks.horizon==60)&checks.prediction.notna(),'prediction']
            if future60.empty:continue
            packet={'task':'glucose_forecast','task_meaning':'历史记录回放；血糖单位mg/dL，未来数值由本地模型估计。',
                    'feature_values':{'measured_glucose_mg_dl':current},
                    'local_model_result':{'predicted_glucose_30min_mg_dl':float(r.prediction),'predicted_glucose_60min_mg_dl':float(future60.iloc[0])},
                    'instruction':'只根据当前实测和已保存预测给健康建议；未提供真实未来值。'}
            records.append({'source_row':int(r['row'])+2,'packet':packet})
        return records,test
    schema=schema_for(selected)
    if selected['kind']=='survey':
        name='diabetes_012_health_indicators_BRFSS2015.csv'
        candidates=[imports/name,ROOT/'datasets/cdc_diabetes'/name]
    elif schema.get('task')=='heart_disease_status':
        name='processed.cleveland.data';candidates=[imports/name,ROOT/'datasets/heart_disease'/name]
    else:
        original=selected.get('info',{}).get('input')
        if not original:return [],test
        candidates=[Path(original),imports/Path(original).name]
    path=next((p for p in candidates if p.is_file()),None)
    if path is None:return [],test
    raw=read_table(path,schema)
    for _,r in checks.head(limit).iterrows():
        i=int(r.source_row)-1
        if not 0<=i<len(raw):return [],test
        scores={c[6:]:float(r[c]) for c in checks if c.startswith('score_')}
        context=raw.iloc[i].to_dict()
        for key,original in schema.get('mapping',{}).items():context[key]=context.get(original)
        packet=packet_from_row(context,schema,{'predicted_class':str(r.predicted),'class_scores':scores})
        records.append({'source_row':i+1,'packet':packet})
    return records,test


def create_advice(selected,folder,limit=1):
    folder=Path(folder)
    path=next((p for p in [folder/'predictions/results.csv',folder/'results.csv'] if p.exists()),None)
    if path is None:
        records,test=saved_records(selected,folder,limit)
        if not records:return False
        before=hashlib.sha256(test.read_bytes()).digest()
        append_llm_advice(records,folder/'advice',limit=limit)
        if hashlib.sha256(test.read_bytes()).digest()!=before:raise RuntimeError('测试结果发生意外变更')
        return True
    before=hashlib.sha256(path.read_bytes()).digest();records=[]
    for chunk in pd.read_csv(path,chunksize=128):
        for _,row in chunk.loc[chunk.result_status=='ok'].iterrows():
            if selected['kind']=='glucose':
                packet={'task':'glucose_forecast','task_meaning':'历史血糖记录回放；基于过去12次测量预测未来30/60分钟，单位mg/dL。',
                    'feature_values':{'measured_glucose_mg_dl':float(row.result_current_mg_dl)},
                    'local_model_result':{'predicted_glucose_30min_mg_dl':float(row.result_glucose_30min_mg_dl),
                                          'predicted_glucose_60min_mg_dl':float(row.result_glucose_60min_mg_dl)},
                    'instruction':'当前值是实测，未来值是本地模型估计；只生成一般健康建议，不能更改预测。'}
            else:
                schema=schema_for(selected);context=row.to_dict()
                for model_field,raw_field in schema.get('mapping',{}).items():context[model_field]=row.get(raw_field)
                scores={c[len('result_score_'):]:float(row[c]) for c in row.index if c.startswith('result_score_') and c!='result_score_note' and pd.notna(row[c])}
                packet=packet_from_row(context,schema,{'predicted_class':str(row.result_class),'class_scores':scores,
                    'interpretation':'本地模型的类别输出，不是诊断或未来发病概率。'})
            records.append({'source_row':int(row.result_source_row),'packet':packet})
            if len(records)>=limit:break
        if len(records)>=limit:break
    if not records:return False
    append_llm_advice(records,folder/'advice',limit=limit)
    if hashlib.sha256(path.read_bytes()).digest()!=before:raise RuntimeError('预测文件发生意外变更')
    return True
