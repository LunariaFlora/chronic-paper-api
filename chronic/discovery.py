"""Code-only schema discovery. Infers storage types, not hidden clinical meaning."""
from pathlib import Path
import json,re
import pandas as pd
from .data import ROOT
from .table_io import read_table

TARGET_NAMES={'target','label','class','disease','diabetes','diabetes_012','cardio','outcome','num'}
ID_NAMES={'id','patient','patientid','patient_id','subject_id','name','姓名','患者编号','患者id','身份证','身份证号','手机号','phone','email','编号','record_id','index'}
TIME_NAMES={'date','timestamp','time','datetime','日期','时间'}

def discover(input_path,target=None):
    path=Path(input_path)
    if path.name=='processed.cleveland.data':
        schema=json.loads((ROOT/'configs/heart_cleveland.json').read_text(encoding='utf8'))
        if target and target!='num':raise ValueError('Cleveland原始文件的标签列是num')
        d=read_table(path,schema)
        return schema,{'rows':len(d),'columns':list(d),'recognition':'known_cleveland_format','target':'num',
            'note':'采用已知数据字典；未使用大模型'}
    if path.suffix.lower() not in ['.csv','.tsv','.xlsx','.xls']:raise ValueError('未知无表头文件无法猜出字段含义；请提供schema，或使用有表头的CSV/Excel')
    d=read_table(path)
    candidates=[c for c in d if c.casefold() in TARGET_NAMES]
    if target is None:
        if len(candidates)!=1:
            raise ValueError(f'无法唯一确定预测目标，候选={candidates}；请用--target选择。实际列：{list(d)}')
        target=candidates[0]
    if target not in d:raise ValueError(f'没有目标列{target}，实际列：{list(d)}')
    label=d[target].replace({'':None,'?':None}).dropna()
    if label.nunique()>30:raise ValueError('目标超过30种取值；本入口支持分类，连续数值预测需另行指定回归任务')
    numeric=[];categorical=[];excluded={};details=[];nullable=[]
    for c in d:
        series=d[c].replace({'':None,'?':None,'NA':None,'nan':None})
        present=series.dropna();unique=present.nunique()
        reason=None;kind='excluded'
        norm=re.sub(r'\s+','',c.casefold())
        if c==target:reason='预测答案，仅用于训练/评估，不进入特征'
        elif c in candidates:reason='其他潜在标签列；自动流程排除以避免答案泄漏，可通过明确schema另行定义'
        elif norm in ID_NAMES or norm.startswith('unnamed:'):reason='标识或身份信息列'
        elif norm in TIME_NAMES:reason='时间字段需明确时序任务，本分类入口不自动作为数值特征'
        elif unique<2:reason='空列或常量列'
        else:
            numbers=pd.to_numeric(present,errors='coerce')
            ratio=float(numbers.notna().mean()) if len(present) else 0
            if ratio>=.95:
                numeric.append(c);kind='numeric'
            elif unique>50 and unique/max(len(present),1)>.5:reason='高基数字符串，可能是标识/自由文本；不盲目编码'
            else:categorical.append(c);kind='categorical'
        if reason:excluded[c]=reason
        elif series.isna().any():nullable.append(c)
        details.append({'column':c,'storage_type':kind,'missing':int(series.isna().sum()),'unique':int(unique),'reason':reason})
    if not numeric and not categorical:raise ValueError('没有可训练特征')
    schema={'task':f'table_classification:{target}','target':target,'numeric':numeric,'categorical':categorical,
        'allow_missing':nullable,'source':str(path),'recommendation_task':'unsupported',
        'interpretation':f'根据本表特征预测标签{target}；标签的临床含义需以数据说明为准，不自动解释为未来发病概率。',
        'discovery':'code_only'}
    known=set(d)
    if 'Diabetes_012'==target and {'HighBP','BMI','PhysActivity','GenHlth'}<=known:
        from .contracts import SURVEY
        schema.update(task='survey_diabetes_status',recommendation_task='survey',numeric=list(SURVEY),categorical=[],
            ranges={k:list(v) for k,v in SURVEY.items()},
            interpretation='CDC问卷糖尿病状态三分类：0无糖尿病或仅妊娠史，1前期，2糖尿病；不是未来发病概率。')
    group=next((c for c in d if c.casefold() in ['patient_id','subject_id','patientid','患者编号','患者id']),None)
    if group:schema['group_column']=group
    return schema,{'rows':len(d),'columns':details,'target':target,'target_values':sorted(label.astype(str).unique().tolist()),
        'excluded_columns':excluded,'group_column':group,'recognition':'known_survey' if schema.get('recommendation_task')=='survey' else 'inferred_types',
        'note':'行列及存储类型由代码识别；目标列可自动匹配常见名称或由用户指定。不会自动理解所有医学字段、单位或未来预测含义。'}
