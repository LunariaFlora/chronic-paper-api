"""Explicit units, timestamps and survey encodings shared by CLI and tests."""
from datetime import datetime, timezone
import math
import numpy as np

VERSION='2.0.0'
MG_PER_MMOL=18.0182
LOOKBACK=12
LOW=70.0
HIGH=180.0
RULE_SOURCE='https://doi.org/10.2337/dc26-S006'
# BRFSS survey codes, not automatically inferred from patient measurements.
SURVEY={
    'HighBP':(0,1),'HighChol':(0,1),'CholCheck':(0,1),'BMI':(12,98),
    'Smoker':(0,1),'Stroke':(0,1),'HeartDiseaseorAttack':(0,1),
    'PhysActivity':(0,1),'Fruits':(0,1),'Veggies':(0,1),'HvyAlcoholConsump':(0,1),
    'AnyHealthcare':(0,1),'NoDocbcCost':(0,1),'GenHlth':(1,5),
    'MentHlth':(0,30),'PhysHlth':(0,30),'DiffWalk':(0,1),'Sex':(0,1),
    'Age':(1,13),'Education':(1,6),'Income':(1,8),
}

def number(x,name):
    if isinstance(x,bool) or not isinstance(x,(int,float)) or not math.isfinite(x):
        raise ValueError(f'{name}必须为有限数值')
    return float(x)

def timestamp(x):
    if not isinstance(x,str):raise ValueError('timestamp必须是带时区的ISO时间字符串')
    try:t=datetime.fromisoformat(x.replace('Z','+00:00'))
    except ValueError as e:raise ValueError('timestamp不是有效的ISO时间') from e
    if t.tzinfo is None:raise ValueError('timestamp必须包含时区，例如+08:00')
    return t.astimezone(timezone.utc)

def validate_glucose(payload):
    if not isinstance(payload,dict):raise ValueError('血糖输入必须是JSON对象')
    if set(payload)-{'unit','measurements','as_of'}:raise ValueError('血糖输入存在未知字段')
    unit=payload.get('unit')
    if unit not in ('mg/dL','mmol/L'):raise ValueError('unit必须明确指定mg/dL或mmol/L')
    ms=payload.get('measurements')
    if not isinstance(ms,list) or len(ms)!=LOOKBACK:raise ValueError('需要恰好12条按旧到新排列的测量')
    times=[];values=[]
    for i,m in enumerate(ms):
        if not isinstance(m,dict) or set(m)!={'timestamp','value'}:raise ValueError('每条测量必须且只能包含timestamp和value')
        times.append(timestamp(m['timestamp']))
        v=number(m['value'],f'measurements[{i}].value')
        if unit=='mmol/L':v*=MG_PER_MMOL
        # Supported operating envelope, not a medical normal range.
        if not 20<=v<=600:raise ValueError('数值超出本程序支持范围20至600 mg/dL，请核对单位或设备原始结果')
        values.append(round(v,6))
    gaps=np.diff([t.timestamp() for t in times])
    if not np.all(np.abs(gaps-900)<=1):raise ValueError('时间必须严格递增、无重复且连续间隔15分钟，不能缺测后拼接')
    if 'as_of' in payload:
        age=(timestamp(payload['as_of'])-times[-1]).total_seconds()
        if age<0:raise ValueError('最后测量时间不能晚于as_of')
        if age>1200:raise ValueError('最后测量距离as_of超过20分钟，不能作为实时输入')
    return np.asarray(values),times,unit

def validate_survey(features):
    if not isinstance(features,dict):raise ValueError('问卷输入必须是JSON对象')
    missing=set(SURVEY)-set(features);extra=set(features)-set(SURVEY)
    if missing or extra:raise ValueError(f'问卷字段不匹配，缺少{sorted(missing)}，多余{sorted(extra)}')
    row=[]
    for k,(lo,hi) in SURVEY.items():
        v=number(features[k],k)
        if v!=int(v) or not lo<=v<=hi:raise ValueError(f'{k}必须使用{lo}至{hi}的问卷整数编码')
        row.append(v)
    return np.asarray(row)

def glucose_state(value):
    if value<54:return 'very_low'
    if value<LOW:return 'low'
    if value>250:return 'very_high'
    if value>HIGH:return 'high'
    return 'within_reference_range'
