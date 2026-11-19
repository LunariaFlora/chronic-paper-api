"""Reject incompatible tables before a batch run, and offer matching local models."""
from functools import lru_cache
import joblib
from .contracts import SURVEY
from .batch import resolve_column


@lru_cache(maxsize=12)
def bundle(path,mtime):return joblib.load(path)


def schema_for(selected):
    if selected['kind']=='table':
        p=selected['path'];return bundle(str(p),p.stat().st_mtime_ns)['schema']
    if selected['kind']=='survey':return {'numeric':list(SURVEY),'categorical':[],'target':'Diabetes_012','task':'survey_diabetes_status',
        'interpretation':'CDC健康问卷当前糖尿病状态；0无糖尿病或仅妊娠史，1前期，2糖尿病。Age为年龄分组编码。'}
    return {}


def missing_columns(frame,selected):
    if selected['kind']=='glucose':
        missing=[]
        for key,aliases in [('timestamp',['Date','timestamp','time','时间','测量时间']),('glucose',['CGM (mg / dl)','CGM (mg/dl)','CGM','glucose','value','血糖'])]:
            try:resolve_column(frame,{},key,aliases)
            except ValueError:missing.append('时间列' if key=='timestamp' else '血糖测量列')
        return missing
    s=schema_for(selected)
    return [s.get('mapping',{}).get(c,c) for c in s['numeric']+s.get('categorical',[]) if s.get('mapping',{}).get(c,c) not in frame]
