"""Discover bundled research results and models trained locally, without a fixed registry."""
from pathlib import Path
import json


def read_json(path):
    try:return json.loads(Path(path).read_text(encoding='utf-8-sig'))
    except (OSError,ValueError):return {}


def model_catalog(program,app):
    program=Path(program);app=Path(app);items=[]
    known=[('heart','心脏病状态','table',program/'artifacts/table_heart_v1/table_model.joblib'),
           ('survey','糖尿病问卷状态','survey',program/'artifacts/v2/survey.joblib'),
           ('glucose','连续血糖30/60分钟','glucose',program/'artifacts/v2/glucose.joblib')]
    for key,name,kind,path in known:
        if path.exists():
            gallery=app/'已保存结果'
            if not gallery.exists():gallery=app/'成果样例'
            folder=gallery/key
            items.append({'name':name,'origin':'bundled sample','path':path,'kind':kind,'folder':folder,
                          'info':read_json(folder/'模型信息.json')})
    for path in sorted((app/'结果').glob('*/model/table_model.joblib')):
        folder=path.parent.parent;report=read_json(folder/'run_summary.json')
        if not report:continue
        items.append({'name':f"{Path(report.get('input','')).stem} → {report.get('target','')} · {folder.name}",
                      'origin':'本机新训练','path':path,'kind':'table','folder':folder,'info':report})
    return items
