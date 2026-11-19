"""Small interactive launcher for users who want to hand the program a whole table."""
from pathlib import Path
from datetime import datetime
import json,sys
from chronic.batch import run_batch

ROOT=Path(__file__).resolve().parent

if __name__=='__main__':
    print('整表预测与健康建议：原始表格会保留，结果写到新目录。')
    print('1 = 健康问卷表；2 = 连续血糖表/文件夹；3 = Cleveland心脏病检查表')
    mode=input('选择处理方式 [1/2/3]：').strip()
    defaults={'1':ROOT/'datasets/cdc_diabetes/diabetes_012_health_indicators_BRFSS2015.csv',
        '2':ROOT/'datasets/shanghai_cgm/Shanghai_T2DM',
        '3':ROOT/'datasets/heart_disease/processed.cleveland.data'}
    if mode not in defaults:raise SystemExit('请选择1、2或3')
    raw=input(f'粘贴原始文件或文件夹路径（直接回车使用 {defaults[mode]}）：').strip().strip('"')
    source=Path(raw) if raw else defaults[mode]
    kwargs={}
    if mode=='1':kwargs['profile']='survey'
    if mode=='2':
        kwargs['profile']='glucose'
        if not raw:
            kwargs.update(unit='mg/dL',timezone='Asia/Shanghai',single_patient=True)
            print('上海原始数据适配：mg/dL，Asia/Shanghai，每个文件是一人的一次记录。')
        else:
            kwargs['unit']=input('原表血糖单位 [mg/dL 或 mmol/L，留空仅使用表中明确单位]：').strip() or None
            kwargs['timezone']=input('无时区时间的实际时区 [如 Asia/Shanghai；已有时区可留空]：').strip() or None
            kwargs['single_patient']=input('每个文件是否只含一个人的一次记录？[y/N，有患者ID列可选N]：').strip().lower()=='y'
    if mode=='3':kwargs['model']=ROOT/'artifacts/table_heart_v1/table_model.joblib'
    output=ROOT/'outputs'/('table_run_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    try:
        r=run_batch(source,output,**kwargs)
        print(json.dumps(r['status_counts'],ensure_ascii=False,indent=2))
        print(f'结果表：{output / "results.csv"}\n问题行：{output / "issues.csv"}')
    except (ValueError,OSError,KeyError) as e:raise SystemExit(f'未完成：{e}')
