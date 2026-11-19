"""Two terminal workflows: use a saved model, or train a model from a table."""
from pathlib import Path
from datetime import datetime
import json,os
import pandas as pd
from .table_io import read_table
from .discovery import discover
from .selection import select_features,column_number
from .workflow import run_workflow
from .batch import run_batch
from .catalog import model_catalog,read_json
from .api_session import load_api_env,request_recommendations
from .model_match import schema_for,missing_columns
from .terminal_results import tabulate_preview,load_view,show_page,show_issues,show_advice,advice_path

ROOT=Path(__file__).resolve().parents[1]
APP=ROOT.parent if ROOT.name=='程序' else ROOT
INPUTS=APP/'导入数据'
OUTPUTS=APP/'结果'
EXTENSIONS={'.csv','.tsv','.xlsx','.xls','.data'}


def preview(source):
    config=read_json(ROOT/'configs/heart_cleveland.json') if source.name=='processed.cleveland.data' else None
    d=read_table(source,config)
    print(f'\n{source.name} | {len(d)}行、{len(d.columns)}列')
    rows=[['序号','列名','前3条数据']]
    for i,c in enumerate(d,1):rows.append([f'{i}.',c,' / '.join(str(v)[:40] for v in d[c].head(3))])
    print(tabulate_preview(rows));return d


def choose_file():
    INPUTS.mkdir(exist_ok=True)
    files=sorted(p for p in INPUTS.rglob('*') if p.is_file() and p.suffix.lower() in EXTENSIONS and not p.name.startswith('~$'))
    if not files and APP==ROOT:
        files=[p for p in [ROOT/'datasets/heart_disease/processed.cleveland.data',ROOT/'datasets/cdc_diabetes/diabetes_012_health_indicators_BRFSS2015.csv'] if p.exists()]
    choices=[];seen=set()
    for path in files:
        if path.parent.name in ['连续血糖','Shanghai_T2DM']:
            if path.parent in seen:continue
            seen.add(path.parent);choices.append(path.parent)
        else:choices.append(path)
    for i,path in enumerate(choices,1):
        try:label=str(path.relative_to(INPUTS))
        except ValueError:label=path.name
        print(f'{i}. {label}'+('（文件夹）' if path.is_dir() else ''))
    text=input('文件序号或完整路径（回车返回）：').strip().strip('"')
    if not text:return None
    if text.isdigit() and 1<=int(text)<=len(choices):return choices[int(text)-1]
    path=Path(text)
    if not path.exists():raise ValueError('找不到文件，请检查路径。')
    return path


def read_source(source):
    if source.is_dir():
        files=sorted(p for p in source.iterdir() if p.suffix.lower() in EXTENSIONS and not p.name.startswith('~$'))
        if not files:raise ValueError('文件夹中没有表格。')
        print(f'共{len(files)}个文件，预览第一份：');return preview(files[0])
    return preview(source)


def choose_model():
    options=model_catalog(ROOT,APP)
    for i,item in enumerate(options,1):print(f"{i}. {item['name']}")
    text=input('模型序号或模型文件路径（回车返回）：').strip().strip('"')
    if not text:return None
    if text.isdigit() and 1<=int(text)<=len(options):return options[int(text)-1]
    path=Path(text)
    if not path.is_file():raise ValueError('找不到模型文件。')
    folder=path.parent.parent if path.parent.name=='model' else path.parent
    return {'name':path.stem,'path':path,'kind':'table','folder':folder,'info':read_json(folder/'run_summary.json')}


def choose_matching(source,d,selected):
    missing=missing_columns(d,selected)
    if not missing:return selected
    print(f"\n这份数据不能用于“{selected['name']}”：缺少 {', '.join(missing)}。")
    options=[m for m in model_catalog(ROOT,APP) if not missing_columns(d,m)]
    if options:
        print('以下模型具备所需输入列：')
        for i,item in enumerate(options,1):print(f"{i}. {item['name']}")
        text=input('输入序号换用该模型；f重新选文件；回车返回：').strip().lower()
        if text.isdigit() and 1<=int(text)<=len(options):return options[int(text)-1]
    else:
        print('没有匹配模型。可在“训练模型”中用这份数据训练。')
        text=input('f重新选文件；回车返回：').strip().lower()
    return 'file' if text=='f' else None


def advice(selected,folder):
    if advice_path(folder):show_advice(folder);return
    load_api_env(APP)
    if not os.environ.get('KIMI_API_KEY'):
        print('未配置API，请按README填写.env。');return
    from .batch_advice import create_advice
    print('正在生成简短建议…',flush=True)
    if create_advice(selected,folder):show_advice(folder)
    else:print('找不到这份结果对应的原始输入，请先预测数据表。')


def view_results(selected,folder,path,saved=False):
    if path is None:
        print('暂无保存的预测记录。');frame=pd.DataFrame();issues=pd.DataFrame()
    else:
        schema=schema_for(selected) if selected['kind']!='glucose' else {}
        frame,issues=load_view(path,schema.get('target'),schema.get('label_map'))
    show_issues(issues)
    if not frame.empty:show_page(frame)
    while True:
        command=input(('1预测新数据 | ' if saved else '')+'a健康建议 | 回车返回：').strip().lower()
        if not command:return
        if command=='a':advice(selected,folder)
        elif command=='1' and saved:predict_flow(selected);return
        else:print('请输入菜单中的选项。')


def show_model(selected):
    info=selected['info'];print('\n'+selected['name'])
    if 'test_accuracy' in info:
        print(f"测试{info.get('test_rows','')}条 | 准确率 {info['test_accuracy']:.2%}")
        if 'majority_baseline_accuracy' in info:print(f"全部猜多数类的对照准确率 {info['majority_baseline_accuracy']:.2%}")
    if 'mae' in info:print('血糖平均绝对误差(mg/dL)：'+', '.join(f'{h}分钟 {v:.2f}' for h,v in info['mae'].items()))
    if selected['kind']=='survey':print('0=无糖尿病或仅妊娠史，1=前期，2=糖尿病。')
    folder=selected['folder']
    path=next((p for p in [folder/'测试集_真实值与预测.csv',folder/'model/test_predictions.csv'] if p.exists()),None)
    view_results(selected,folder,path,saved=True)


def predict_flow(selected):
    while True:
        source=choose_file()
        if source is None:return
        d=read_source(source)
        choice=choose_matching(source,d,selected)
        if choice=='file':continue
        if choice is None:return
        selected=choice;break
    kwargs={'include_rule_advice':False}
    if selected['kind']=='glucose':
        if input('每文件一人、15分钟间隔、mg/dL、Asia/Shanghai，符合？[y/N]：').strip().lower()!='y':return
        kwargs.update(profile='glucose',unit='mg/dL',timezone='Asia/Shanghai',single_patient=True)
    elif selected['kind']=='survey':kwargs.update(profile='survey')
    else:kwargs.update(model=selected['path'])
    out=OUTPUTS/('run_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    print(f"正在使用{selected['name']}预测…",flush=True)
    summary=run_batch(source,out,**kwargs)
    success=summary['status_counts'].get('ok',0)
    print(f"完成：{success}/{summary['input_rows']}条有效预测。",flush=True)
    view_results(selected,out,out/'results.csv')


def train_flow():
    source=choose_file()
    if source is None:return
    if source.is_dir():raise ValueError('训练请选择一张有答案列的表格。')
    d=preview(source)
    print('目标列（标签y）=要预测的结果；输入列（特征X）=预测时能获得的信息。')
    choice=input('目标列序号（回车自动识别）：').strip()
    target=column_number(choice,list(d)) if choice else None
    schema,_=discover(source,target)
    meanings={'class':'数据集分类标签','Diabetes_012':'糖尿病状态','HeartDiseaseorAttack':'心脏病或心梗史','Alopecia':'脱发','num':'心脏病状态'}
    target_name=schema['target'];meaning=meanings.get(target_name,'')
    print(f"本次预测：第{list(d).index(target_name)+1}列 {target_name}"+(f'（{meaning}）' if meaning else ''))
    candidates=schema['numeric']+schema.get('categorical',[])
    print('可用输入列序号：'+','.join(str(i) for i,c in enumerate(d,1) if c in candidates))
    choice=input('输入列序号，逗号分隔（如1,2,3；回车使用上述全部）：').strip()
    features=[column_number(v,list(d)) for v in choice.replace('，',',').split(',')] if choice else None
    select_features(schema,features)
    choice=input('训练方式：回车自动比较；1逻辑回归；2随机森林：').strip()
    strategies={'':'auto','1':'logistic','2':'random_forest'}
    if choice not in strategies:raise ValueError('训练方式请选择1、2或回车。')
    want=request_recommendations()
    out=OUTPUTS/('run_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    run_workflow(source,out,target,recommendations=want,feature_columns=features,verbose=False,algorithm=strategies[choice])
    selected={'name':source.stem,'kind':'table','path':out/'model/table_model.joblib','folder':out,'info':read_json(out/'run_summary.json')}
    print('模型已保存。以下是未参与训练的测试结果：')
    if want:show_advice(out)
    view_results(selected,out,out/'model/test_predictions.csv')


def main():
    load_api_env(APP)
    while True:
        try:
            print('\n慢病预测\n1 使用模型\n2 训练模型')
            mode=input('选择（回车退出）：').strip()
            if not mode:return
            if mode=='1':
                selected=choose_model()
                if selected is not None:show_model(selected)
            elif mode=='2':train_flow()
            else:print('请选择1或2。')
        except EOFError:return
        except (ValueError,KeyError,OSError,RuntimeError,TypeError) as e:
            print('无法处理：'+str(e))
