"""Interactive full-training workflow; inference-only launcher remains separate."""
from pathlib import Path
from datetime import datetime
from chronic.workflow import run_workflow
from chronic.discovery import discover

ROOT=Path(__file__).resolve().parent
if __name__=='__main__':
    print('完整实操：导入整表 -> 本地整理/训练 -> 留出测试 -> 整表预测 -> 可选大模型建议')
    print('1 = UCI心脏病原始表（303行，较快）；2 = Kaggle/CDC问卷（253680行，需等待训练）；3 = 自己的有标签CSV/Excel')
    mode=input('选择 [1/2/3]：').strip()
    paths={'1':ROOT/'datasets/heart_disease/processed.cleveland.data','2':ROOT/'datasets/cdc_diabetes/diabetes_012_health_indicators_BRFSS2015.csv'}
    if mode in paths:source=paths[mode];target=None
    elif mode=='3':
        source=Path(input('原始表格路径：').strip().strip('"'))
        target=input('要预测的标签列名（留空由常见列名识别）：').strip() or None
    else:raise SystemExit('请选择1、2或3')
    try:
        schema,inspection=discover(source,target)
        print(f"识别到预测目标：{schema['target']}；数值字段{len(schema['numeric'])}个，分类字段{len(schema.get('categorical',[]))}个。")
        print('训练和预测在本机运行，不调用大模型。')
        want=input('预测完成后，是否仅为前3条测试记录调用Kimi生成建议？[y/N]：').strip().lower()=='y'
        out=ROOT/'outputs'/('workflow_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
        run_workflow(source,out,target,recommendations=want)
        print(f'完成。打开这个文件夹查看全过程：{out}')
        print('先看 model/test_predictions.csv 的真实值和预测值，再看 predictions/results.csv。')
    except (ValueError,KeyError,OSError,RuntimeError) as e:raise SystemExit(f'未完成：{e}')
