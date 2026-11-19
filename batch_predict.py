"""Original table -> automatic preparation -> predictions + grounded advice -> CSV."""
import argparse,json
from pathlib import Path
from chronic.batch import run_batch

if __name__=='__main__':
    p=argparse.ArgumentParser(description='原始整表批量预测及健康建议；输出results.csv、issues.csv和summary.json')
    p.add_argument('input');p.add_argument('--output',required=True)
    p.add_argument('--profile',choices=['auto','survey','glucose'],default='auto')
    p.add_argument('--model',help='只加载可信的本地train_table生成的table_model.joblib')
    p.add_argument('--mapping',help='JSON：模型字段名到原始列名的映射')
    p.add_argument('--unit',choices=['mg/dL','mmol/L']);p.add_argument('--timezone')
    p.add_argument('--single-patient',action='store_true',help='明确每个输入文件只含一个人的一次连续记录')
    p.add_argument('--sheet',default='0',help='Excel工作表名称或从0开始的序号；默认第一张')
    a=p.parse_args()
    try:
        mapping=json.loads(Path(a.mapping).read_text(encoding='utf-8-sig')) if a.mapping else None
        sheet=int(a.sheet) if a.sheet.isdigit() else a.sheet
        r=run_batch(a.input,a.output,a.profile,a.model,mapping,a.unit,a.timezone,a.single_patient,sheet)
        print(json.dumps({k:r[k] for k in ['input_rows','output_rows','status_counts','recommendation_engine']},ensure_ascii=False,indent=2))
        print(f'结果已写入：{Path(a.output).resolve()}')
        if r['input_rows'] and not r['status_counts'].get('ok',0):
            p.exit(3,'没有记录满足模型预测条件；原始行和原因已保存在结果表中。\n')
    except (ValueError,KeyError,OSError,TypeError) as e:p.exit(2,f'批量处理未完成：{e}\n')
