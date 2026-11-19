"""One entry: original table -> local train -> held-out evaluation -> batch results -> optional LLM advice."""
import argparse
from chronic.workflow import run_workflow

if __name__=='__main__':
    p=argparse.ArgumentParser(description='纯代码分类训练和预测全过程；仅建议阶段可选用Kimi')
    p.add_argument('input');p.add_argument('--output',required=True);p.add_argument('--target');p.add_argument('--schema')
    p.add_argument('--features',nargs='+',help='可选：明确选择输入列名；默认自动识别的全部特征')
    p.add_argument('--algorithm',choices=['auto','logistic','random_forest'],default='auto')
    p.add_argument('--recommendations',choices=['none','kimi'],default='none')
    p.add_argument('--advice-limit',type=int,default=3,choices=range(1,21),metavar='1..20')
    a=p.parse_args()
    try:run_workflow(a.input,a.output,a.target,a.schema,a.recommendations=='kimi',a.advice_limit,feature_columns=a.features,algorithm=a.algorithm)
    except (ValueError,KeyError,OSError,RuntimeError) as e:p.exit(2,f'流程未完成：{e}\n已完成的步骤保留在输出目录。\n')
