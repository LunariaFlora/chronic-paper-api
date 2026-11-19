"""Train from an original whole table and an explicit task schema."""
import argparse,json
from chronic.table_models import train_table

if __name__=='__main__':
    p=argparse.ArgumentParser(description='用原始整表训练分类模型；字段与标签由schema明确指定')
    p.add_argument('input');p.add_argument('--schema',required=True);p.add_argument('--output',required=True)
    a=p.parse_args()
    try:
        r=train_table(a.input,a.schema,a.output)
        print(json.dumps({k:r[k] for k in ['task','rows','valid_rows','selected_model','partitions','test']},ensure_ascii=False,indent=2))
    except (ValueError,KeyError,OSError) as e:p.exit(2,f'训练未完成：{e}\n')
