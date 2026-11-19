"""Local CLI, independent of the HTTP API to be agreed with the team."""
from pathlib import Path
import argparse,json
from chronic.inference import ChronicPredictor

def main():
    p=argparse.ArgumentParser(description='离线慢病预测')
    p.add_argument('task',choices=['diabetes','glucose'])
    p.add_argument('input',type=Path)
    p.add_argument('--output',type=Path)
    args=p.parse_args()
    try:
        data=json.loads(args.input.read_text(encoding='utf-8-sig'))
        predictor=ChronicPredictor()
        result=predictor.diabetes(data) if args.task=='diabetes' else predictor.glucose(data)
        text=json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)
        if args.output:
            args.output.parent.mkdir(exist_ok=True,parents=True);args.output.write_text(text,encoding='utf8')
        print(text)
    except (ValueError,KeyError,TypeError,OSError) as e:p.exit(2,f'输入或模型错误：{e}\n')

if __name__=='__main__':main()
