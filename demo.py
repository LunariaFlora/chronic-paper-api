"""Run the completed local prediction flow against real held-out examples."""
import json
from pathlib import Path
from chronic.inference import ChronicPredictor

ROOT=Path(__file__).resolve().parent
ART=ROOT/'artifacts/v2'

if __name__=='__main__':
    p=ChronicPredictor();out=ROOT/'outputs/completion';out.mkdir(parents=True,exist_ok=True)
    results={}
    results['survey']={'input':json.loads((ART/'survey_input.json').read_text(encoding='utf8')),
        'truth':json.loads((ART/'survey_truth.json').read_text(encoding='utf8'))}
    results['survey']['output']=p.diabetes(results['survey']['input'])
    truth=json.loads((ART/'glucose_example_truth.json').read_text(encoding='utf8'))
    for name,t in truth.items():
        payload=json.loads((ART/t['input']).read_text(encoding='utf8'));pred=p.glucose(payload)
        results[name]={'input':payload,'output':pred,'truth':t}
        print(f"{name}: current={pred['current']['mg_dl']:.1f} mg/dL")
        for h,v in pred['forecasts'].items():
            flags=[k for k,z in v['risks'].items() if z['flag']]
            print(f"  +{h} min: {v['point_mg_dl']:.1f} mg/dL; flags={flags}")
    (out/'demo_results.json').write_text(json.dumps(results,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf8')
    print('问卷三分类结果:',results['survey']['output']['predicted_label'])
    print('完整输入、预测及真实值已保存到 outputs/completion/demo_results.json')
