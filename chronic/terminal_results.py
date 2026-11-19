"""Readable, paged results in the terminal; saved CSVs remain export copies."""
from pathlib import Path
import json,unicodedata
import numpy as np,pandas as pd


def tabulate_preview(rows):
    rows=[[str(c).replace('\t',' ').replace('\r',' ').replace('\n',' ') for c in row] for row in rows]
    if not rows:return ''
    def width(s):return sum(0 if unicodedata.combining(c) else 2 if unicodedata.east_asian_width(c) in ('W','F') else 1 for c in s)
    widths=[(max(width(row[i]) for row in rows)//8+1)*8 for i in range(len(rows[0])-1)]
    return '\n'.join(''.join(c+'\t'*(widths[i]//8-width(c)//8) for i,c in enumerate(row[:-1]))+row[-1] for row in rows)


def value(v):
    if pd.isna(v):return '—'
    if isinstance(v,(float,np.floating)):return f'{v:.2f}'.rstrip('0').rstrip('.')
    if isinstance(v,(bool,np.bool_)):return '是' if v else '否'
    return str(v)


def load_view(path,target=None,label_map=None):
    columns=pd.read_csv(path,nrows=0).columns
    allowed={'source_row','actual','predicted','correct','patient','horizon','prediction','event','actual_event','event_flag',
             'result_source_row','result_class','result_status','result_reason','result_current_mg_dl',
             'result_glucose_30min_mg_dl','result_glucose_60min_mg_dl',target}
    d=pd.read_csv(path,usecols=lambda c:c in allowed)
    issues=pd.DataFrame()
    if 'result_status' in d:
        bad=d.result_status.ne('ok');issues=d.loc[bad,['result_source_row','result_reason']].rename(columns={'result_source_row':'行号','result_reason':'原因'})
        d=d.loc[~bad].copy()
        result=pd.DataFrame({'行号':d.result_source_row})
        if 'result_class' in d:
            result['预测']=d.result_class
            if target in d:
                truth=d[target]
                if label_map:
                    truth=truth.map(lambda x:label_map.get(str(int(float(x))) if pd.notna(x) and str(x).replace('.','',1).isdigit() else str(x),np.nan))
                result['真实']=truth
                result['正确']=[value(a)==value(b) if pd.notna(a) else None for a,b in zip(result['预测'],truth)]
        else:
            for col,title in [('result_current_mg_dl','当前血糖'),('result_glucose_30min_mg_dl','30分钟预测'),('result_glucose_60min_mg_dl','60分钟预测')]:
                if col in d:result[title]=d[col]
    elif 'predicted' in d:
        result=d[['source_row','actual','predicted']].rename(columns={'source_row':'记录','actual':'真实','predicted':'预测'})
        result['正确']=d.correct if 'correct' in d else d.actual==d.predicted
    else:
        event=d.event.fillna('') if 'event' in d else pd.Series('',index=d.index)
        result=pd.DataFrame({'记录':np.arange(len(d))+1,'提前(分)':d.horizon,'项目':event.replace({'':'血糖','low':'低血糖事件','high':'高血糖事件'}),
            '真实':d.actual.where(event.eq(''),d.get('actual_event')),'预测':d.prediction.where(event.eq(''),d.get('event_flag'))})
    return result.reset_index(drop=True),issues.reset_index(drop=True)


def show_page(frame,start=0,size=8):
    if frame.empty:print('没有可显示的预测。');return
    part=frame.iloc[start:start+size]
    print(tabulate_preview([list(frame.columns)]+[[value(v) for v in row] for row in part.itertuples(index=False,name=None)]))
    if len(frame)>len(part):print(f'…… 其余{len(frame)-len(part)}条省略，共{len(frame)}条。')


def show_issues(issues):
    if issues.empty:return
    print(f'另有{len(issues)}条未预测：')
    for reason,count in issues['原因'].fillna('未说明').value_counts().head(4).items():print(f'  {count}条：{reason}')


def advice_path(folder):
    folder=Path(folder)
    return next((p for p in [folder/'advice/recommendations.json',folder/'健康建议.json'] if p.exists()),None)


def show_advice(folder):
    path=advice_path(folder)
    if path is None:print('暂无健康建议。');return False
    for item in json.loads(path.read_text(encoding='utf-8-sig')):
        print(f"\n记录 {item.get('source_row','')}：")
        if item.get('status')!='ok':print(item.get('error','建议未生成'));continue
        response=item.get('response',{}).get('advice',{})
        for advice in response.get('recommendations',[])[:2]:print('  • '+advice.get('text',''))
    return True
