"""Read original tables without modifying them; retain every source row."""
from pathlib import Path
import json,csv
import pandas as pd

def read_table(path,config=None,sheet=0):
    path=Path(path);config=config or {}
    if path.suffix.lower() in ('.xls','.xlsx'):
        d=pd.read_excel(path,sheet_name=sheet,dtype=object)
    else:
        opts={'dtype':object,'keep_default_na':False,'sep':config.get('separator',None),'engine':'python'}
        if config.get('columns') and (path.suffix.lower()=='.data' or config.get('header') is False):
            opts.update(header=None,names=config['columns'])
        for enc in ['utf-8-sig','gb18030']:
            try:
                if not config.get('separator'):
                    with path.open(encoding=enc) as f:sample=f.read(8192)
                    try:opts['sep']=csv.Sniffer().sniff(sample,delimiters=',;\t|').delimiter
                    except csv.Error:opts['sep']=','
                d=pd.read_csv(path,encoding=enc,**opts);break
            except UnicodeDecodeError:
                if enc=='gb18030':raise
    d.columns=[str(c).strip() for c in d.columns]
    if d.columns.duplicated().any():raise ValueError('清理空格后存在重复列名，不能猜测列对应关系')
    if any(c.startswith('result_') for c in d):raise ValueError('输入已含result_结果列；请使用原始表格，避免覆盖原结果')
    return d.reset_index(drop=True)

def mapped(d,mapping):
    missing=[v for v in mapping.values() if v not in d]
    if missing:raise ValueError(f'映射指定的原始列不存在：{missing}')
    if len(set(mapping.values()))!=len(mapping):raise ValueError('多个模型字段不能映射为同一原始列')
    return d.rename(columns={v:k for k,v in mapping.items()}).copy()

def write_results(original,result,out,summary):
    out=Path(out)
    if out.exists():raise FileExistsError(f'输出目录已存在：{out}；请换一个目录以保留旧结果')
    out.mkdir(parents=True)
    if len(original)!=len(result):raise RuntimeError('输入和结果行数不一致')
    joined=pd.concat([original.reset_index(drop=True),result.reset_index(drop=True)],axis=1)
    # Escape executable spreadsheet formulas in string cells, preserving numerical values.
    escaped=0
    for c in joined.select_dtypes(include=['object','str']).columns:
        def safe(v):
            nonlocal escaped
            if isinstance(v,str) and v.lstrip().startswith(('=','+','-','@')):
                escaped+=1;return "'"+v
            return v
        joined[c]=joined[c].map(safe)
    joined.to_csv(out/'results.csv',index=False,encoding='utf-8-sig')
    joined[joined['result_status']!='ok'].to_csv(out/'issues.csv',index=False,encoding='utf-8-sig')
    summary.update({'input_rows':len(original),'output_rows':len(joined),
        'status_counts':joined['result_status'].value_counts().to_dict(),
        'spreadsheet_formula_cells_escaped':escaped,
        'mode':'historical_batch','recommendation_engine':'none' if summary.get('requested_advice_mode')=='none' else 'evidence_rules',
        'recommendation_version':None if summary.get('requested_advice_mode')=='none' else '1.0.0'})
    (out/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2,default=str),encoding='utf8')
    return summary
