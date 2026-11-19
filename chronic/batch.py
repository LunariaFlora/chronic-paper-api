"""Whole-table adapters: deterministic cleaning, vectorized inference and row-level reasons."""
from pathlib import Path
import json,hashlib
import numpy as np,pandas as pd,joblib
from threadpoolctl import threadpool_limits
from .contracts import SURVEY,MG_PER_MMOL,glucose_state
from .data import ROOT,features,forecast_values
from .recommendations import recommend,flatten_advice
from .table_io import read_table,mapped,write_results
from .table_models import prepare

def result_frame(n,path,header=True):
    return pd.DataFrame({'result_source_file':[str(path)]*n,'result_source_row':np.arange(n)+(2 if header else 1),
        'result_status':['pending']*n,'result_reason':['']*n})

def append_advice(result,task,contexts,predictions):
    # Cache shared advice branches rather than regenerate the same text for 250k identical contexts.
    cache={};items=[]
    for context,prediction in zip(contexts,predictions):
        key=json.dumps([context,prediction],sort_keys=True,default=str)
        if key not in cache:cache[key]=flatten_advice(recommend(task,context,prediction))
        items.append(cache[key])
    return pd.concat([result,pd.DataFrame(items)],axis=1)

def survey_batch(d,path,mapping=None,include_advice=True):
    x=mapped(d,mapping or {});result=result_frame(len(d),path)
    missing=set(SURVEY)-set(x)
    if missing:raise ValueError(f'问卷表缺少字段：{sorted(missing)}；可用--mapping指定列名对应，不能把真实年龄当年龄分组编码')
    numeric=pd.DataFrame(index=x.index);errors=pd.Series('',index=x.index,dtype=object)
    for c,(lo,hi) in SURVEY.items():
        v=pd.to_numeric(x[c],errors='coerce');numeric[c]=v
        bad=(~np.isfinite(v))|(~v.between(lo,hi))|(v%1!=0)
        errors.loc[bad]+=f'{c}需要{lo}至{hi}的问卷整数编码；'
    good=errors.eq('');result['result_status']=np.where(good,'ok','invalid_input');result['result_reason']=errors
    b=joblib.load(ROOT/'artifacts/v2/survey.joblib')
    contexts=[{} for _ in range(len(d))];predictions=[{} for _ in range(len(d))]
    ix=np.flatnonzero(good)
    with threadpool_limits(limits=4):
        probs=b['model'].predict_proba(numeric.loc[good]) if len(ix) else np.empty((0,3))
    for j,name in enumerate(['no_diabetes','prediabetes','diabetes']):result.loc[ix,'result_score_'+name]=probs[:,j]
    result.loc[ix,'result_class']=(probs*np.asarray(b['decision_weights'])).argmax(axis=1) if len(ix) else []
    flags=probs[:,2]>=b['threshold'];result['result_screening_flag']=pd.Series(pd.NA,index=result.index,dtype='boolean');result.loc[ix,'result_screening_flag']=flags
    result['result_scope']='健康问卷当前状态分类；前期分类能力弱；不是未来发病预测'
    for k,i in enumerate(ix):
        contexts[i]={c:float(numeric.at[i,c]) for c in ['PhysActivity','Veggies','Fruits','HighBP','HighChol']}
        predictions[i]={'positive':bool(flags[k])}
    if not include_advice:return result,{'profile':'survey','mapping':mapping or {},'model':'artifacts/v2/survey.joblib'}
    result=append_advice(result,'survey',contexts,predictions)
    # Never give individualized advice when the input record itself is invalid.
    result.loc[~good,'result_advice']='先补齐或修正问卷字段；本行没有运行模型或生成个体建议。'
    result.loc[~good,'result_advice_priority']='data_needed'
    for col in ['result_advice_reason','result_advice_sources']:result.loc[~good,col]=''
    return result,{'profile':'survey','mapping':mapping or {},'model':'artifacts/v2/survey.joblib'}

def resolve_column(d,mapping,key,aliases,required=True):
    if key in mapping:
        if mapping[key] not in d:raise ValueError(f'指定的{key}列不存在：{mapping[key]}')
        return mapping[key]
    matches=[c for c in d if c.strip().casefold() in [v.casefold() for v in aliases]]
    if len(matches)>1:raise ValueError(f'{key}存在多个候选列：{matches}；请明确映射')
    if not matches and required:raise ValueError(f'找不到{key}列；请通过--mapping指定')
    return matches[0] if matches else None

def glucose_batch(d,path,mapping=None,unit=None,timezone=None,single_patient=False,include_advice=True):
    mapping=mapping or {};result=result_frame(len(d),path)
    for h in ['30','60']:
        for suffix in ['mg_dl','mmol_l']:result[f'result_glucose_{h}min_{suffix}']=np.nan
        result[f'result_target_{h}min']=''
        for e in ['low','high']:
            result[f'result_{e}_{h}min_score']=np.nan
            result[f'result_{e}_{h}min_flag']=pd.Series(pd.NA,index=result.index,dtype='boolean')
    tc=resolve_column(d,mapping,'timestamp',['Date','timestamp','time','时间','测量时间'])
    vc=resolve_column(d,mapping,'glucose',['CGM (mg / dl)','CGM (mg/dl)','CGM','glucose','value','血糖'])
    pc=resolve_column(d,mapping,'patient_id',['patient_id','patient','患者ID','患者编号'],False)
    sc=resolve_column(d,mapping,'session_id',['session_id','visit_id','记录段'],False)
    uc=resolve_column(d,mapping,'unit',['unit','单位'],False)
    if not pc and not single_patient:raise ValueError('表里没有患者ID；确认为单人原始记录时用--single-patient，避免把不同人的记录拼接')
    if not unit and not uc:
        compact=vc.lower().replace(' ','')
        if 'mg/dl' in compact:unit='mg/dL'
        elif 'mmol/l' in compact:unit='mmol/L'
        else:raise ValueError('血糖单位不明确，需--unit mg/dL或mmol/L，不能根据数值猜测')
    if unit and unit not in ['mg/dL','mmol/L']:raise ValueError('单位必须为mg/dL或mmol/L')
    values=pd.to_numeric(d[vc],errors='coerce').to_numpy(dtype=float,copy=True)
    ts=[];row_errors=['']*len(d);localized=0
    for i,v in enumerate(d[tc]):
        try:
            if pd.isna(v) or str(v).strip()=='':raise ValueError('时间为空')
            if isinstance(v,(int,float)):raise ValueError('数值时间不能猜测为Excel日期')
            t=pd.Timestamp(v)
            if pd.isna(t):raise ValueError('时间无效')
            if t.tzinfo is None:
                if not timezone:raise ValueError('原表没有时区，需--timezone明确指定')
                t=t.tz_localize(timezone,ambiguous='raise',nonexistent='raise');localized+=1
            ts.append(t.tz_convert('UTC'))
        except (ValueError,TypeError,KeyError) as e:ts.append(pd.NaT);row_errors[i]+=f'时间无效：{e}；'
        u=str(d.iloc[i][uc]).strip() if uc else unit
        if u not in ['mg/dL','mmol/L']:row_errors[i]+='单位不支持；';values[i]=np.nan
        elif unit and u!=unit:row_errors[i]+='行单位与--unit冲突；';values[i]=np.nan
        elif u=='mmol/L':values[i]*=MG_PER_MMOL
        if not np.isfinite(values[i]) or not 20<=values[i]<=600:row_errors[i]+='血糖缺失、非数值或超出20至600 mg/dL；'
    patients=d[pc].fillna('').astype(str).str.strip() if pc else pd.Series([Path(path).stem]*len(d))
    sessions=d[sc].fillna('').astype(str).str.strip() if sc else pd.Series(['file']*len(d))
    result['result_patient_id']=patients;result['result_timestamp_utc']=[t.isoformat() if pd.notna(t) else '' for t in ts]
    result['result_current_mg_dl']=values
    metadata=pd.DataFrame({'patient':patients,'session':sessions,'timestamp':ts})
    windows=[];accepted=[]
    for _,group in metadata.groupby(['patient','session'],sort=False,dropna=False):
        ids=group.index.tolist()
        if group.patient.eq('').any() or (sc and group.session.eq('').any()):
            result.loc[ids,['result_status','result_reason']]=['invalid_identity','缺少患者或记录段标识，未混入其他患者']
            continue
        if group.timestamp.isna().any():
            # A row with unknown time cannot be silently removed and bridged over.
            result.loc[ids,['result_status','result_reason']]=['ambiguous_timeline','同一患者记录段内有无法定位的时间；修正后重跑']
            continue
        ordered=group.sort_values('timestamp',kind='stable').index.to_numpy()
        duplicates=group.timestamp.duplicated(keep=False)
        for j,i in enumerate(ordered):
            if row_errors[i]:result.loc[i,['result_status','result_reason']]=['invalid_input',row_errors[i]];continue
            if duplicates.loc[i]:
                result.loc[i,['result_status','result_reason']]=['duplicate_timestamp','同一患者/记录段存在重复时间，未擅自删除或平均'];continue
            if j<11:result.loc[i,['result_status','result_reason']]=['insufficient_history',f'该记录前仅有{j+1}条，需连续12条'];continue
            win=ordered[j-11:j+1]
            if any(row_errors[k] for k in win):result.loc[i,['result_status','result_reason']]=['invalid_window','过去12条包含无效测量，未插值或删除后拼接'];continue
            gaps=np.diff([ts[k].timestamp() for k in win])
            if not np.all(np.abs(gaps-900)<=1):result.loc[i,['result_status','result_reason']]=['irregular_interval','过去12条不是连续15分钟间隔，未跨缺测拼接'];continue
            windows.append(values[win]);accepted.append(i)
    b=joblib.load(ROOT/'artifacts/v2/glucose.joblib')
    risks=[{} for _ in range(len(d))]
    if windows:
        raw=np.round(np.asarray(windows),6);accepted=np.asarray(accepted)
        keep=((raw>=b['training_min'])&(raw<=b['training_max'])).all(axis=1)
        rejected=accepted[~keep];result.loc[rejected,['result_status','result_reason']]=['unsupported_training_range','输入超出模型训练观测范围，保留当前状态但不推断未来']
        ix=accepted[keep];raw=raw[keep];x=features(raw);current=raw[:,-1]
        local_times = [ts[i].tz_convert(timezone or 'Asia/Shanghai') for i in ix]
        minutes = [t.hour * 60 + t.minute for t in local_times]
        result.loc[ix,'result_status']='ok'
        with threadpool_limits(limits=4):
            for h,v in b['forecasters'].items():
                pp=forecast_values(v,x,current,minutes) if len(ix) else current
                result.loc[ix,f'result_glucose_{h}min_mg_dl']=pp
                result.loc[ix,f'result_glucose_{h}min_mmol_l']=pp/MG_PER_MMOL
                result.loc[ix,f'result_target_{h}min']=[(ts[i]+pd.Timedelta(minutes=int(h))).isoformat() for i in ix]
                for event in ['low','high']:
                    e=b['events'][h+'_'+event]
                    probs=e['model'].predict_proba(x)[:,1] if len(ix) else np.array([])
                    th=np.where((current>=70)&(current<=180),e['early_warning_threshold'],e['threshold'])
                    flag=probs>=th
                    result.loc[ix,f'result_{event}_{h}min_score']=probs
                    col=f'result_{event}_{h}min_flag';result[col]=pd.Series(pd.NA,index=result.index,dtype='boolean');result.loc[ix,col]=flag
                    for k,i in enumerate(ix):risks[i].setdefault(h,{})[event]=bool(flag[k])
    result['result_current_state']=[glucose_state(v) if np.isfinite(v) and 20<=v<=600 else 'invalid' for v in values]
    result['result_scope']='历史记录逐窗口血糖预测；同一行不使用任何未来数据；不预测未来发病'
    contexts=[{'current_mg_dl':float(v) if np.isfinite(v) and 20<=v<=600 else None} for v in values]
    if include_advice:result=append_advice(result,'glucose',contexts,[{'risks':r} for r in risks])
    return result,{'profile':'glucose','mapping':{'timestamp':tc,'glucose':vc,'patient_id':pc,'session_id':sc,'unit':uc},
        'explicit_unit':unit,'naive_time_timezone':timezone,'timezone_localized_rows':localized,
        'single_patient_assertion':single_patient,'model':'artifacts/v2/glucose.joblib',
        'advice_note':'即使前11条历史不足，仍可根据有效当前测量生成提示；不把提示当作模型预测'}

def custom_batch(d,path,bundle,mapping=None,include_advice=True):
    schema=dict(bundle['schema']);schema['mapping']={**schema.get('mapping',{}),**(mapping or {})}
    x,errors=prepare(d,schema);good=errors.eq('');ix=np.flatnonzero(good)
    result=result_frame(len(d),path,header=Path(path).suffix.lower()!='.data')
    result['result_status']=np.where(good,'ok','invalid_input');result['result_reason']=errors
    with threadpool_limits(limits=4):
        if len(ix):
            p=bundle['model'].predict_proba(x.loc[good]);result.loc[ix,'result_class']=np.asarray(bundle['classes'])[p.argmax(axis=1)]
            for j,c in enumerate(bundle['classes']):result.loc[ix,'result_score_'+str(c)]=p[:,j]
    result['result_scope']=schema['interpretation']
    result['result_score_note']='模型类别分数，未作临床概率校准；不是未来发病概率'
    imputed=[]
    for _,row in x.iterrows():imputed.append(','.join(c for c in schema.get('allow_missing',[]) if pd.isna(row[c])))
    result['result_imputed_columns']=imputed
    if include_advice:
        result=append_advice(result,schema.get('recommendation_task','unsupported'),[{}]*len(d),[{}]*len(d))
        result.loc[~good,'result_advice']='先修正输入，本行未生成预测或个体建议。';result.loc[~good,'result_advice_priority']='data_needed'
        for col in ['result_advice_reason','result_advice_sources']:result.loc[~good,col]=''
    return result,{'profile':schema['task'],'interpretation':schema['interpretation'],'mapping':mapping or {},'model_classes':bundle['classes']}

def run_batch(input_path,out,profile='auto',model=None,mapping=None,unit=None,timezone=None,single_patient=False,sheet=0,include_rule_advice=True):
    path=Path(input_path);out=Path(out)
    if out.exists():raise FileExistsError(f'输出目录已存在：{out}，请换新目录')
    files=sorted(p for p in path.iterdir() if p.suffix.lower() in ['.csv','.xlsx','.xls','.data'] and not p.name.startswith('~$')) if path.is_dir() else [path]
    if not files:raise ValueError('没有找到可读取的表格文件')
    bundle=joblib.load(model) if model else None
    if bundle and bundle.get('format_version')!=1:raise ValueError('表格模型版本不兼容')
    originals=[];results=[];summaries=[]
    for f in files:
        d=read_table(f,bundle['schema'] if bundle else {},sheet=sheet)
        chosen=profile
        try:
            if bundle:chosen='custom'
            if chosen=='auto':
                candidate=mapped(d,mapping or {})
                if set(SURVEY)<=set(candidate):chosen='survey'
                else:
                    # Auto only matches explicit known headers, never guesses clinical units/meaning.
                    resolve_column(d,mapping or {},'timestamp',['Date','timestamp','time','时间','测量时间'])
                    resolve_column(d,mapping or {},'glucose',['CGM (mg / dl)','CGM (mg/dl)','CGM','glucose','value','血糖'])
                    chosen='glucose'
            if chosen=='survey':r,s=survey_batch(d,f,mapping,include_rule_advice)
            elif chosen=='glucose':r,s=glucose_batch(d,f,mapping,unit,timezone,single_patient,include_rule_advice)
            elif chosen=='custom':r,s=custom_batch(d,f,bundle,mapping,include_rule_advice)
            else:raise ValueError(f'未知profile：{chosen}')
        except (ValueError,KeyError) as e:
            r=result_frame(len(d),f,header=f.suffix.lower()!='.data')
            r['result_status']='unsupported_schema';r['result_reason']=str(e)
            r['result_advice']='表格尚未匹配模型输入；保留原始行，需补充字段映射或训练对应任务。'
            r['result_advice_priority']='data_needed';r['result_advice_engine']='evidence_rules'
            s={'profile':chosen,'error':str(e)}
        r['result_source_sheet']=str(sheet) if f.suffix.lower() in ['.xlsx','.xls'] else ''
        if not include_rule_advice:
            r=r.drop(columns=[c for c in r if c.startswith('result_advice')])
        originals.append(d);results.append(r)
        s.update({'file':str(f),'sha256':hashlib.sha256(f.read_bytes()).hexdigest(),'rows':len(d)});summaries.append(s)
    summary={'files':summaries,'requested_advice_mode':'rules' if include_rule_advice else 'none',
        'note':'批量处理原始整表，保留全部行。输入内已有标签不参与预测。整表可能包含训练记录，不能把整表输出准确率当留出测试。'}
    return write_results(pd.concat(originals,ignore_index=True),pd.concat(results,ignore_index=True),out,summary)
