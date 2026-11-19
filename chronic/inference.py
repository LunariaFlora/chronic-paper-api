from pathlib import Path
from datetime import timedelta
import os
from zoneinfo import ZoneInfo
import joblib,numpy as np,pandas as pd
from threadpoolctl import threadpool_limits
from .contracts import VERSION,MG_PER_MMOL,SURVEY,LOW,HIGH,RULE_SOURCE,validate_glucose,validate_survey,glucose_state
from .data import ROOT,features,forecast_values

class ChronicPredictor:
    def __init__(self,model_dir=None):
        self.model_dir=Path(model_dir) if model_dir else ROOT/'artifacts/v2'
        self._cache={}

    def _load(self,name):
        if name not in self._cache:
            # joblib is executable serialization: only load our locally trained artifacts.
            b=joblib.load(self.model_dir/f'{name}.joblib')
            if b.get('version')!=VERSION:raise ValueError('模型与代码版本不兼容')
            self._cache[name]=b
        return self._cache[name]

    def diabetes(self,features):
        row=validate_survey(features);b=self._load('survey')
        with threadpool_limits(limits=4):
            p=b['model'].predict_proba(pd.DataFrame([row],columns=list(SURVEY)))[0]
        labels=['no_diabetes_or_gestational_only','prediabetes','diabetes']
        predicted=int((p*np.asarray(b['decision_weights'])).argmax())
        return {'task':'survey_diabetes_status','model_version':VERSION,
            'predicted_class':predicted,'predicted_label':labels[predicted],
            'class_scores':{label:float(v) for label,v in zip(labels,p)},
            'decision_weights':b['decision_weights'],
            'diabetes_screening_flag':bool(p[2]>=b['threshold']),
            'flag_threshold':float(b['threshold']),
            'limitations':['糖尿病前期类别样本少且区分能力差，不能依赖其分类标签筛除或确认前期状态',
                '健康问卷模型不能由血糖或血压单项测量替代输入'],
            'scope':'BRFSS健康问卷状态分类；包含独立的糖尿病前期类别',
            'interpretation':'分数是样本分布上的校准估计，不是未来发病概率；筛查标记与三分类采用不同决策规则，不等于诊断'}

    def glucose(self,payload):
        raw,times,unit=validate_glucose(payload);b=self._load('glucose')
        current=float(raw[-1]);elapsed=(times[-1]-times[0]).total_seconds()/60
        slope=float(np.polyfit(np.arange(12)*15,raw,1)[0]*60)
        warnings=[]
        outside=bool(np.any(raw<b['training_min']) or np.any(raw>b['training_max']))
        if outside:warnings.append('输入超出训练数据的观测范围，保留当前指标分析，但不提供学习模型预测')
        if 'as_of' not in payload:warnings.append('未提供as_of：按历史离线序列处理，未核验实时新鲜度')
        if np.max(np.abs(np.diff(raw)))>100:warnings.append('相邻测量变化较大，请核对原始设备数据；未自动平滑或修改原值')
        result={'task':'glucose_forecast_and_risk','model_version':VERSION,
            'status':'unsupported_training_range' if outside else 'ok',
            'measurement_unit':unit,'internal_unit':'mg/dL','interval_minutes':15,
            'current':{'timestamp':times[-1].isoformat(),'mg_dl':current,'mmol_l':current/MG_PER_MMOL,
                'state':glucose_state(current)},
            'trend':{'window_minutes':elapsed,'change_mg_dl':float(raw[-1]-raw[0]),
                'linear_slope_mg_dl_per_hour':slope,'mean_mg_dl':float(raw.mean()),'std_mg_dl':float(raw.std())},
            'forecasts':{},'risk_reference':{'low_below_mg_dl':LOW,'high_above_mg_dl':HIGH,'source':RULE_SOURCE},
            'warnings':warnings,'scope':'上海2型糖尿病队列训练；预测短期血糖与越界事件，不预测未来是否患糖尿病'}
        x=features(raw)
        local_time = times[-1].astimezone(ZoneInfo(os.environ.get('CHRONIC_TIMEZONE', 'Asia/Shanghai')))
        minutes = [local_time.hour * 60 + local_time.minute]
        with threadpool_limits(limits=4):
            for horizon,v in b['forecasters'].items():
                h=int(horizon)
                entry={'target_timestamp':(times[-1]+timedelta(minutes=h)).isoformat(),
                    'point_mg_dl':None,'point_mmol_l':None,'residual_band_mg_dl':None,
                    'risks':{},'unavailable_reason':'outside_training_range' if outside else None}
                if not outside:
                    point=float(forecast_values(v,x,np.asarray([current]),minutes)[0])
                    entry.update({'point_mg_dl':point,'point_mmol_l':point/MG_PER_MMOL,
                        'point_state':glucose_state(point),
                        'residual_band_mg_dl':[max(0,point-v['interval_half_width']),point+v['interval_half_width']],
                        'band_note':'校准患者残差构成的经验区间，不是个体临床保证'})
                    for event in ['low','high']:
                        e=b['events'][f'{horizon}_{event}'];p=float(e['model'].predict_proba(x)[0,1])
                        already=current<LOW if event=='low' else current>HIGH
                        in_range=LOW<=current<=HIGH
                        threshold=e['early_warning_threshold'] if in_range else e['threshold']
                        entry['risks'][event]={'model_score':p,'threshold':threshold,
                            'decision_mode':'early_warning' if in_range else 'continuation_or_crossing',
                            'flag':bool(p>=threshold),'already_present':bool(already),
                            'event_definition':f'未来(0,{h}]分钟的采样点中至少一次'+('<70 mg/dL' if event=='low' else '>180 mg/dL')}
                result['forecasts'][horizon]=entry
        result['interpretation']='当前状态、未来点预测、未来窗口越界风险分别输出；未触发标记不等于没有风险，不自动调整用药'
        return result
