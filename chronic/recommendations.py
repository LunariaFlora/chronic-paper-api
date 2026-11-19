"""Versioned, source-linked health education rules. No LLM, diagnosis or dosing."""
import math

RULE_VERSION='1.0.0'
SOURCES={
 'low':'https://www.cdc.gov/diabetes/treatment/treatment-low-blood-sugar-hypoglycemia.html',
 'monitor':'https://www.cdc.gov/diabetes/treatment/index.html',
 'food':'https://www.cdc.gov/diabetes/healthy-eating/diabetes-meal-planning.html',
 'activity':'https://www.cdc.gov/diabetes/living-with/physical-activity.html',
 'heart':'https://www.nhlbi.nih.gov/health/heart-healthy-living',
}

def recommend(task, context, prediction=None):
    """Context must contain measured values or explicit questionnaire codes, never guessed values."""
    items=[]
    def add(code,priority,reason,text,source):
        items.append({'code':code,'priority':priority,'reason':reason,'text':text,'source':SOURCES[source]})
    if task=='glucose':
        value=context.get('current_mg_dl')
        if not isinstance(value,(float,int)) or not math.isfinite(value):
            return {'engine':'evidence_rules','version':RULE_VERSION,'priority':'data_needed','items':[],
                    'note':'当前血糖无有效数值，先核对数据；未生成个体健康建议。'}
        if value<70:
            add('measured_low','urgent',f'当前测量 {value:.1f} mg/dL，低于70',
                '先处理当前低血糖，遵循医生给你的低血糖处理方案并复测；若意识不清、抽搐或无法吞咽，立即寻求急救，不要口服喂食。','low')
            add('review_lows','follow_up','存在当前低血糖',
                '记录本次血糖、饮食、活动与用药情况，反复出现时联系医疗团队；不要自行改变药物剂量。','low')
            return {'engine':'evidence_rules','version':RULE_VERSION,'priority':'urgent','items':items,
                    'note':'以当前测量优先；本次不生成日常运动或控糖饮食计划。历史表格中的建议只针对记录当时。'}
        if value>180:
            add('measured_high','prompt_review',f'当前测量 {value:.1f} mg/dL，高于本实验180参考界限',
                '核对测量并结合进食时间及个人目标复测、记录；持续偏高时联系医疗团队，不根据本程序自行加药。','monitor')
        if value>=240:
            add('high_and_ill','prompt_review','当前血糖达到240 mg/dL；是否生病未知',
                '如果同时生病，按既定计划检查酮体；酮体偏高时及时联系医生。不能只等待模型后续预测。','monitor')
        risks=(prediction or {}).get('risks',{})
        if any(r.get('low',False) for r in risks.values()):
            add('forecast_low','monitor','模型标记未来窗口可能偏低，尚不等于实际低血糖',
                '增加对后续测量和症状的关注，按个人监测计划复测；未实测偏低时不要仅凭预测自动补糖或改药。','monitor')
        if any(r.get('high',False) for r in risks.values()):
            add('forecast_high','monitor','模型标记未来窗口可能偏高',
                '记录后续血糖及进餐、活动情况，供复诊分析；模型报警可能误报，先以实际测量核实。','monitor')
        add('glucose_record','routine','提供了连续血糖记录',
            '持续记录血糖、进餐和活动时间，和医疗团队商定适合自己的监测频率与目标。','monitor')
        add('regular_meals','routine','日常血糖管理；未提供个体饮食禁忌',
            '规律进餐，避免随意漏餐；日常餐食搭配非淀粉类蔬菜、蛋白质和适量主食，具体份量结合个人治疗计划。','food')
        if risks and 70<=value<=180 and not any(r.get('low',False) for r in risks.values()):
            add('activity_plan','routine','当前在实验参考区间内，未触发模型偏低标记',
                '在身体状况允许、医生认为适宜时从轻量活动开始，逐步增加；有活动限制者先商定适合自己的方式。','activity')
    elif task=='survey':
        if (prediction or {}).get('positive',False):
            add('screen_follow_up','follow_up','问卷模型触发糖尿病筛查标记，并非诊断',
                '把问卷和既往检查结果带给医疗人员，讨论是否需要进一步检查；不要据此自行确诊或用药。','monitor')
        add('survey_limit','routine','问卷不包含完整临床检查，前期分类能力弱',
            '模型未标记也不能排除糖尿病或前期状态；按既定检查和随访计划管理健康。','monitor')
        if context.get('PhysActivity')==0:
            add('inactive','routine','问卷报告近期没有体育活动',
                '身体条件允许时，从短时轻量活动开始逐步增加；先和医生确认适合的活动及限制。','activity')
        if context.get('Veggies')==0 or context.get('Fruits')==0:
            add('food_pattern','routine','问卷报告蔬菜或水果摄入频率不足',
                '检查日常餐食搭配，适量增加非淀粉类蔬菜；水果与其他碳水一起计入餐食安排，避免把果汁当作等量水果。','food')
        if context.get('HighBP')==1 or context.get('HighChol')==1:
            add('bp_lipids','follow_up','问卷报告高血压或高胆固醇史',
                '保留血压及血脂检查记录，按医疗团队计划复查；这项建议依据病史，不是模型预测了高血压。','heart')
    elif task=='heart':
        add('heart_review','follow_up','使用了心脏检查特征；模型结果只供研究参考',
            '把原始检查结果交由医生综合判断，按要求复诊；不要把模型阴性作为排除心脏病的依据。','heart')
        add('heart_habits','routine','心血管健康教育；个体活动限制未知',
            '结合医生意见管理血压、血脂和体重，选择适合的饮食与活动，并保持良好睡眠。','heart')
    else:
        return {'engine':'evidence_rules','version':RULE_VERSION,'priority':'not_supported','items':[],
                'note':'这个新任务尚无匹配的健康建议规则；没有把标签自动扩写成治疗方案。'}
    priority=next((p for p in ['urgent','prompt_review','follow_up','monitor','routine'] if any(i['priority']==p for i in items)),'routine')
    return {'engine':'evidence_rules','version':RULE_VERSION,'priority':priority,'items':items,
            'note':'一般健康教育与随访提示；未生成药物处方，不代替个体诊疗。历史记录的建议针对当时情况。'}

def flatten_advice(advice):
    return {'result_advice_engine':advice['engine'],'result_advice_version':advice['version'],
        'result_advice_priority':advice['priority'],
        'result_advice':'；'.join(i['text'] for i in advice['items']) or advice['note'],
        'result_advice_reason':'；'.join(i['reason'] for i in advice['items']),
        'result_advice_sources':' | '.join(dict.fromkeys(i['source'] for i in advice['items'])),
        'result_advice_note':advice['note']}
