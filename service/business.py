"""Patient-facing import and report contract, independent of the batch/CLI contract."""
import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from fastapi import HTTPException, UploadFile, File, Form, Query
from fastapi.responses import FileResponse
from .schemas import AnalysisInput, RecordInput
import logging

def jsonify(value=None, **values):
    return value if value is not None else values

def NotFound(message):
    return HTTPException(404, message)

def Conflict(message):
    return HTTPException(409, message)

def send_file(path, as_attachment=False, download_name=None):
    return FileResponse(path, filename=download_name)
from chronic.data import ROOT
from chronic.contracts import SURVEY
from chronic.table_io import read_table
from chronic.batch import run_batch
from chronic.recommendations import recommend

NAMES = {'glucose': '连续血糖分析', 'survey': '糖尿病健康问卷', 'heart': '心脏检查分析'}
HEART = json.loads((ROOT / 'configs/heart_cleveland.json').read_text(encoding='utf8'))


def now():
    return datetime.now(timezone.utc).isoformat()


def records(frame):
    return json.loads(frame.to_json(orient='records', force_ascii=False))


def inspect_file(path):
    d = read_table(path, HEART if path.suffix == '.data' else None)
    if not len(d):
        raise ValueError('文件没有数据行，请上传有测量记录的表格。')
    if len(d) > 100000:
        raise ValueError('单次最多导入100000条记录，请拆分文件。')
    columns = set(d)
    mapping = {}
    if set(SURVEY) <= columns:
        kind = 'survey'
    elif set(HEART['numeric'] + HEART['categorical']) <= columns:
        kind = 'heart'
    else:
        aliases = {
            'timestamp': ['timestamp', 'Date', 'time', '时间', '测量时间'],
            'glucose': ['glucose', 'value', '血糖', 'CGM', 'CGM (mg / dl)', 'CGM (mg/dl)',
                        '血糖(mmol/L)', '血糖（mmol/L）', 'glucose_mmol_l', 'glucose_mg_dl'],
            'patient_id': ['patient_id', 'patient', '患者ID', '患者编号'],
            'unit': ['unit', '单位'],
        }
        for key, names in aliases.items():
            hits = [c for c in d if c.casefold() in [n.casefold() for n in names]]
            if len(hits) > 1:
                raise ValueError(f'表中有多个{key}列，请保留本次要分析的一列。')
            if hits:
                mapping[key] = hits[0]
        if not {'timestamp', 'glucose'} <= mapping.keys():
            raise ValueError('尚未识别这份表格。血糖记录需包含“测量时间”和“血糖”列；心脏检查或健康问卷请使用对应模板。只有姓名、年龄或心率的数据不能用于现有模型。')
        kind = 'glucose'
        if 'patient_id' in mapping and d[mapping['patient_id']].astype(str).nunique() > 1:
            raise ValueError('这份记录包含多位用户，请按用户拆分后分别导入，避免把他人数据记到当前档案。')
    unit = None
    if kind == 'glucose':
        col = mapping['glucose'].lower().replace(' ', '')
        if 'mg/dl' in col or 'mg_dl' in col:
            unit = 'mg/dL'
        elif 'mmol/l' in col or 'mmol_l' in col:
            unit = 'mmol/L'
        elif 'unit' in mapping:
            values = d[mapping['unit']].dropna().astype(str).unique()
            if len(values) == 1 and values[0] in ['mg/dL', 'mmol/L']:
                unit = values[0]
    return d, {'kind': kind, 'name': NAMES[kind], 'rows': len(d), 'columns': list(d),
               'mapping': mapping, 'unit': unit, 'needs_unit': kind == 'glucose' and unit is None,
               'needs_row': kind != 'glucose' and len(d) > 1,
               'timezone': os.environ.get('CHRONIC_TIMEZONE', 'Asia/Shanghai')}


def make_report(folder, kind, profile):
    d = pd.read_csv(folder / 'output/results.csv')
    good = d[d.result_status == 'ok']
    latest = good.iloc[-1] if len(good) else d.iloc[-1]
    if kind == 'glucose':
        valid_time = d[d.result_timestamp_utc.notna()].sort_values('result_timestamp_utc')
        latest = valid_time.iloc[-1] if len(valid_time) else d.iloc[-1]
    r = json.loads(latest.to_json(force_ascii=False))
    context = {'current_mg_dl': r.get('result_current_mg_dl')} if kind == 'glucose' else {
        key: r.get(key) for key in ['PhysActivity', 'Veggies', 'Fruits', 'HighBP', 'HighChol']}
    prediction = {'positive': bool(r.get('result_screening_flag')), 'risks': {
        str(h): {event: bool(r.get(f'result_{event}_{h}min_flag')) for event in ['low', 'high']}
        for h in [30, 60] if r.get(f'result_glucose_{h}min_mmol_l') is not None}}
    advice = recommend(kind, context, prediction)
    report = {'schema_version': '2.0', 'kind': kind, 'title': NAMES[kind], 'profile_snapshot': profile,
              'record_count': len(d), 'predicted_count': len(good), 'created_at': now(),
              'scope': r.get('result_scope'), 'metrics': [], 'series': [], 'forecasts': [],
              'recommendations': advice['items'] if kind == 'glucose' or r.get('result_status') == 'ok' else [{'text': '先补齐或修正本次输入，再重新分析。'}],
              'timezone': os.environ.get('CHRONIC_TIMEZONE', 'Asia/Shanghai'),
              'recommendation_engine': r.get('result_advice_engine', 'evidence_rules'),
              'sources': str(r.get('result_advice_sources') or '').split(' | '),
              'issues': records(d.loc[d.result_status != 'ok', ['result_source_row', 'result_status', 'result_reason']].head(100)),
              'issue_count': int((d.result_status != 'ok').sum()),
              'status': 'completed' if len(good) else 'needs_data'}
    if kind == 'glucose':
        current = r.get('result_current_mg_dl')
        report['metrics'].append({'label': '最近一次血糖', 'value': round(current / 18.0182, 2) if current else None, 'unit': 'mmol/L'})
        report['observed_at'] = r.get('result_timestamp_utc')
        report['headline'] = '血糖趋势分析已完成' if r.get('result_status') == 'ok' else '已保存测量记录，最近时点暂不能预测'
        if r.get('result_status') != 'ok':
            report['detail'] = r.get('result_reason')
        timeline = d[['result_timestamp_utc', 'result_current_mg_dl']].dropna().sort_values('result_timestamp_utc').tail(2000)
        report['series'] = [{'time': x['result_timestamp_utc'], 'value': round(x['result_current_mg_dl'] / 18.0182, 2)} for x in records(timeline)]
        for h in (30, 60):
            v = r.get(f'result_glucose_{h}min_mmol_l')
            if v is not None:
                report['forecasts'].append({'minutes': h, 'value': round(v, 2), 'time': r.get(f'result_target_{h}min'), 'unit': 'mmol/L'})
                report['metrics'].append({'label': f'{h}分钟后预测', 'value': round(v, 2), 'unit': 'mmol/L'})
    else:
        label = r.get('result_class')
        labels = ['未标记糖尿病', '糖尿病前期类别', '糖尿病类别'] if kind == 'survey' else ['未标记心脏病状态', '标记心脏病状态']
        report['headline'] = labels[int(label)] if label is not None else '数据需要补充或修正'
        report['detail'] = r.get('result_reason') or '这是基于本次输入的模型分类结果。'
        report['metrics'] = [{'label': '本次分析', 'value': report['headline'], 'unit': ''}]
        report['class_scores'] = {k.removeprefix('result_score_'): v for k, v in r.items() if k.startswith('result_score_') and isinstance(v, (int, float))}
    return report


def register_business(app, base, sql, pool, require_user):
    sql('CREATE TABLE IF NOT EXISTS imports(id TEXT PRIMARY KEY,user_id TEXT NOT NULL,metadata TEXT NOT NULL,path TEXT NOT NULL)')
    sql('CREATE TABLE IF NOT EXISTS analyses(id TEXT PRIMARY KEY,user_id TEXT NOT NULL,import_id TEXT NOT NULL,status TEXT NOT NULL,created TEXT NOT NULL,request TEXT NOT NULL,report TEXT,error TEXT)')

    def imported(iid):
        r = sql('SELECT * FROM imports WHERE id=?', (iid,), True)
        if not r:
            raise NotFound('导入记录不存在')
        r['metadata'] = json.loads(r['metadata'])
        return r

    def analysis(aid):
        r = sql('SELECT * FROM analyses WHERE id=?', (aid,), True)
        if not r:
            raise NotFound('分析记录不存在')
        r.pop('request')
        r['report'] = json.loads(r['report']) if r['report'] else None
        return r

    def compute(aid):
        # Atomic claim prevents duplicate submissions within the single service instance.
        row = sql('UPDATE analyses SET status=? WHERE id=? AND status=? RETURNING *', ('running', aid, 'queued'), True)
        if not row:
            return
        try:
            args = json.loads(row['request'])
            imp = imported(row['import_id']); meta = imp['metadata']; kind = meta['kind']
            folder = base / 'analyses' / aid; folder.mkdir(parents=True, exist_ok=True)
            d = read_table(imp['path'], HEART if Path(imp['path']).suffix == '.data' else None)
            if kind != 'glucose':
                d = d.iloc[[args.get('row', 0)]]
            path = folder / 'input.csv'; d.to_csv(path, index=False)
            run_batch(path, folder / 'output', profile=kind if kind != 'heart' else 'auto',
                      model=ROOT / 'artifacts/table_heart_v1/table_model.joblib' if kind == 'heart' else None,
                      mapping=meta['mapping'], unit=args.get('unit') or meta['unit'],
                      timezone=meta['timezone'], single_patient=True)
            report = make_report(folder, kind, args['profile_snapshot'])
            engine = args.get('advice_engine', 'kimi' if os.environ.get('KIMI_API_KEY') else 'rules')
            if engine == 'kimi':
                from chronic.llm_advice import generate_advice
                try:
                    result = generate_advice({'task': kind, 'mode': 'historical_record',
                        'metrics': report['metrics'], 'forecasts': report['forecasts'],
                        'classification': report['headline'], 'scope': report['scope']})
                    report['recommendations'] = result['advice']['recommendations']
                    report['sources'] = list(dict.fromkeys(source for item in report['recommendations'] for source in item.get('sources', [])))
                    report['recommendation_engine'] = 'kimi_api'
                    report['recommendation_model'] = result['model']
                    report['advice_status'] = 'completed'
                except (RuntimeError, ValueError) as exc:
                    report['advice_status'] = 'failed'
                    report['advice_error'] = str(exc)
            else:
                report['advice_status'] = 'rules'
            sql('UPDATE analyses SET status=?,report=? WHERE id=?', (report['status'], json.dumps(report, ensure_ascii=False), aid))
        except Exception:
            logging.getLogger(__name__).exception('Analysis failed: %s', aid)
            sql('UPDATE analyses SET status=?,error=? WHERE id=?', ('failed', '分析未完成，请核对文件内容后重试；管理员可通过分析编号查看日志。', aid))

    @app.get('/healthz')
    def health():
        sql('SELECT 1')
        return jsonify(status='ok', service='chronic', version='2.0')

    @app.get('/api/v2/users', tags=['1. 用户档案'], summary='读取用户列表')
    def list_users():
        return jsonify(users=[dict(id=r['id'], **json.loads(r['data'])) for r in sql('SELECT * FROM users ORDER BY rowid DESC')])


    @app.post('/api/v2/imports', status_code=201, tags=['2. 导入数据'], summary='上传CSV或Excel，自动识别数据')
    def upload(user_id: str = Form(..., description='建立用户后返回的id'), file: UploadFile = File(..., description='该用户的CSV或Excel检查记录')):
        uid = user_id; require_user(uid)
        f = file
        if not f or Path(f.filename).suffix.lower() not in ['.csv', '.xlsx', '.xls', '.data']:
            raise ValueError('请选择CSV或Excel文件。')
        iid = uuid.uuid4().hex; folder = base / 'imports' / iid; folder.mkdir(parents=True)
        path = folder / ('source' + Path(f.filename).suffix.lower())
        size = 0
        with path.open('wb') as out:
            while chunk := f.file.read(1024 * 1024):
                size += len(chunk)
                if size > 32 * 1024 * 1024:
                    out.close(); path.unlink(); folder.rmdir()
                    raise HTTPException(413, '文件超过32MB')
                out.write(chunk)
        try:
            d, meta = inspect_file(path)
        except Exception:
            path.unlink(); folder.rmdir()
            raise
        meta.update(filename=Path(f.filename).name, created_at=now())
        sql('INSERT INTO imports VALUES(?,?,?,?)', (iid, uid, json.dumps(meta, ensure_ascii=False), str(path)))
        return jsonify(id=iid, user_id=uid, **meta, preview=records(d.head(5)))

    @app.get('/api/v2/imports/{iid}/rows', tags=['2. 导入数据'], summary='预览文件中的数据行')
    def import_rows(iid: str, offset: int = Query(0, ge=0, description="从第几行开始，0表示第一行")):
        imp = imported(iid); path = Path(imp['path'])
        d = read_table(path, HEART if path.suffix == '.data' else None)
        return jsonify(offset=offset, total=len(d), rows=records(d.iloc[offset:offset + 50]))

    @app.post('/api/v2/records', status_code=201, tags=['2. 导入数据'], summary='直接提交血糖JSON记录（实验功能）')
    def device_records(payload: RecordInput):
        args = payload.model_dump(exclude_none=True)
        if not isinstance(args, dict) or set(args) - {'user_id', 'unit', 'measurements'}:
            raise ValueError('设备记录支持 user_id、unit、measurements。')
        uid = args.get('user_id'); require_user(uid)
        unit = args.get('unit')
        if unit not in ['mmol/L', 'mg/dL']:
            raise ValueError('unit需要为mmol/L或mg/dL')
        rows = args.get('measurements')
        if not isinstance(rows, list) or not 1 <= len(rows) <= 100000:
            raise ValueError('measurements需要包含1至100000条测量')
        for row in rows:
            if not isinstance(row, dict) or set(row) != {'timestamp', 'value'}:
                raise ValueError('每条测量需要timestamp和value')
            from chronic.contracts import timestamp, number
            timestamp(row['timestamp']); number(row['value'], 'value')
        iid = uuid.uuid4().hex; folder = base / 'imports' / iid; folder.mkdir(parents=True)
        path = folder / 'source.csv'
        pd.DataFrame(rows).rename(columns={'value': 'glucose'}).assign(unit=unit).to_csv(path, index=False)
        _, meta = inspect_file(path); meta.update(filename='设备测量记录', created_at=now())
        sql('INSERT INTO imports VALUES(?,?,?,?)', (iid, uid, json.dumps(meta, ensure_ascii=False), str(path)))
        return jsonify(id=iid, user_id=uid, **meta)

    @app.post('/api/v2/analyses', status_code=202, tags=['3. 模型计算与结果'], summary='提交分析，返回分析编号')
    def submit(payload: AnalysisInput):
        args = payload.model_dump(exclude_none=True)
        if not isinstance(args, dict) or set(args) - {'import_id', 'row', 'unit', 'advice_engine'}:
            raise ValueError('请求支持 import_id、row、unit。')
        if args.get('advice_engine', 'rules') not in ['rules', 'kimi']:
            raise ValueError('advice_engine支持rules或kimi')
        imp = imported(args.get('import_id')); meta = imp['metadata']
        if meta['needs_unit'] and args.get('unit') not in ['mmol/L', 'mg/dL']:
            raise ValueError('原文件没有写明血糖单位，请根据设备或原始记录选择单位。')
        if args.get('unit') and args['unit'] not in ['mmol/L', 'mg/dL']:
            raise ValueError('血糖单位无效')
        if meta['unit'] and args.get('unit') and meta['unit'] != args['unit']:
            raise ValueError('所选单位与文件标注不一致')
        if meta['needs_row'] and 'row' not in args:
            raise ValueError('文件包含多份独立检查记录，请选择属于当前用户的一行。')
        idx = args.get('row', 0)
        if isinstance(idx, bool) or not isinstance(idx, int) or not 0 <= idx < meta['rows']:
            raise ValueError('所选记录行不存在')
        args['profile_snapshot'] = require_user(imp['user_id'])
        aid = uuid.uuid4().hex
        sql('INSERT INTO analyses VALUES(?,?,?,?,?,?,?,?)', (aid, imp['user_id'], imp['id'], 'queued', now(), json.dumps(args, ensure_ascii=False), None, None))
        pool.submit(compute, aid)
        return jsonify(id=aid, status='queued', status_url=f'/api/v2/analyses/{aid}')

    @app.get('/api/v2/analyses/{aid}', tags=['3. 模型计算与结果'], summary='查询计算状态及完整报告')
    def get_analysis(aid):
        return jsonify(analysis(aid))

    @app.get('/api/v2/users/{uid}/analyses', tags=['3. 模型计算与结果'], summary='读取用户分析历史')
    def history(uid):
        require_user(uid)
        return jsonify(analyses=[analysis(r['id']) for r in sql('SELECT id FROM analyses WHERE user_id=? ORDER BY created DESC LIMIT 100', (uid,))])

    @app.get('/api/v2/users/{uid}/integration', tags=['4. 多场景读取'], summary='读取用户档案及关联结果')
    def integration_v2(uid):
        profile = require_user(uid)
        rows = sql("SELECT id FROM analyses WHERE user_id=? AND status IN ('completed','needs_data') ORDER BY created DESC LIMIT 20", (uid,))
        return jsonify(schema_version='2.0', module='chronic', user_id=uid, profile=profile,
                       results=[analysis(r['id']) for r in rows])

    @app.get('/api/v2/analyses/{aid}/download', tags=['3. 模型计算与结果'], summary='下载逐行计算结果CSV')
    def download_v2(aid):
        r = analysis(aid)
        if r['status'] not in ['completed', 'needs_data']:
            raise Conflict('分析尚未完成')
        return send_file(base / 'analyses' / aid / 'output/results.csv', as_attachment=True, download_name='health-analysis.csv')

    @app.get('/api/v2/templates/{kind}', tags=['2. 导入数据'], summary='下载输入模板：heart、survey或glucose')
    def template(kind):
        if kind not in NAMES:
            raise NotFound('模板不存在')
        return send_file(Path(__file__).parent / 'templates' / f'{kind}.csv', as_attachment=True, download_name=f'{kind}.csv')

    # Recovery is explicit: interrupted running jobs are kept for inspection; queued jobs resume.
    sql("UPDATE analyses SET status='failed',error='服务在分析时重启，请重新发起分析。' WHERE status='running'")
    for r in sql("SELECT id FROM analyses WHERE status='queued'"):
        pool.submit(compute, r['id'])
