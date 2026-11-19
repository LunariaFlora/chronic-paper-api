"""Previous batch URLs remain callable but are hidden from the new-user docs."""
import json,uuid
from datetime import datetime,timezone
from pathlib import Path
import pandas as pd
from fastapi import Form,File,UploadFile,HTTPException
from fastapi.responses import FileResponse
from chronic.batch import run_batch
from chronic.contracts import SURVEY
from chronic.data import ROOT

def register_legacy(app,base,sql,pool,require_user):
 sql('CREATE TABLE IF NOT EXISTS jobs(id TEXT PRIMARY KEY,user_id TEXT,status TEXT,created TEXT,summary TEXT,error TEXT)')
 sql("UPDATE jobs SET status='failed',error='服务重启中断，请重新提交' WHERE status IN ('queued','running')")
 def job(jid):
  r=sql('SELECT * FROM jobs WHERE id=?',(jid,),True)
  if not r:raise HTTPException(404,'任务不存在')
  r['summary']=json.loads(r['summary']) if r['summary'] else None
  return r
 def compute(jid,path,model,options):
  sql("UPDATE jobs SET status='running' WHERE id=?",(jid,))
  try:
   s=run_batch(path,base/jid/'output',profile=model if model!='heart' else 'auto',model=ROOT/'artifacts/table_heart_v1/table_model.joblib' if model=='heart' else None,**options)
   good=s['status_counts'].get('ok',0)
   status='completed' if good and good==s['input_rows'] else ('partial' if good else 'rejected')
   sql('UPDATE jobs SET status=?,summary=? WHERE id=?',(status,json.dumps(s,ensure_ascii=False,default=str),jid))
  except Exception as exc:sql("UPDATE jobs SET status='failed',error=? WHERE id=?",(str(exc),jid))
 @app.get('/api/v1/models',include_in_schema=False)
 def models():return {'models':[{'id':'survey','name':'糖尿病问卷分类','fields':list(SURVEY)},{'id':'heart','name':'心脏病状态分类','fields':json.loads((ROOT/'configs/heart_cleveland.json').read_text(encoding='utf8'))},{'id':'glucose','name':'连续血糖预测','fields':['timestamp','glucose','patient_id'],'interval_minutes':15,'history_points':12}]}
 @app.get('/api/v1/users',include_in_schema=False)
 def users():return {'users':[dict(id=r['id'],**json.loads(r['data'])) for r in sql('SELECT * FROM users')]}
 @app.post('/api/v1/predictions',status_code=202,include_in_schema=False)
 def predict(user_id:str=Form(...),model:str=Form('survey'),options:str=Form('{}'),file:UploadFile=File(...)):
  require_user(user_id)
  if model not in ['survey','heart','glucose']:raise ValueError('未知模型')
  opts=json.loads(options)
  if not isinstance(opts,dict) or set(opts)-{'mapping','unit','timezone','single_patient','sheet'}:raise ValueError('options字段不支持')
  if 'single_patient' in opts and not isinstance(opts['single_patient'],bool):raise ValueError('single_patient须为布尔值')
  if 'mapping' in opts and (not isinstance(opts['mapping'],dict) or not all(isinstance(k,str) and isinstance(v,str) for k,v in opts['mapping'].items())):raise ValueError('mapping须为字段名映射对象')
  suffix=Path(file.filename).suffix.lower()
  if suffix not in ['.csv','.xls','.xlsx','.data']:raise ValueError('文件格式不支持')
  content=file.file.read(32*1024*1024+1)
  if len(content)>32*1024*1024:raise HTTPException(413,'文件超过32MB')
  jid=uuid.uuid4().hex;folder=base/jid;folder.mkdir();path=folder/('input'+suffix);path.write_bytes(content)
  sql('INSERT INTO jobs VALUES(?,?,?,?,?,?)',(jid,user_id,'queued',datetime.now(timezone.utc).isoformat(),None,None))
  pool.submit(compute,jid,path,model,opts)
  return {'job_id':jid,'status':'queued'}
 @app.get('/api/v1/predictions/{jid}',include_in_schema=False)
 def status(jid:str):return job(jid)
 @app.get('/api/v1/predictions/{jid}/results',include_in_schema=False)
 def results(jid:str,offset:int=0,limit:int=100):
  r=job(jid)
  if r['status'] not in ['completed','partial','rejected']:raise HTTPException(409,'结果尚未就绪')
  offset=max(0,offset);limit=max(1,min(500,limit))
  d=pd.read_csv(base/jid/'output/results.csv').iloc[offset:offset+limit]
  return {'job_id':jid,'total':r['summary']['output_rows'],'offset':offset,'rows':json.loads(d.to_json(orient='records',force_ascii=False))}
 @app.get('/api/v1/predictions/{jid}/download',include_in_schema=False)
 def download(jid:str):
  if job(jid)['status'] not in ['completed','partial','rejected']:raise HTTPException(409,'结果尚未就绪')
  return FileResponse(base/jid/'output/results.csv',filename=jid+'.csv')
 @app.get('/api/v1/users/{uid}/integration',include_in_schema=False)
 def integration(uid:str):return {'user':require_user(uid),'predictions':[job(r['id']) for r in sql('SELECT id FROM jobs WHERE user_id=? ORDER BY created DESC',(uid,))]}
