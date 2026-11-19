from fastapi.testclient import TestClient
import io,json,tempfile,time,unittest
from pathlib import Path
from service.app import create_app
ROOT=Path(__file__).parent
class BusinessTest(unittest.TestCase):
 def setUp(self):
  import os
  from unittest.mock import patch
  self.enterContext(patch.dict(os.environ,{'CHRONIC_API_TOKEN':'','KIMI_API_KEY':''}))
 def test_business_flow(self):
  with tempfile.TemporaryDirectory() as tmp:
   app=create_app(tmp);c=TestClient(app)
   uid=c.post('/api/v2/users',json={'name':'验收用户','age':35,'sex':'female','height_cm':165,'weight_kg':60}).json()['id']
   ids=[]
   for kind in ['glucose','survey','heart']:
    raw=(ROOT/'service/templates'/f'{kind}.csv').read_bytes()
    imp=c.post('/api/v2/imports',data={'user_id':uid},files={'file':(kind+'.csv',raw)})
    self.assertEqual(imp.status_code,201,imp.json());self.assertEqual(imp.json()['kind'],kind)
    a=c.post('/api/v2/analyses',json={'import_id':imp.json()['id'],'advice_engine':'rules'});self.assertEqual(a.status_code,202,a.json())
    aid=a.json()['id'];ids.append(aid)
    for _ in range(300):
     r=c.get('/api/v2/analyses/'+aid).json()
     if r['status'] not in ['queued','running']:break
     time.sleep(.1)
    self.assertEqual(r['status'],'completed',r)
    self.assertTrue(r['report']['recommendations']);self.assertEqual(r['report']['profile_snapshot']['id'],uid)
    if kind=='glucose':
     self.assertEqual(len(r['report']['forecasts']),2);self.assertEqual(len(r['report']['series']),12)
    download=c.get('/api/v2/analyses/'+aid+'/download');self.assertEqual(download.status_code,200);download.close()
   self.assertEqual(len(c.get('/api/v2/users/'+uid+'/integration').json()['results']),3)
   bad=c.post('/api/v2/imports',data={'user_id':uid},files={'file':('bad.csv',b'name,age\nx,20\n')})
   self.assertEqual(bad.status_code,400)
   multi=(ROOT/'datasets/heart_disease/processed.cleveland.data').read_bytes()
   imp=c.post('/api/v2/imports',data={'user_id':uid},files={'file':('heart.data',multi)}).json()
   self.assertTrue(imp['needs_row']);self.assertEqual(c.post('/api/v2/analyses',json={'import_id':imp['id']}).status_code,400)
   self.assertEqual(c.get('/api/v2/imports/'+imp['id']+'/rows?offset=200').json()['offset'],200)
   g=json.loads((ROOT/'artifacts/v2/glucose_in_range_input.json').read_text())
   r=c.post('/api/v2/records',json={'user_id':uid,'unit':g['unit'],'measurements':g['measurements']});self.assertEqual(r.status_code,201,r.json())
   app2=create_app(tmp);c2=TestClient(app2);self.assertEqual(c2.get('/api/v2/analyses/'+ids[0]).json()['status'],'completed')
   for path in ['/','/assets/workspace.css','/assets/workspace.js','/api/v2/templates/glucose','/healthz']:
    r=c2.get(path);self.assertEqual(r.status_code,200);r.close()
 def test_authentication(self):
  import os
  from unittest.mock import patch
  with tempfile.TemporaryDirectory() as tmp,patch.dict(os.environ,{'CHRONIC_API_TOKEN':'integration-test-only'}):
   c=TestClient(create_app(tmp))
   self.assertEqual(c.get('/api/v2/users').status_code,401)
   self.assertEqual(c.post('/session',json={'password':'wrong'}).status_code,401)
   self.assertEqual(c.post('/session',json={'password':'integration-test-only'}).status_code,200)
   self.assertEqual(c.get('/api/v2/users').status_code,200)
   self.assertEqual(c.post('/api/v2/users',json={'name':'blocked'}).status_code,403)
   self.assertEqual(c.post('/api/v2/users',json={'name':'permitted'},headers={'X-Requested-With':'chronic-workspace'}).status_code,201)
 def test_import_boundaries(self):
  import pandas as pd
  with tempfile.TemporaryDirectory() as tmp:
   c=TestClient(create_app(tmp));uid=c.post('/api/v2/users',json={'name':'数据格式验证'}).json()['id']
   d=pd.read_csv(ROOT/'service/templates/glucose.csv');buffer=io.BytesIO();d.to_excel(buffer,index=False)
   response=c.post('/api/v2/imports',data={'user_id':uid},files={'file':('device.xlsx',buffer.getvalue())})
   self.assertEqual(response.status_code,201,response.json());self.assertFalse(response.json()['needs_unit'])
   ambiguous=d.rename(columns={'血糖(mmol/L)':'血糖'})
   response=c.post('/api/v2/imports',data={'user_id':uid},files={'file':('device.csv',ambiguous.to_csv(index=False).encode())})
   self.assertTrue(response.json()['needs_unit'])
   self.assertEqual(c.post('/api/v2/analyses',json={'import_id':response.json()['id']}).status_code,400)
   mixed=d.assign(patient_id=['a']*6+['b']*6)
   response=c.post('/api/v2/imports',data={'user_id':uid},files={'file':('mixed.csv',mixed.to_csv(index=False).encode())})
   self.assertEqual(response.status_code,400)
   self.assertEqual(c.get('/api/v2/analyses/not-found').status_code,404)
if __name__=='__main__':unittest.main()
