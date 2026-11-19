from fastapi.testclient import TestClient
import tempfile,time,io,os,unittest
from pathlib import Path
from service.app import create_app
ROOT=Path(__file__).parent
class ServiceTest(unittest.TestCase):
 def setUp(self):
  import os
  from unittest.mock import patch
  self.enterContext(patch.dict(os.environ,{'CHRONIC_API_TOKEN':'','KIMI_API_KEY':''}))
 def test_flow(self):
  with tempfile.TemporaryDirectory() as folder:
   app=create_app(folder);c=TestClient(app)
   response=c.get('/');self.assertEqual(response.status_code,200);response.close()
   self.assertEqual(len(c.get('/api/v1/models').json()['models']),3)
   self.assertEqual(c.post('/api/v1/users',json={'name':'x','age':-1}).status_code,422)
   u=c.post('/api/v1/users',json={'name':'接口测试','age':30}).json()['id']
   def run(data,model='heart',name='heart.data',options='{}'):
    r=c.post('/api/v1/predictions',data={'user_id':u,'model':model,'options':options},files={'file':(name,data)})
    self.assertEqual(r.status_code,202,r.json());jid=r.json()['job_id']
    for _ in range(300):
     j=c.get('/api/v1/predictions/'+jid).json()
     if j['status'] not in ['queued','running']:return jid,j
     time.sleep(.1)
    self.fail('timeout')
   raw=(ROOT/'datasets/heart_disease/processed.cleveland.data').read_bytes()
   jid,j=run(raw);self.assertIn(j['status'],['completed','partial'],j)
   rows=c.get('/api/v1/predictions/'+jid+'/results?limit=2').json()
   self.assertEqual(len(rows['rows']),2);self.assertIn('result_advice',rows['rows'][0])
   response=c.get('/api/v1/predictions/'+jid+'/download');self.assertEqual(response.status_code,200);response.close()
   survey=(ROOT/'datasets/cdc_diabetes/diabetes_012_health_indicators_BRFSS2015.csv').read_bytes().splitlines()
   _,good=run(b'\n'.join(survey[:11]),'survey','survey.csv');self.assertEqual(good['status'],'completed',good)
   _,bad=run(b'wrong\n1\n','survey','bad.csv');self.assertEqual(bad['status'],'rejected')
   excel=next((ROOT/'datasets/shanghai_cgm/Shanghai_T2DM').glob('*.xlsx'))
   _,g=run(excel.read_bytes(),'glucose','sample.xlsx','{"single_patient":true,"timezone":"Asia/Shanghai","unit":"mg/dL"}')
   self.assertIn(g['status'],['completed','partial'],g)
   self.assertEqual(len(c.get('/api/v1/users/'+u+'/integration').json()['predictions']),4)
   c2=TestClient(create_app(folder));self.assertEqual(c2.get('/api/v1/users/'+u).json()['name'],'接口测试')
   self.assertEqual(c2.get('/api/v1/predictions/'+jid).json()['status'],j['status'])
   os.environ['CHRONIC_API_TOKEN']='test-token'
   try:
    self.assertEqual(c.get('/api/v1/users').status_code,401)
    self.assertEqual(c.get('/api/v1/users',headers={'Authorization':'Bearer test-token'}).status_code,200)
   finally:del os.environ['CHRONIC_API_TOKEN']
if __name__=='__main__':unittest.main()
