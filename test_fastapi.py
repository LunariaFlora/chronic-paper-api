import json,os,time,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
from fastapi.testclient import TestClient
from service.app import create_app
ROOT=Path(__file__).parent
class FastAPITest(unittest.TestCase):
 def test_flow_and_docs(self):
  with tempfile.TemporaryDirectory() as tmp, TestClient(create_app(tmp)) as c:
   self.assertEqual(c.get('/docs').status_code,200)
   schema=c.get('/openapi.json').json()
   self.assertIn('multipart/form-data',schema['paths']['/api/v2/imports']['post']['requestBody']['content'])
   self.assertIn('UserInput',schema['components']['schemas'])
   self.assertEqual(c.post('/api/v2/users',json={'name':'错误','age':-1}).status_code,422)
   u=c.post('/api/v2/users',json={'name':'FastAPI验证','age':45}).json()['id']
   for kind in ['heart','survey','glucose']:
    r=c.post('/api/v2/imports',data={'user_id':u},files={'file':(kind+'.csv',(ROOT/'service/templates'/f'{kind}.csv').read_bytes(),'text/csv')})
    self.assertEqual(r.status_code,201,r.text)
    a=c.post('/api/v2/analyses',json={'import_id':r.json()['id'],'advice_engine':'rules'})
    self.assertEqual(a.status_code,202,a.text);aid=a.json()['id']
    for _ in range(300):
     r=c.get('/api/v2/analyses/'+aid).json()
     if r['status'] not in ['queued','running']:break
     time.sleep(.1)
    self.assertEqual(r['status'],'completed',r)
    self.assertTrue(r['report']['recommendations'])
    self.assertEqual(c.get('/api/v2/analyses/'+aid+'/download').status_code,200)
   self.assertEqual(len(c.get('/api/v2/users/'+u+'/integration').json()['results']),3)
   self.assertEqual(c.get('/api/v2/analyses/absent').status_code,404)
   with TestClient(create_app(tmp)) as reopened:
    self.assertEqual(reopened.get('/api/v2/analyses/'+aid).json()['status'],'completed')
 def test_auth(self):
  with tempfile.TemporaryDirectory() as tmp,patch.dict(os.environ,{'CHRONIC_API_TOKEN':'test-only'}),TestClient(create_app(tmp)) as c:
   self.assertEqual(c.get('/api/v2/users').status_code,401)
   self.assertEqual(c.get('/api/v2/users',headers={'Authorization':'Bearer test-only'}).status_code,200)
   self.assertEqual(c.post('/session',json={'password':'test-only'}).status_code,200)
   self.assertEqual(c.get('/api/v2/users').status_code,200)
   self.assertEqual(c.post('/api/v2/users',json={'name':'x'}).status_code,403)
   self.assertEqual(c.post('/api/v2/users',json={'name':'x'},headers={'X-Requested-With':'chronic-workspace'}).status_code,201)
if __name__=='__main__':unittest.main()
