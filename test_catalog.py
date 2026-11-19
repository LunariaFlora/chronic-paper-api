import json,os,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
from chronic.catalog import model_catalog
from chronic.api_session import request_recommendations,load_api_env


class CatalogTests(unittest.TestCase):
    def test_saved_training_is_found_again_without_registry(self):
        with tempfile.TemporaryDirectory() as temp:
            app=Path(temp);program=app/'程序';folder=app/'结果/run_new'
            (folder/'model').mkdir(parents=True)
            (folder/'model/table_model.joblib').write_bytes(b'not loaded by catalog')
            (folder/'run_summary.json').write_text(json.dumps({'input':'new.csv','target':'class','model':'logistic'}))
            first=model_catalog(program,app);second=model_catalog(program,app)
            self.assertEqual(len(first),1);self.assertEqual(first,second)
            self.assertEqual(first[0]['origin'],'本机新训练')
            self.assertEqual(first[0]['info']['model'],'logistic')

    def test_incomplete_training_not_presented_as_complete(self):
        with tempfile.TemporaryDirectory() as temp:
            app=Path(temp);path=app/'结果/run_bad/model';path.mkdir(parents=True)
            (path/'table_model.joblib').write_bytes(b'partial')
            self.assertEqual(model_catalog(app/'程序',app),[])


class SessionApiTests(unittest.TestCase):
    def test_reuses_local_environment_without_asking_for_key(self):
        with patch.dict(os.environ,{'KIMI_API_KEY':'unit-test-placeholder'}),patch('builtins.input',return_value='y'),patch('getpass.getpass') as ask:
            self.assertTrue(request_recommendations());ask.assert_not_called()

    def test_no_key_noninteractive_skips_cloud(self):
        with patch.dict(os.environ,{},clear=True),patch('builtins.input',return_value='y'),patch('sys.stdin.isatty',return_value=False):
            self.assertFalse(request_recommendations())

    def test_loads_only_documented_fields_without_rewriting_env(self):
        with tempfile.TemporaryDirectory() as temp,patch.dict(os.environ,{},clear=True):
            p=Path(temp)/'.env';content='KIMI_API_KEY="test-placeholder"\nKIMI_MODEL=test-model\nUNRELATED=ignored\n'
            p.write_text(content);load_api_env(temp)
            self.assertEqual(os.environ['KIMI_API_KEY'],'test-placeholder')
            self.assertEqual(os.environ['KIMI_MODEL'],'test-model')
            self.assertNotIn('UNRELATED',os.environ)
            self.assertEqual(p.read_text(),content)

    def test_blank_template_preserves_existing_local_key(self):
        with tempfile.TemporaryDirectory() as temp,patch.dict(os.environ,{'KIMI_API_KEY':'existing-test'}):
            (Path(temp)/'.env').write_text('KIMI_API_KEY=\n')
            load_api_env(temp);self.assertEqual(os.environ['KIMI_API_KEY'],'existing-test')


if __name__=='__main__':unittest.main()
