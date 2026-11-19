import copy,json,socket,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import pandas as pd
from chronic.discovery import discover
from chronic.workflow import local_only,run_workflow
from chronic.llm_advice import packet_from_row,validate_response,append_llm_advice

ROOT=Path(__file__).resolve().parent

class DiscoveryTests(unittest.TestCase):
    def test_known_original_heart_file(self):
        s,i=discover(ROOT/'datasets/heart_disease/processed.cleveland.data')
        self.assertEqual(i['rows'],303);self.assertEqual(s['target'],'num');self.assertNotIn('num',s['numeric']+s['categorical'])

    def test_code_infers_numeric_category_and_excludes_identity(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'raw.csv'
            pd.DataFrame({'patient_id':['a','b','c'],'age':[20,30,40],'sex':['f','m','f'],'disease':['n','y','n']}).to_csv(p,index=False)
            s,i=discover(p);self.assertEqual(s['numeric'],['age']);self.assertEqual(s['categorical'],['sex'])
            self.assertEqual(s['group_column'],'patient_id');self.assertEqual(s['target'],'disease')

    def test_ambiguous_target_needs_only_target_choice(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'raw.csv';pd.DataFrame({'x':[1,2,3],'label':[0,1,0],'target':[1,0,1]}).to_csv(p,index=False)
            with self.assertRaises(ValueError):discover(p)
            s,_=discover(p,'label');self.assertEqual(s['target'],'label');self.assertNotIn('target',s['numeric']+s['categorical'])

class WorkflowTests(unittest.TestCase):
    def test_network_is_blocked_and_restored(self):
        original=socket.create_connection
        with local_only():
            with self.assertRaisesRegex(RuntimeError,'禁止联网'):socket.create_connection(('example.com',443))
        self.assertIs(socket.create_connection,original)

    def test_whole_pipeline_retrains_offline_and_csv_metrics_reproduce(self):
        with tempfile.TemporaryDirectory() as td:
            out=Path(td)/'run'
            with patch('chronic.llm_advice.generate_advice',side_effect=AssertionError('LLM must not run')):
                r=run_workflow(ROOT/'datasets/heart_disease/processed.cleveland.data',out)
            self.assertTrue(r['test_accuracy_recomputed_from_csv']);self.assertTrue(r['local_ml_network_blocked'])
            self.assertTrue(r['prediction_unchanged_after_advice']);self.assertEqual(r['rows'],303)
            pred=pd.read_csv(out/'predictions/results.csv');self.assertFalse(any(c.startswith('result_advice') for c in pred))
            self.assertFalse((out/'advice').exists())

    def test_cloud_failure_keeps_complete_predictions(self):
        with tempfile.TemporaryDirectory() as td:
            out=Path(td)/'run'
            with patch('chronic.llm_advice.generate_advice',side_effect=RuntimeError('simulated unavailable')):
                r=run_workflow(ROOT/'datasets/heart_disease/processed.cleveland.data',out,recommendations=True,advice_limit=1)
            self.assertTrue(r['prediction_unchanged_after_advice']);self.assertEqual(r['recommendations']['failed'],1)
            self.assertEqual(len(pd.read_csv(out/'predictions/results.csv')),303)

class AdviceIsolationTests(unittest.TestCase):
    def test_only_whitelisted_measurements_leave_machine(self):
        s={'task':'t','interpretation':'i','numeric':['age','patient_id'],'categorical':['name','disease']}
        p=packet_from_row({'age':60,'patient_id':123456,'name':'PRIVATE','disease':'TRUE_LABEL'},s,{'predicted_class':'0'})
        text=json.dumps(p);self.assertNotIn('PRIVATE',text);self.assertNotIn('123456',text);self.assertNotIn('TRUE_LABEL',text)
        self.assertEqual(p['feature_values'],{'age':60.})

    def test_cannot_return_replacement_prediction(self):
        with self.assertRaises(ValueError):validate_response({'prediction':0,'summary':'','recommendations':[],'limitations':''})

    def test_unknown_source_rejected(self):
        d={'summary':'教育提示','recommendations':[{'category':'follow_up','text':'复查','reason':'模型仅供参考','source_ids':['invented']}],'limitations':'不是诊断'}
        with self.assertRaises(ValueError):validate_response(d)

    def test_dose_rejected(self):
        d={'summary':'教育提示','recommendations':[{'category':'follow_up','text':'服用10毫克','reason':'','source_ids':['monitor']}],'limitations':'不是诊断'}
        with self.assertRaises(ValueError):validate_response(d)

    def test_explicit_call_limit(self):
        with tempfile.TemporaryDirectory() as td:
            records=[{'source_row':i,'packet':{}} for i in range(5)]
            with patch('chronic.llm_advice.generate_advice',return_value={'advice':{}}) as call:
                r=append_llm_advice(records,td,limit=2)
            self.assertEqual(len(r),2);self.assertEqual(call.call_count,2)

if __name__=='__main__':unittest.main()
