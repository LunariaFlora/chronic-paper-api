import copy,json,tempfile,unittest
from pathlib import Path
import numpy as np,pandas as pd,joblib
from chronic.batch import glucose_batch,survey_batch,run_batch,custom_batch
from chronic.inference import ChronicPredictor
from chronic.recommendations import recommend
from chronic.table_io import read_table
from chronic.table_models import prepare

ROOT=Path(__file__).resolve().parent

def glucose_frame(n=16):
    return pd.DataFrame({'patient_id':['A']*n,'timestamp':pd.date_range('2026-10-01',periods=n,freq='15min',tz='Asia/Shanghai').astype(str),
        'glucose':np.arange(n)+100.,'unit':['mg/dL']*n,'unrelated_note':['保留我']*n})

class WholeTableTests(unittest.TestCase):
    def test_all_rows_kept_and_first_11_explained(self):
        d=glucose_frame();r,_=glucose_batch(d,'raw.csv')
        self.assertEqual(len(r),16);self.assertEqual((r.result_status=='ok').sum(),5)
        self.assertEqual((r.result_status=='insufficient_history').sum(),11)
        self.assertTrue(r.result_advice.str.len().gt(0).all())

    def test_short_table_still_has_empty_prediction_columns(self):
        r,_=glucose_batch(glucose_frame(5),'raw.csv')
        self.assertTrue(r.result_glucose_30min_mg_dl.isna().all())
        self.assertTrue(r.result_high_60min_flag.isna().all())

    def test_vectorized_predictions_match_existing_function(self):
        d=glucose_frame();r,_=glucose_batch(d,'raw.csv')
        p={'unit':'mg/dL','measurements':[{'timestamp':x.timestamp,'value':x.glucose} for x in d.iloc[:12].itertuples()]}
        expected=ChronicPredictor().glucose(p)
        for h in ['30','60']:
            self.assertAlmostEqual(r.loc[11,f'result_glucose_{h}min_mg_dl'],expected['forecasts'][h]['point_mg_dl'])

    def test_units_and_sorting_preserve_original_row_order(self):
        d=glucose_frame();a,_=glucose_batch(d,'a.csv')
        d['glucose']/=18.0182;d['unit']='mmol/L';d=d.iloc[::-1].reset_index(drop=True)
        b,_=glucose_batch(d,'b.csv')
        self.assertAlmostEqual(a.loc[15,'result_glucose_30min_mg_dl'],b.loc[0,'result_glucose_30min_mg_dl'])
        self.assertEqual(b.loc[0,'result_source_row'],2)

    def test_no_cross_patient_windows(self):
        d=glucose_frame(20);d.loc[10:,'patient_id']='B'
        r,_=glucose_batch(d,'raw.csv');self.assertFalse(r.result_status.eq('ok').any())

    def test_no_future_leakage(self):
        d=glucose_frame();a,_=glucose_batch(d,'a.csv');d.loc[12:,'glucose']=400
        b,_=glucose_batch(d,'b.csv');self.assertEqual(a.loc[11,'result_glucose_60min_mg_dl'],b.loc[11,'result_glucose_60min_mg_dl'])

    def test_missing_measurement_not_silently_removed(self):
        d=glucose_frame();d['glucose']=d['glucose'].astype(object);d.loc[5,'glucose']='bad';r,_=glucose_batch(d,'raw.csv')
        self.assertEqual(r.loc[5,'result_status'],'invalid_input');self.assertFalse(r.result_status.eq('ok').any())

    def test_duplicates_and_gaps_block_windows(self):
        for kind in ['duplicate','gap']:
            d=glucose_frame()
            d.loc[8,'timestamp']=d.loc[7,'timestamp'] if kind=='duplicate' else '2026-10-01T02:02:00+08:00'
            r,_=glucose_batch(d,'raw.csv');self.assertFalse(r.result_status.eq('ok').any())

    def test_invalid_time_does_not_get_removed_then_bridged(self):
        d=glucose_frame();d.loc[2,'timestamp']='unknown';r,_=glucose_batch(d,'raw.csv')
        self.assertTrue(r.result_status.eq('ambiguous_timeline').all())

    def test_explicit_single_person_timezone_and_unit(self):
        d=glucose_frame().drop(columns=['patient_id','unit'])
        d['timestamp']=pd.to_datetime(d.timestamp).dt.tz_localize(None)
        with self.assertRaises(ValueError):glucose_batch(d,'raw.csv')
        r,s=glucose_batch(d,'raw.csv',unit='mg/dL',timezone='Asia/Shanghai',single_patient=True)
        self.assertEqual(s['timezone_localized_rows'],16);self.assertEqual(r.result_status.eq('ok').sum(),5)

    def test_training_range_abstention(self):
        d=glucose_frame();d['glucose']=599;r,_=glucose_batch(d,'raw.csv')
        self.assertEqual(r.loc[11,'result_status'],'unsupported_training_range')
        self.assertTrue(pd.isna(r.loc[11,'result_glucose_30min_mg_dl']))

    def test_survey_ignores_target_and_preserves_bad_row(self):
        good=json.loads((ROOT/'artifacts/v2/survey_input.json').read_text(encoding='utf8'))
        d=pd.DataFrame([good,dict(good,Age=60)]);d['Diabetes_012']=[0,2]
        r,_=survey_batch(d,'raw.csv');expected=ChronicPredictor().diabetes(good)
        self.assertAlmostEqual(r.loc[0,'result_score_diabetes'],expected['class_scores']['diabetes'])
        self.assertEqual(r.loc[1,'result_status'],'invalid_input');self.assertEqual(r.loc[1,'result_advice_priority'],'data_needed')

    def test_csv_output_retains_rows_extra_columns_and_rejects_overwrite(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td);d=glucose_frame();d.to_csv(p/'in.csv',index=False)
            summary=run_batch(p/'in.csv',p/'out');out=pd.read_csv(p/'out/results.csv')
            self.assertEqual(len(out),len(d));self.assertEqual(out.unrelated_note.tolist(),d.unrelated_note.tolist())
            self.assertTrue((p/'out/issues.csv').exists());self.assertEqual(summary['output_rows'],16)
            with self.assertRaises(FileExistsError):run_batch(p/'in.csv',p/'out')

    def test_excel_chinese_mapping(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td);d=glucose_frame().rename(columns={'timestamp':'监测时刻','glucose':'读数','patient_id':'受试者'})
            d.to_excel(p/'in.xlsx',index=False)
            s=run_batch(p/'in.xlsx',p/'out',profile='glucose',mapping={'timestamp':'监测时刻','glucose':'读数','patient_id':'受试者'})
            self.assertEqual(s['status_counts']['ok'],5)

    def test_new_schema_handles_numeric_category_encoding(self):
        schema=json.loads((ROOT/'configs/heart_cleveland.json').read_text())
        d=read_table(ROOT/'datasets/heart_disease/processed.cleveland.data',schema).iloc[:2]
        d.loc[0,'sex']=1;d.loc[0,'thal']='?';x,e=prepare(d,schema)
        self.assertTrue(e.eq('').all());self.assertTrue(pd.isna(x.loc[0,'thal']));self.assertEqual(x.loc[0,'sex'],'1.0')

    def test_unsupported_schema_retains_original_rows_and_reason(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td);pd.DataFrame({'other_feature':[1,2,3]}).to_csv(p/'in.csv',index=False)
            s=run_batch(p/'in.csv',p/'out');r=pd.read_csv(p/'out/results.csv')
            self.assertEqual(s['status_counts'],{'unsupported_schema':3});self.assertEqual(r.other_feature.tolist(),[1,2,3])
            self.assertTrue(r.result_reason.str.len().gt(0).all())

    @unittest.skipUnless((ROOT/'artifacts/table_heart_v1/table_model.joblib').exists(),'heart model not yet trained')
    def test_new_task_does_not_use_target_and_preserves_missing_flags(self):
        b=joblib.load(ROOT/'artifacts/table_heart_v1/table_model.joblib')
        d=read_table(ROOT/'datasets/heart_disease/processed.cleveland.data',b['schema'])
        a,_=custom_batch(d,'original.data',b);d['num']='0';z,_=custom_batch(d,'original.data',b)
        np.testing.assert_allclose(a.result_score_1,z.result_score_1)
        self.assertTrue(a.result_status.eq('ok').all());self.assertTrue(a.result_imputed_columns.str.len().gt(0).any())

class AdviceTests(unittest.TestCase):
    def test_low_takes_priority_over_lifestyle_and_high_prediction(self):
        r=recommend('glucose',{'current_mg_dl':50},{'risks':{'60':{'high':True}}})
        self.assertEqual(r['priority'],'urgent');codes=[x['code'] for x in r['items']]
        self.assertNotIn('activity_plan',codes);self.assertNotIn('regular_meals',codes);self.assertIn('measured_low',codes)

    def test_predicted_low_does_not_trigger_measured_low_treatment(self):
        r=recommend('glucose',{'current_mg_dl':110},{'risks':{'60':{'low':True}}})
        codes=[x['code'] for x in r['items']];self.assertIn('forecast_low',codes);self.assertNotIn('measured_low',codes);self.assertNotIn('activity_plan',codes)

    def test_advice_responds_to_questionnaire(self):
        a=recommend('survey',{'PhysActivity':0,'Veggies':0},{'positive':True})
        b=recommend('survey',{'PhysActivity':1,'Veggies':1},{'positive':False})
        self.assertGreater(len(a['items']),len(b['items']));self.assertTrue(all(x['source'].startswith('https://') for x in a['items']))

    def test_unknown_task_and_missing_measurement_abstain(self):
        self.assertEqual(recommend('untrained_disease',{},None)['priority'],'not_supported')
        self.assertEqual(recommend('glucose',{'current_mg_dl':None})['priority'],'data_needed')

    def test_no_prediction_is_not_treated_as_a_negative_prediction(self):
        r=recommend('glucose',{'current_mg_dl':110},{'risks':{}})
        self.assertNotIn('activity_plan',[x['code'] for x in r['items']])

if __name__=='__main__':unittest.main()
