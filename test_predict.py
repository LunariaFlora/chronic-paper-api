import copy,json,subprocess,sys,unittest
from pathlib import Path
from datetime import timedelta
import joblib,numpy as np,pandas as pd
from threadpoolctl import threadpool_limits
from chronic.contracts import validate_glucose,validate_survey,glucose_state,MG_PER_MMOL,SURVEY
from chronic.data import windows_from_frame,features,load_glucose
from chronic.inference import ChronicPredictor
from chronic.metrics import binary_metrics,threshold_on_validation

ROOT=Path(__file__).resolve().parent
ART=ROOT/'artifacts/v2'

def read(name):return json.loads((ART/name).read_text(encoding='utf8'))

def example():return read('glucose_in_range_input.json')

class InputTests(unittest.TestCase):
    def bad(self,p):
        with self.assertRaises(ValueError):validate_glucose(p)

    def test_valid_record(self):
        v,t,u=validate_glucose(example());self.assertEqual(len(v),12);self.assertEqual(u,'mg/dL')
        self.assertEqual((t[-1]-t[0]).total_seconds(),165*60)

    def test_units_convert_identically(self):
        a=example();b=copy.deepcopy(a);b['unit']='mmol/L'
        for r in b['measurements']:r['value']/=MG_PER_MMOL
        np.testing.assert_allclose(validate_glucose(a)[0],validate_glucose(b)[0],rtol=1e-12)

    def test_missing_or_unknown_unit(self):
        for unit in [None,'mg/dl','mmol',18]:
            p=example();p['unit']=unit;self.bad(p)

    def test_count(self):
        for n in [0,11,13]:
            p=example();p['measurements']=(p['measurements']*2)[:n];self.bad(p)

    def test_nonfinite_or_nonnumeric(self):
        for value in [None,True,'5.6',float('nan'),float('inf')]:
            p=example();p['measurements'][3]['value']=value;self.bad(p)

    def test_unit_mistake_and_range(self):
        for value in [5,0,-1,601]:
            p=example();p['measurements'][3]['value']=value;self.bad(p)

    def test_timestamp_required(self):
        p=example();del p['measurements'][0]['timestamp'];self.bad(p)

    def test_timezone_required(self):
        p=example();p['measurements'][0]['timestamp']=p['measurements'][0]['timestamp'][:-6];self.bad(p)

    def test_invalid_datetime(self):
        p=example();p['measurements'][0]['timestamp']='2026-99-99';self.bad(p)

    def test_gap_rejected(self):
        p=example();p['measurements'][5]['timestamp']=p['measurements'][7]['timestamp'];self.bad(p)

    def test_duplicate_rejected(self):
        p=example();p['measurements'][5]['timestamp']=p['measurements'][4]['timestamp'];self.bad(p)

    def test_reverse_rejected(self):
        p=example();p['measurements'].reverse();self.bad(p)

    def test_stale_asof(self):
        p=example();t=pd.Timestamp(p['as_of']);p['as_of']=(t+pd.Timedelta(minutes=21)).isoformat();self.bad(p)

    def test_future_measurement(self):
        p=example();t=pd.Timestamp(p['as_of']);p['as_of']=(t-pd.Timedelta(minutes=1)).isoformat();self.bad(p)

    def test_equivalent_offset_time(self):
        p=example()
        for r in p['measurements']:r['timestamp']=pd.Timestamp(r['timestamp']).tz_convert('UTC').isoformat()
        np.testing.assert_array_equal(validate_glucose(p)[0],validate_glucose(example())[0])

    def test_wrong_top_level_or_unknown_fields(self):
        for p in [[],None,{'unit':'mg/dL','measurements':None},dict(example(),typo=1)]:self.bad(p)

    def test_survey_schema(self):
        p=read('survey_input.json');self.assertEqual(len(validate_survey(p)),21)
        for key,value in [('HighBP',140),('Age',60),('BMI',float('nan')),('Sex',2)]:
            bad=dict(p);bad[key]=value
            with self.assertRaises(ValueError):validate_survey(bad)

    def test_survey_missing_extra_and_type(self):
        p=read('survey_input.json')
        for bad in [{},dict(p,unknown=1),[],None]:
            with self.assertRaises(ValueError):validate_survey(bad)

class DefinitionTests(unittest.TestCase):
    def test_current_rule_boundaries(self):
        for value,label in [(53.9,'very_low'),(54,'low'),(69.9,'low'),(70,'within_reference_range'),
            (180,'within_reference_range'),(180.1,'high'),(250,'high'),(250.1,'very_high')]:
            self.assertEqual(glucose_state(value),label)

    def test_window_and_horizon_alignment(self):
        d=pd.DataFrame({'Date':pd.date_range('2026-01-01',periods=20,freq='15min'),'CGM':np.arange(100,120)})
        x,y,m=windows_from_frame(d,'p','sample')
        np.testing.assert_array_equal(x[0],np.arange(100,112))
        np.testing.assert_array_equal(y[0],[112,113,114,115])
        self.assertEqual(m[0]['row'],11)

    def test_future_not_a_feature(self):
        d=pd.DataFrame({'Date':pd.date_range('2026-01-01',periods=20,freq='15min'),'CGM':np.arange(100,120)})
        x,y,_=windows_from_frame(d,'p','sample');d.loc[12:15,'CGM']=200
        a,b,_=windows_from_frame(d,'p','sample')
        np.testing.assert_array_equal(features(x[0]),features(a[0]));self.assertFalse(np.array_equal(y[0],b[0]))

    def test_missing_and_gaps_never_bridged(self):
        for error in ['missing','gap']:
            d=pd.DataFrame({'Date':pd.date_range('2026-01-01',periods=16,freq='15min'),'CGM':[100.]*16})
            if error=='missing':d.loc[4,'CGM']=np.nan
            else:d.loc[4,'Date']+=pd.Timedelta(minutes=2)
            x,_,_=windows_from_frame(d,'p','sample');self.assertEqual(len(x),0)

    def test_early_event_even_if_end_normal(self):
        from chronic.data import risk_targets
        y=np.asarray([[65,100,100,100],[100,100,200,100],[70,180,100,100]])
        self.assertEqual(risk_targets(y,2)['low'].tolist(),[True,False,False])
        self.assertEqual(risk_targets(y,2)['high'].tolist(),[False,False,False])
        self.assertEqual(risk_targets(y,4)['high'].tolist(),[False,True,False])

    def test_metric_counts_independent_example(self):
        d=binary_metrics([0,0,1,1],[.1,.6,.4,.9],.5)
        self.assertEqual(d['counts'],{'tn':1,'fp':1,'fn':1,'tp':1});self.assertEqual(d['precision'],.5)

    def test_threshold_selected_from_argument(self):
        threshold=threshold_on_validation(np.array([0,0,1,1]),np.array([.1,.2,.8,.9]))
        self.assertEqual(threshold,.8)

class ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.p=ChronicPredictor()

    def test_predictions_json_and_units(self):
        a=example();b=copy.deepcopy(a);b['unit']='mmol/L'
        for r in b['measurements']:r['value']/=MG_PER_MMOL
        x=self.p.glucose(a);y=self.p.glucose(b)
        json.dumps(x,allow_nan=False)
        for h in ['30','60']:
            self.assertAlmostEqual(x['forecasts'][h]['point_mg_dl'],y['forecasts'][h]['point_mg_dl'],places=6)
            for event in ['low','high']:
                z=x['forecasts'][h]['risks'][event];self.assertTrue(0<=z['model_score']<=1)
                self.assertEqual(z['flag'],z['model_score']>=z['threshold'])

    def test_ood_abstains_but_retains_current_state(self):
        p=example()
        for r in p['measurements']:r['value']=599
        d=self.p.glucose(p)
        self.assertEqual(d['status'],'unsupported_training_range');self.assertEqual(d['current']['state'],'very_high')
        self.assertIsNone(d['forecasts']['30']['point_mg_dl']);self.assertTrue(d['warnings'])

    def test_survey_three_classes_preserved(self):
        d=self.p.diabetes(read('survey_input.json'))
        self.assertEqual(set(d['class_scores']),{'no_diabetes_or_gestational_only','prediabetes','diabetes'})
        self.assertAlmostEqual(sum(d['class_scores'].values()),1)

    def test_target_timestamp(self):
        p=example();d=self.p.glucose(p)
        t=pd.Timestamp(p['measurements'][-1]['timestamp'])
        for h in ['30','60']:self.assertEqual(pd.Timestamp(d['forecasts'][h]['target_timestamp']),t+pd.Timedelta(minutes=int(h)))

    def test_missing_asof_is_explicitly_historical(self):
        p=example();del p['as_of'];self.assertTrue(self.p.glucose(p)['warnings'])

    def test_current_high_is_not_claimed_new_onset(self):
        d=self.p.glucose(read('glucose_high_input.json'))
        self.assertTrue(d['forecasts']['30']['risks']['high']['already_present'])
        self.assertEqual(d['forecasts']['30']['risks']['high']['decision_mode'],'continuation_or_crossing')

class DataAndIntegrationTests(unittest.TestCase):
    def test_survey_results_match_saved_model(self):
        from chronic.training import survey_metrics
        d=pd.read_csv(ROOT/'datasets/cdc_diabetes/diabetes_012_health_indicators_BRFSS2015.csv')
        split=pd.read_csv(ART/'survey_split.csv');ix=np.flatnonzero(split['split']=='test')
        b=joblib.load(ART/'survey.joblib');y=d.Diabetes_012.astype(int).to_numpy()[ix]
        with threadpool_limits(limits=4):p=b['model'].predict_proba(d[list(SURVEY)].iloc[ix])
        metrics=survey_metrics(y,p,b['decision_weights']);report=read('survey_metrics.json')
        self.assertEqual(metrics['confusion_matrix'],report['test']['confusion_matrix'])
        self.assertEqual(binary_metrics(y==2,p[:,2],b['threshold'])['counts'],report['test_diabetes_flag']['counts'])

    def test_no_patient_overlap_in_four_splits(self):
        d=read('glucose_metrics.json')['split'];keys=list(d)
        for i,k in enumerate(keys):
            for other in keys[i+1:]:self.assertFalse(set(d[k]['patients'])&set(d[other]['patients']))
        self.assertEqual(len(set(p for v in d.values() for p in v['patients'])),100)

    def test_no_survey_profile_overlap(self):
        d=pd.read_csv(ART/'survey_split.csv',dtype={'feature_group':str})
        self.assertEqual(int(d.groupby('feature_group')['split'].nunique().max()),1)

    def test_raw_data_window_independent_reconstruction(self):
        x,f,m=load_glucose()
        # Source rows independently read, not the training cache's own target builder.
        for idx in [0,len(x)//2,len(x)-1]:
            row=m.iloc[idx];d=pd.read_excel(ROOT/'datasets/shanghai_cgm/Shanghai_T2DM'/row['file'])
            c=next(c for c in d if 'CGM' in str(c));i=int(row['row'])
            np.testing.assert_allclose(x[idx],pd.to_numeric(d[c].iloc[i-11:i+1]))
            np.testing.assert_allclose(f[idx],pd.to_numeric(d[c].iloc[i+1:i+5]))

    def test_saved_report_matches_actual_models(self):
        from chronic.data import risk_targets
        x,f,m=load_glucose();b=joblib.load(ART/'glucose.joblib');report=read('glucose_metrics.json')
        ix=np.flatnonzero(m.patient.isin(report['split']['test']['patients']).to_numpy());xx=features(x[ix]);current=x[ix,-1]
        with threadpool_limits(limits=4):
            for h,steps in [('30',2),('60',4)]:
                fore=b['forecasters'][h]['model'];p=fore.predict(xx) if fore else current
                self.assertAlmostEqual(float(np.abs(p-f[ix,steps-1]).mean()),report['horizons'][h]['test']['mae'],places=9)
                for event in ['low','high']:
                    e=b['events'][h+'_'+event];ps=e['model'].predict_proba(xx)[:,1]
                    threshold=np.where((current>=70)&(current<=180),e['early_warning_threshold'],e['threshold'])
                    actual=binary_metrics(risk_targets(f[ix],steps)[event],ps,threshold)
                    self.assertEqual(actual['counts'],report['horizons'][h]['events'][event]['test']['counts'])

    def test_cli_valid_and_invalid(self):
        p=subprocess.run([sys.executable,'-X','utf8',str(ROOT/'predict.py'),'glucose',str(ART/'glucose_in_range_input.json')],capture_output=True,encoding='utf8')
        self.assertEqual(p.returncode,0,p.stderr);self.assertEqual(json.loads(p.stdout)['status'],'ok')
        p=subprocess.run([sys.executable,'-X','utf8',str(ROOT/'predict.py'),'glucose',str(ROOT/'no_such_input.json')],capture_output=True,encoding='utf8')
        self.assertEqual(p.returncode,2);self.assertNotIn('Traceback',p.stderr)

if __name__=='__main__':unittest.main()
