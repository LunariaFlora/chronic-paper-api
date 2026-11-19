from pathlib import Path
import io,json,tempfile,unittest
from contextlib import redirect_stdout
from unittest.mock import patch
import pandas as pd
from chronic.console import choose_matching,show_model
from chronic.model_match import missing_columns
from chronic.catalog import model_catalog
from chronic.terminal_results import load_view,show_page,show_issues,show_advice,advice_path

ROOT=Path(__file__).resolve().parent
SHARE=ROOT/'分享版_慢病预测_Windows64'


class TerminalTests(unittest.TestCase):
    def test_reported_heart_survey_mismatch_and_reselection(self):
        entries=model_catalog(ROOT,ROOT)
        heart=entries[0];survey=entries[1]
        d=pd.read_csv(ROOT/'datasets/cdc_diabetes/diabetes_012_health_indicators_BRFSS2015.csv',nrows=2)
        self.assertIn('chol',missing_columns(d,heart));self.assertEqual(missing_columns(d,survey),[])
        text=io.StringIO()
        with patch('builtins.input',return_value='1'),patch('chronic.console.model_catalog',return_value=[heart,survey]),redirect_stdout(text):
            chosen=choose_matching(Path('survey.csv'),d,heart)
        self.assertEqual(chosen['kind'],'survey')
        self.assertIn('缺少',text.getvalue());self.assertIn('糖尿病问卷',text.getvalue())

    def test_result_and_invalid_reason_are_readable_inline(self):
        with tempfile.TemporaryDirectory() as temp:
            p=Path(temp)/'results.csv'
            pd.DataFrame({'result_source_row':[2,3],'result_status':['ok','invalid_input'],
                          'result_reason':['','年龄缺失'],'result_class':[1,None],'answer':[1,0]}).to_csv(p,index=False)
            frame,issues=load_view(p,'answer')
            self.assertEqual(len(frame),1);self.assertTrue(frame.iloc[0]['正确'])
            text=io.StringIO()
            with redirect_stdout(text):show_page(frame);show_issues(issues)
            self.assertIn('预测',text.getvalue());self.assertIn('年龄缺失',text.getvalue())

    def test_saved_results_and_advice_display_without_cloud_or_explorer(self):
        selected=model_catalog(SHARE/'程序',SHARE)[0]
        text=io.StringIO()
        with redirect_stdout(text),patch('builtins.input',side_effect=['a','']),patch('chronic.llm_advice.generate_advice',side_effect=AssertionError('no cloud')):
            show_model(selected)
        output=text.getvalue()
        expected=json.loads(advice_path(selected['folder']).read_text(encoding='utf8'))[0]['response']['advice']['recommendations'][0]['text']
        self.assertIn('真实',output);self.assertIn('预测',output);self.assertIn(expected,output)
        self.assertNotIn('成果样例',output);self.assertNotIn('本机新训练',output)
        self.assertNotIn('成果目录',output);self.assertNotIn('KIMI_API_KEY',output)


if __name__=='__main__':unittest.main()
