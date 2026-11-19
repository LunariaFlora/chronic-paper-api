"""Run the real suite and retain machine-readable and human-readable results."""
import io,json,unittest
from pathlib import Path

if __name__=='__main__':
    root=Path(__file__).resolve().parent;out=root/'outputs/completion';out.mkdir(parents=True,exist_ok=True)
    stream=io.StringIO();suite=unittest.defaultTestLoader.discover(str(root),pattern='test_predict.py')
    result=unittest.TextTestRunner(stream=stream,verbosity=2).run(suite)
    text=stream.getvalue();print(text)
    (out/'test_results.txt').write_text(text,encoding='utf8')
    (out/'test_results.json').write_text(json.dumps({'tests_run':result.testsRun,'passed':result.wasSuccessful(),
        'failures':[{'test':str(t),'traceback':m} for t,m in result.failures],
        'errors':[{'test':str(t),'traceback':m} for t,m in result.errors]},ensure_ascii=False,indent=2),encoding='utf8')
    raise SystemExit(0 if result.wasSuccessful() else 1)
