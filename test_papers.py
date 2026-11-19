"""Contract tests for the adapter around the published hypertension classifier.

The author weights and reference predictions are not part of this repository,
so tests that need them are skipped unless the files are supplied locally.
"""
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

from service.app import create_app

ROOT = Path(__file__).parent
ARTIFACT = ROOT/'artifacts/papers/hypertension/model.joblib'
RECORD = dict(male=1, age=50, currentSmoker=0, cigsPerDay=0, BPMeds=0, diabetes=0,
              totChol=200, sysBP=130, diaBP=80, BMI=25, heartRate=70, glucose=90)


class PaperAPITest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.env = patch.dict(os.environ, {'CHRONIC_API_TOKEN': '', 'KIMI_API_KEY': ''})
        cls.env.start()
        cls.client = TestClient(create_app(cls.temp.name))
        cls.client.__enter__()

    @classmethod
    def tearDownClass(cls):
        cls.client.__exit__(None, None, None)
        cls.env.stop()
        cls.temp.cleanup()

    def test_reject_invalid_records(self):
        for record in [{**RECORD, 'male': 2}, {**RECORD, 'diaBP': 200},
                       {**RECORD, 'sysBP': None}, {**RECORD, 'unknown': 1}]:
            with self.subTest(record=record):
                self.assertEqual(self.client.post(
                    '/api/v2/hypertension/predict', json={'records': [record]}).status_code, 422)
        self.assertEqual(self.client.post(
            '/api/v2/hypertension/predict', json={'records': []}).status_code, 422)
        self.assertEqual(self.client.post(
            '/api/v2/hypertension/predict', json={'records': [RECORD]*1001}).status_code, 422)

    def test_reverse_column_order_is_accepted(self):
        frame = pd.DataFrame([RECORD, {**RECORD, 'sysBP': 160, 'diaBP': 100, 'glucose': None}])
        frame['Risk'] = [1, 0]
        raw = frame[list(reversed(frame.columns))].to_csv(index=False).encode()
        result = self.client.post('/api/v2/hypertension/predict-file',
                                  files={'file': ('records.csv', raw)})
        # 503 means the request reached the adapter but the author weights are absent;
        # 200 means they are present. Anything else is a contract regression.
        self.assertIn(result.status_code, (200, 503), result.text)

    def test_upload_errors(self):
        result = self.client.post('/api/v2/hypertension/predict-file',
                                  files={'file': ('records.txt', b'a,b\n1,2\n')})
        self.assertEqual(result.status_code, 422)
        with patch('service.papers.MAX_UPLOAD_BYTES', 10):
            result = self.client.post('/api/v2/hypertension/predict-file',
                                      files={'file': ('big.csv', b'x'*11)})
            self.assertEqual(result.status_code, 413)

    def test_docs_and_auth(self):
        schema = self.client.get('/openapi.json').json()
        self.assertEqual(self.client.get('/docs').status_code, 200)
        self.assertEqual(
            schema['paths']['/api/v2/hypertension/predict']['post']['responses']['200']
            ['content']['application/json']['schema']['$ref'],
            '#/components/schemas/ClassificationResult')
        # The unpublished waveform endpoint is not part of this snapshot.
        self.assertNotIn('/api/v2/blood-pressure/estimate', schema['paths'])
        with patch.dict(os.environ, {'CHRONIC_API_TOKEN': 'paper-test-token'}):
            self.assertEqual(self.client.get('/api/v2/paper-models').status_code, 401)
            self.assertIn(self.client.get(
                '/api/v2/paper-models',
                headers={'Authorization': 'Bearer paper-test-token'}).status_code, (200, 503))

    def test_local_env_does_not_override_environment(self):
        from service.settings import load_local_env
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {'CHRONIC_PORT': '8769'}):
            Path(tmp, '.env').write_text(
                'CHRONIC_PORT=8999\nPAPER_TORCH_THREADS="2"\nUNRELATED_VARIABLE=ignored\n',
                encoding='utf-8')
            load_local_env(Path(tmp))
            self.assertEqual(os.environ['CHRONIC_PORT'], '8769')
            self.assertNotIn('UNRELATED_VARIABLE', os.environ)

    @unittest.skipUnless(ARTIFACT.is_file(), 'author weights are not redistributed with this repository')
    def test_classification_with_author_weights(self):
        frame = pd.DataFrame([RECORD, {**RECORD, 'sysBP': 160, 'diaBP': 100, 'glucose': None}])
        response = self.client.post('/api/v2/hypertension/predict',
                                    json={'records': frame.replace({np.nan: None}).to_dict('records')})
        self.assertEqual(response.status_code, 200, response.text)
        result = response.json()
        self.assertEqual(result['count'], 2)
        self.assertIn('glucose', result['predictions'][1]['imputed_fields'])
        self.assertEqual(self.client.post('/api/v2/hypertension/predict-file',
                                          files={'file': ('r.csv', frame.replace({np.nan: None})
                                                          .to_csv(index=False).encode())}).json(),
                         result)


if __name__ == '__main__':
    unittest.main()
