"""Local inference adapter for the published hypertension classifier.

This file contains only the adapter and input handling written for this
service. The model itself was produced by the paper authors and is not
redistributed here; see ``ATTRIBUTION.md`` and ``SOURCES.md``.
"""
from functools import lru_cache
from pathlib import Path
import csv
import io
import json
import threading

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT/'artifacts/papers'
MAX_UPLOAD_BYTES = 32 * 1024 * 1024
MAX_ROWS = 150000

# Feature order used by the published classification model. These are the
# dataset column names, not model internals.
FEATURE_COLS = ['male', 'age', 'currentSmoker', 'cigsPerDay', 'BPMeds', 'diabetes',
                'totChol', 'sysBP', 'diaBP', 'BMI', 'heartRate', 'glucose']


def read_table(raw: bytes, filename: str) -> pd.DataFrame:
    """Parse an actual numeric CSV/TSV, retaining header order and rejecting duplicates."""
    if not raw:
        raise ValueError('文件为空')
    if len(raw) > MAX_UPLOAD_BYTES:
        raise ValueError('文件超过 32MB')
    if Path(filename or '').suffix.lower() not in {'.csv', '.tsv'}:
        raise ValueError('请上传带表头的 CSV 或 TSV 文件')
    try:
        text = raw.decode('utf-8-sig')
    except UnicodeDecodeError as exc:
        raise ValueError('文件编码须为 UTF-8') from exc
    if not text.strip():
        raise ValueError('文件为空')
    sep = '\t' if '\t' in text.splitlines()[0] else ','
    header = next(csv.reader(io.StringIO(text), delimiter=sep))
    if len(header) != len(set(header)):
        raise ValueError('表头有重复列名')
    try:
        frame = pd.read_csv(io.StringIO(text), sep=sep, nrows=MAX_ROWS+1)
    except (pd.errors.ParserError, pd.errors.EmptyDataError) as exc:
        raise ValueError('无法读取表格，请检查分隔符和列数') from exc
    if frame.empty:
        raise ValueError('表格没有数据行')
    if len(frame) > MAX_ROWS:
        raise ValueError(f'最多支持 {MAX_ROWS} 行')
    return frame


class PaperModels:
    """Loads the author-provided model artifact and serves one classification task."""

    def __init__(self):
        model_path = ARTIFACTS/'hypertension/model.joblib'
        meta_path = ARTIFACTS/'hypertension/metadata.json'
        if not model_path.is_file():
            raise FileNotFoundError(
                'artifacts/papers/hypertension/model.joblib is missing. '
                'The author weights are not redistributed with this repository; '
                'see ATTRIBUTION.md for how to obtain them.')
        self.lock = threading.Lock()
        self.hypertension = joblib.load(model_path)
        self.hypertension_meta = json.loads(meta_path.read_text(encoding='utf-8'))

    def classify(self, records: list[dict]):
        frame = pd.DataFrame(records)[FEATURE_COLS].astype(float)
        probabilities = self.hypertension.predict_proba(frame)[:, 1]
        return {
            'model_id': self.hypertension_meta['model_id'],
            'target': 'Risk', 'threshold': .5,
            'meaning': '数据集 Risk 标签分类；不是未来发病概率或临床诊断',
            'count': len(frame),
            'predictions': [
                {'row': i+1, 'risk_label': int(p >= .5), 'risk_probability': float(p),
                 'imputed_fields': frame.columns[frame.iloc[i].isna()].tolist()}
                for i, p in enumerate(probabilities)
            ],
        }


@lru_cache(maxsize=1)
def get_models():
    return PaperModels()
