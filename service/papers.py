"""HTTP contract for the adapter around the published hypertension classifier.

Only the published paper's classifier is exposed here. Provenance, citation and
what is *not* redistributed are documented in ``ATTRIBUTION.md``.
"""
from typing import Literal

import numpy as np
from fastapi import APIRouter, File, HTTPException, UploadFile
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from paper_models.runtime import FEATURE_COLS, MAX_UPLOAD_BYTES, get_models, read_table

router = APIRouter(prefix='/api/v2', tags=['论文算法适配'])


class StrictInput(BaseModel):
    model_config = ConfigDict(extra='forbid', allow_inf_nan=False)


class HypertensionRecord(StrictInput):
    male: Literal[0, 1] | None
    age: float | None = Field(ge=0, le=130)
    currentSmoker: Literal[0, 1] | None
    cigsPerDay: float | None = Field(ge=0, le=200)
    BPMeds: Literal[0, 1] | None
    diabetes: Literal[0, 1] | None
    totChol: float | None = Field(gt=0, le=2000, description='mg/dL')
    sysBP: float = Field(gt=0, le=400, description='已测得的收缩压，mmHg')
    diaBP: float = Field(gt=0, le=300, description='已测得的舒张压，mmHg')
    BMI: float | None = Field(gt=0, le=150)
    heartRate: float | None = Field(gt=0, le=350)
    glucose: float | None = Field(gt=0, le=2000, description='mg/dL，不是 mmol/L')

    @model_validator(mode='after')
    def check_bp(self):
        if self.sysBP < self.diaBP:
            raise ValueError('sysBP 收缩压不能小于 diaBP 舒张压')
        return self


class HypertensionInput(StrictInput):
    records: list[HypertensionRecord] = Field(min_length=1, max_length=1000)
    model_config = ConfigDict(extra='forbid', allow_inf_nan=False, json_schema_extra={
        'example': {'records': [{'male': 1, 'age': 50, 'currentSmoker': 0, 'cigsPerDay': 0,
             'BPMeds': 0, 'diabetes': 0, 'totChol': 200, 'sysBP': 130, 'diaBP': 80,
             'BMI': 25, 'heartRate': 70, 'glucose': 90}]}})


class ClassificationRow(BaseModel):
    row: int
    risk_label: Literal[0, 1]
    risk_probability: float
    imputed_fields: list[str]


class ClassificationResult(BaseModel):
    model_id: str
    target: str
    threshold: float
    meaning: str
    count: int
    predictions: list[ClassificationRow]


def models_or_503():
    """The author weights are not part of this repository; report that plainly."""
    try:
        return get_models()
    except FileNotFoundError as exc:
        raise HTTPException(503, str(exc)) from exc


def uploaded_table(file: UploadFile):
    raw = file.file.read(MAX_UPLOAD_BYTES+1)
    if len(raw) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, '文件超过 32MB')
    try:
        return read_table(raw, file.filename or '')
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc


@router.get('/paper-models', summary='查看已发表论文模型适配层的字段与版本')
def catalog():
    models = models_or_503()
    return {'models': [
        {'id': models.hypertension_meta['model_id'], 'task': '高血压标签分类',
         'inputs': FEATURE_COLS, 'output': ['risk_label', 'risk_probability'],
         'endpoint': '/api/v2/hypertension/predict',
         'validation': models.hypertension_meta['test'],
         'provenance': 'published paper, cited in ATTRIBUTION.md'}]}


@router.post('/hypertension/predict', response_model=ClassificationResult,
             summary='高血压分类：输入一条或多条检查记录')
def hypertension(payload: HypertensionInput):
    return models_or_503().classify([row.model_dump() for row in payload.records])


@router.post('/hypertension/predict-file', response_model=ClassificationResult,
             summary='高血压分类：上传整张 CSV/TSV，最多 1000 行')
def hypertension_file(file: UploadFile = File(...)):
    frame = uploaded_table(file)
    missing = set(FEATURE_COLS)-set(frame.columns)
    if missing:
        raise HTTPException(422, '缺少输入列：'+', '.join(sorted(missing)))
    extra = set(frame.columns)-set(FEATURE_COLS)-{'Risk'}
    if extra:
        raise HTTPException(422, '未知列：'+', '.join(sorted(extra)))
    # Ground truth Risk, when present, is explicitly excluded from inference.
    records = frame[FEATURE_COLS].replace({np.nan: None}).to_dict(orient='records')
    try:
        validated = HypertensionInput(records=records)
    except ValidationError as exc:
        errors = [{'field': '.'.join(map(str, e['loc'])), 'message': e['msg']}
                  for e in exc.errors()[:10]]
        raise HTTPException(422, {'message': '检查表字段不符合要求', 'details': errors}) from exc
    return hypertension(validated)
