# HTTP interface for the published classifier adapter

Python 3.12/3.13. Install with `pip install -r requirements.txt`; run with `uvicorn paper_api:app --host 127.0.0.1 --port 8765`. The endpoint definitions are served at `/docs`.

This adapter exposes **one** model: the hypertension classifier published by the paper's authors. Citation, author list and the exact boundary of what is and is not redistributed are in [`ATTRIBUTION.md`](../ATTRIBUTION.md). The author's weights are not part of this repository, so the endpoint returns `503` until `artifacts/papers/hypertension/model.joblib` is placed locally.

`requirements.txt` (scikit-learn, pandas, joblib, openpyxl, xlrd) is enough for this adapter; no torch wheel is needed.

You can also register `service.papers.router` into an existing FastAPI application; keep the model directory available. Authentication, CORS and deployment are managed by the integrating platform. The model computes locally and needs no large-model key.

## Hypertension classification

`POST /api/v2/hypertension/predict`, JSON:

```json
{"records":[{"male":1,"age":50,"currentSmoker":0,"cigsPerDay":0,"BPMeds":0,"diabetes":0,"totChol":200,"sysBP":130,"diaBP":80,"BMI":25,"heartRate":70,"glucose":90}]}
```

Fields: sex (1 male, 0 female), age, current smoker, cigarettes per day, on blood-pressure medication, diabetes history, cholesterol, systolic pressure, diastolic pressure, BMI, heart rate, glucose. Binary fields are 0/1; blood pressure is in mmHg and cholesterol and glucose are in mg/dL. All field names must be present, blood pressure values are mandatory, other values may be null, and each request carries 1-1000 records.

Returns a `predictions` array; each entry has `row` (input row number), `risk_label` (0/1), `risk_probability` and `imputed_fields` (fields the model filled in). The value classifies the dataset's current Risk label and does not represent the probability of developing the condition in the future.

For a whole table use `POST /api/v2/hypertension/predict-file` with the multipart field `file`, uploading a UTF-8 CSV/TSV whose header uses the field names above in any order; an accompanying `Risk` column is accepted and excluded from inference.

`GET /api/v2/paper-models` reports the model id, input fields and the authors' reported validation figures. Format errors return 422; an oversized upload returns 413. No user data is stored.
