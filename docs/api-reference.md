# Chronic Disease Prediction Service API Integration Guide

Training provenance, held-out test results, upload templates and online demo files for the five algorithms are collected in the project's generated result folder, which is not part of the published repository.

Paths written as `dir/file` in this document are relative to the repository root.

This service accepts health data, computes with already-trained models, and returns prediction results plus health advice. The deliverable is the API only; the bundled validation page exists to verify that the endpoints work and is not the end-user front end of a platform. A platform front end should call the HTTP endpoints described in this document directly.

## Adapter for the published classifier (v3.1)

This repository contains an adapter around the classifier published in the paper cited in [`ATTRIBUTION.md`](../ATTRIBUTION.md). Prediction runs locally and needs no large-model key. Open the adapter group in `/docs` to try it.

| Feature | Request | Input | Output |
|---|---|---|---|
| List models | GET /api/v2/paper-models | none | Model version, fields and task |
| Hypertension classification | POST /api/v2/hypertension/predict | JSON, `records` holding 1-1000 examination records | Risk class and probability for each record |
| Hypertension whole-table classification | POST /api/v2/hypertension/predict-file | CSV/TSV upload, at most 1000 rows | All classification results in original row order |

These endpoints return results directly. They do not require creating a user first and do not store input in any user profile. A front end can associate the returned values with its own user identifiers. The existing advice and profile endpoints are called as described further below.

The author weights are not redistributed with this repository, so these endpoints answer `503` until you place `artifacts/papers/hypertension/model.joblib` next to the code; see [`ATTRIBUTION.md`](../ATTRIBUTION.md).

### Hypertension classification: what to send

One row is one person's examination record. See `service/templates/hypertension.csv` for a CSV example:

```csv
male,age,currentSmoker,cigsPerDay,BPMeds,diabetes,totChol,sysBP,diaBP,BMI,heartRate,glucose
1,50,0,0,0,0,200,130,80,25,70,90
```

| Field | Meaning and unit |
|---|---|
| male | Sex code, 1 male, 0 female |
| age | Age in years |
| currentSmoker, BPMeds, diabetes | Whether the person smokes, takes blood-pressure medication, or has a diabetes history; all 0/1 |
| cigsPerDay | Cigarettes per day |
| totChol, glucose | Total cholesterol and glucose, both in mg/dL; mmol/L values cannot be entered directly |
| sysBP, diaBP | Already measured systolic and diastolic pressure, mmHg |
| BMI, heartRate | Body mass index and heart rate in beats per minute |

All column names must be present and the order does not matter. `sysBP` and `diaBP` must have values; other fields may be `null` in JSON or left empty in a table, and the service fills them with training medians while reporting which fields were filled in `imputed_fields`. A CSV may carry the original answer column `Risk`, which does not take part in prediction.

The JSON form follows, with the same fields as the CSV:

```json
{"records":[{"male":1,"age":50,"currentSmoker":0,"cigsPerDay":0,"BPMeds":0,"diabetes":0,"totChol":200,"sysBP":130,"diaBP":80,"BMI":25,"heartRate":70,"glucose":90}]}
```

Each entry of the returned `predictions` contains `row` (input row number starting at 1), `risk_label` (0/1), `risk_probability` (the model probability of Risk=1) and `imputed_fields`. The response also returns `model_id` so the caller can confirm which model was used.

The algorithm judges the dataset's current Risk label; it is **not** the probability of developing the condition in the coming years. The parameters are the authors' published ones (decision tree, RFE feature selection and isotonic calibration, `seed=42`). On the authors' single 424-row test split the metadata records 0.9104 accuracy, 0.9084 recall and 0.0696 Brier; the paper's headline figures are means over 30 splits. Neither is a new evaluation performed here.

### Calling it from the command line

```bash
python examples/call_paper_api.py hypertension
python examples/call_paper_api.py hypertension --file service/templates/hypertension.csv
```

On Windows use the installed `.venv\Scripts\python.exe`; on Linux activate the virtual environment and use `python`. The service must be running first. The script prints results in the terminal; add `--base http://server-address:port` for a remote service. When a token is configured, the client must set the same `CHRONIC_API_TOKEN` environment variable.

## How to run on Windows

64-bit Python 3.12 or 3.13 is required. On the first run in this directory, execute `setup.cmd` to install dependencies, then start the service with `start-api.cmd` and open **http://127.0.0.1:8765/docs**. CPU is sufficient; no graphics card is needed.

Local settings may be written to `.env` in the format shown by `.env.example`. Numeric prediction needs no key; configure the project token for remote access; the existing Kimi advice feature uses your own key. The shared package ships an empty configuration template only.

## How to run

Linux environment: Python 3.12 or 3.13. Create and activate a virtual environment first; the interface uses the FastAPI framework.

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements-service.txt

export CHRONIC_API_TOKEN="project-access-token"
export CHRONIC_STORAGE="./data"   # directory for user profiles, uploads and reports

python -m service.server
```

The service exposes HTTP endpoints. For cross-origin browser calls, configure explicit `CHRONIC_ALLOWED_ORIGINS`; wildcards are not used.

## Which models are used

Current work focuses on chronic disease classification. Continuous glucose prediction still needs improvement and is kept as an experimental feature; a runnable endpoint does not mean the accuracy target has been met.

The two paper algorithms above are called through separate endpoints. The following are the three existing models and the user profile flow that remain available.

This deliverable does not include the training pipeline. Computation loads saved models and preprocessing parameters; no fitting happens during a request.

- **Health survey**: standardized logistic regression plus a gradient boosting classifier, classifying a fixed-field questionnaire (current diabetes status, not future incidence).
- **Heart disease examination table**: logistic regression or random forest, binary classification on the UCI Cleveland format. `0` means the heart disease status is not marked, `1` means it is marked; this cannot be used to declare a person healthy. It is the current examination status, not future incidence.
- **Continuous glucose**: standardized ridge regression plus gradient boosting regression, forecasting glucose 30 and 60 minutes ahead, together with an independent classifier and thresholds for high and low risk. Inference requires 12 consecutive points at 15-minute intervals for the same user, and does not bridge missing segments.

## How data flows

Store basic user information, then import examination records and check the format, then call the matching existing model, then save the prediction result, then generate advice, and finally return it to the project platform or the validation page through the API.

## How users and results are stored

User profiles, raw records and analysis results are kept separately:

- `users`: business identity data such as name, age, sex, height and weight.
- `imports`: import metadata; the original file is stored separately in the data directory.
- `analyses`: analysis tasks and final reports; the report carries a snapshot of the user profile at analysis time for later traceability.

The three kinds of data live in different tables of one SQLite database and are linked by `user_id`; they are not three separate databases. The storage root is set by `CHRONIC_STORAGE`.

## Trying the API without writing code

After startup open **http://localhost:8765/docs**. This is the API's own usage and testing page, not a platform front end.

1. If an access token is configured, click **Authorize** in the top right and enter the token (do not add the `Bearer` prefix there).
2. Expand the create-user endpoint under "1. User profiles", click **Try it out**, fill in the fields and click **Execute**. The returned user id is shown below.
3. Expand "2. Import data", supply the `user_id` and choose a CSV/Excel file, then execute to obtain an import id.
4. Under "3. Model computation and results" submit the `import_id`, then query the result with the returned analysis id.

The page shows required parameters, formats, examples and real responses. The page assets are bundled in the package, so no external site is needed to load it.

## Call order

The typical flow is **create user -> upload file -> start analysis -> poll result**. Cross-scenario reads use the integration endpoint.

| Step | Request | Send | Receive |
|---|---|---|---|
| Create user | POST /api/v2/users | Name and optional basic information | User id |
| Upload examination records | POST /api/v2/imports | `user_id` and a CSV/Excel file | Import id, preview and any information still required |
| Start computation | POST /api/v2/analyses | `import_id`; for multi-row imports add `row` to select one row | Analysis id |
| Read result | GET /api/v2/analyses/{id} | Analysis id | Status, classification result or glucose forecast, advice |

Each step returns an id used by the next one. If a file does not state the glucose unit, supply `unit` when starting the analysis; if the response has `needs_row=true`, choose the record belonging to the current user with `row` (zero based). During computation, poll the result every 1-2 seconds.

Runnable client code lives in `examples/call_api.py`; this README does not repeat program code.

`status` values:

- `completed`: at least one record finished model computation.
- `needs_data`: records were read but are not sufficient for a prediction.
- `failed`: computation did not finish; inspect `error` and submit again.

Key report fields:

- `headline`: textual description of the chronic disease classification result.
- `class_scores`: model scores per class, not future incidence probabilities.
- `metrics`: structured indicators such as the latest glucose value.
- `series`: trend points, at most the latest 2000; use the download endpoint for complete data.
- `forecasts`: predicted future glucose values and times.
- `recommendations`: list of health advice items, each with `text`; `recommendation_engine` states the source.
- `issues`: description of data problems.

Full CSV download: `GET /api/v2/analyses/{id}/download`. User history: `GET /api/v2/users/{id}/analyses`. User list: `GET /api/v2/users`; single user: `GET /api/v2/users/{id}`.

### 5. Integration read (for multiple scenarios)

When another business scenario needs chronic disease data, call:

Endpoint: `GET /api/v2/users/{id}/integration`.

The response is normalized JSON with `profile` (current user information) and `results` (the latest 20 analysis results). The link field is `user_id`; name matching is not used.

## What data can be sent

Files may be CSV, XLSX or XLS, up to 32 MB and 100000 rows; for Excel the first worksheet is read. Data containing only name, age or heart rate cannot substitute for the fields a model needs.

The three data source formats follow; the service detects them automatically after upload, so the front end does not specify model paths or field mappings.

**User profile**
- `name`: required, 1-100 characters.
- `age`, `sex` (male/female/unknown), `height_cm`, `weight_kg`: optional.

**Continuous glucose**
- CSV or Excel, first worksheet. Encoding UTF-8 or GB18030.
- Automatically recognizes "measurement time" and "glucose (mmol/L)" columns as well as standard device column names.
- One import binds to one user; glucose prediction requires 12 consecutive points at 15-minute intervals.

**Health survey**
- CSV/Excel that must contain the following fields as integers within the stated ranges:
  - HighBP (0-1), HighChol (0-1), CholCheck (0-1), BMI (12-98), Smoker (0-1), Stroke (0-1), HeartDiseaseorAttack (0-1), PhysActivity (0-1), Fruits (0-1), Veggies (0-1), HvyAlcoholConsump (0-1), AnyHealthcare (0-1), NoDocbcCost (0-1), GenHlth (1-5), MentHlth (0-30), PhysHlth (0-30), DiffWalk (0-1), Sex (0-1), Age (1-13), Education (1-6), Income (1-8).

**Heart disease examination table**
- CSV/Excel with columns: age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal, num (the original label, optional, not used for prediction).
- A raw Cleveland `.data` file without a header is also accepted.
- `ca` and `thal` may be missing.

Downloadable templates: `GET /api/v2/templates/{kind}` with `kind` set to `heart`, `survey` or `glucose`.

## Error codes

The service returns a uniform error body `{"error": "readable message"}`:

- 400: invalid input
- 401: not logged in or wrong token
- 403: invalid request origin
- 404: id does not exist
- 409: result not finished
- 413: file above 32 MB
- 422: required parameter missing, wrong type or wrong value; `details` names the offending fields
- 500: service failure

The readable `error` strings emitted by the reference implementation are currently Chinese.

## Out of scope for this API

Model training scripts (for example `train_table.py`, `chronic/training.py` and similar) belong to the offline modelling and validation workflow and are not part of this service deliverable. The service provides inference endpoints over already-trained models only.

The complete machine-readable definition is in `service/openapi.json`; it is also available as `/openapi.json` while the service runs. Older `/api/v1` routes stay registered for compatibility but are excluded from the generated schema.

## How to enable Kimi advice

Glucose values and classification are computed by the machine-learning models described above. When a large model should turn the result into advice, use the **Kimi API, not the OpenAI API**. Set this before startup:

```bash
export KIMI_API_KEY="your-kimi-key"
# optional: export KIMI_MODEL="a Kimi model available to your account"
python -m service.server
```

With a key configured, an analysis calls Kimi by default. The analysis JSON may also specify `"advice_engine":"kimi"`; use `"advice_engine":"rules"` when only local rules are wanted. Without a key the local rules are used by default. Kimi receives only a prediction summary, never names, user ids or whole files.

`recommendation_engine="kimi_api"` in the response means the Kimi advice succeeded. If the call fails, the prediction is kept, `advice_status="failed"` and `advice_error` explain why, and the local rules are retained with their source marked so that they are never passed off as Kimi output.

## Where the validation page is

`validation/` holds a separate validation page used to upload files, view predictions and check that the API works. It is not the official front end to hand to a platform. After starting the API, open `http://localhost:8765/` to use it; the API still runs when this directory is not shipped.

Main package contents: `service/` is the API code and input templates, `chronic/` and `artifacts/` are the computation code and models, and `requirements-service.txt` lists the dependencies. This README is the complete usage guide; there is no need to jump between several documents.
