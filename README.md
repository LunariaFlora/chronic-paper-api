# Chronic Disease Prediction API

A self-hosted FastAPI service that accepts examination records, runs already-trained models locally, and returns JSON. CPU only, no GPU and no LLM key required for the numeric endpoints, tested on Linux and Windows.

> **What this repository is, and what it is not.** This is an engineering deliverable: the HTTP layer, input validation, storage, deployment files and tests. The classifier exposed under `/api/v2/hypertension/*` was trained by other authors; [`ATTRIBUTION.md`](ATTRIBUTION.md) records the citation, the author list and the exact boundary. Their code, weights and data are **not** redistributed here, and nothing in this repository is a research contribution of the maintainer.

## What it does

The service exposes two families of functionality over one HTTP port.

**Adapter for the published classifier.** One published research algorithm is wrapped as an inference-only HTTP endpoint:

- Hypertension label classification from a routine examination record (12 fields), submitted as JSON or as a whole CSV/TSV table, up to 1000 rows per request. The endpoint answers `503` until the author weights are placed locally, because they are not redistributed here.

Both families compute locally, return JSON synchronously, require no user account, and do not persist the submitted data.

**Existing analysis workflow (kept from earlier versions).** A three-model pipeline with optional user profiles:

- Health survey classification (BRFSS-style questionnaire, current diabetes status).
- Heart disease status classification from a UCI Cleveland style examination table (`0` = not marked, `1` = marked in the source label set).
- Continuous glucose forecasting 30 and 60 minutes ahead, plus high/low risk assessment from an independent classifier and thresholds.

That workflow runs as create user -> upload records -> start analysis -> poll result, and optionally turns the numeric result into advice through a Kimi (Moonshot) chat completion call. Kimi only receives a prediction summary, never names, user IDs or whole files.

## Tech stack

| Layer | Component |
|---|---|
| Runtime | Python 3.12 or 3.13 (64-bit), `venv` |
| Web framework | FastAPI `0.162.0`, Starlette (pulled in), uvicorn `0.54.0` |
| Uploads and sessions | python-multipart `0.0.33`, itsdangerous `2.2.0` |
| Numerics | numpy `2.5.4`, pandas `3.1.0`, scikit-learn `1.9.1`, joblib `1.6.0` |
| Persistence | SQLite from the standard library, files under `CHRONIC_STORAGE` |
| Container | `Dockerfile` on `python:3.12-slim`, `compose.yaml` with a named data volume |
| Optional LLM advice | Kimi Chat Completions endpoint `https://api.moonshot.cn/v1/chat/completions` |

The API application declares version `3.1.0`; the `/healthz` probe reports `{"status": "ok", "service": "chronic", "version": "2.0"}`. Both values are read from the code as published.

## Directory structure

```
.
|-- service/                 FastAPI application
|   |-- app.py               App factory, auth middleware, CORS, error handlers, docs mount
|   |-- server.py            Production entry: env loading, model presence check, single-owner lock
|   |-- business.py          Users, imports, analyses, integration, templates, /healthz
|   |-- papers.py            The published-classifier adapter and its strict request models
|   |-- legacy.py            Older /api/v1 routes, hidden from the OpenAPI schema
|   |-- settings.py          Reads allowlisted keys from a local .env file
|   |-- schemas.py           Pydantic models for user and login payloads
|   |-- templates/           Input templates served by GET /api/v2/templates/{kind}
|   |-- docs_assets/         Offline Swagger UI bundle (Apache-2.0)
|   `-- openapi.json         Extra response schemas merged into the generated OpenAPI document
|-- chronic/                 Offline prediction, batch, workflow, metrics and advice code
|-- paper_models/            Provenance notes and the input adapter for the published classifier
|   |-- runtime.py           Model loading, table reading, upload limits, input adaptation
|   |-- SOURCES.md           Provenance of the author packages, and what is *not* relicensed here
|   `-- (author source packages are not redistributed; obtain them from the paper authors)
|-- configs/                 Task schemas, e.g. heart_cleveland.json, plus copy-me examples
|-- deploy/                  systemd unit, environment template, nginx snippet, Linux notes
|-- docs/                    English API documentation (api-reference.md, paper_api_README.md)
|-- examples/                call_api.py, call_paper_api.py
|-- scripts/                 prepare_hypertension_model.py, package_paper_service.py
|-- validation/              Small HTML page for manual checks; not a production front end
|-- datasets/                Data notes only; the data files themselves are not published
|-- artifacts/               Trained models and held-out results; required at runtime
|-- outputs/                 Generated run results
|-- start_service.sh         Linux entry point
|-- start-api.cmd            Windows entry point (English-named copy of the original launcher)
|-- setup.cmd                Windows one-time environment setup
|-- run_demo.cmd             Windows launcher for demo.py
|-- run_batch.cmd            Windows launcher for run_batch.py
|-- run_pipeline.cmd         Windows launcher for run_pipeline.py
|-- demo.py                  Offline demo on the bundled held-out examples
|-- predict.py               CLI for a single JSON input
|-- batch_predict.py         CLI for whole-table batch prediction
|-- pipeline.py              CLI for the full local train/evaluate/predict workflow
|-- paper_api.py             Standalone app exposing only the published-classifier endpoint
|-- run_tests.py             Test runner (see Tests)
|-- test_*.py                unittest modules
|-- requirements-service.txt Full service dependencies
|-- requirements.lock.txt    Pinned data-science stack
|-- requirements-paper.txt   Optional CPU torch pin for deployments that add their own models
|-- requirements-paper-api.txt Minimal adapter dependency set
|-- Dockerfile, compose.yaml, .dockerignore
|-- .env.example             Configuration template
`-- .gitignore               Excludes .venv, datasets, artifacts, outputs, work, service_data, .env
```

The published documentation set is English only: `README.md`, `docs/api-reference.md` and `docs/paper_api_README.md`, plus the translated notes under `datasets/`, `paper_models/` and `deploy/`. The earlier Chinese-language source documents remain in the working tree without being part of the English release.

`artifacts/`, `datasets/`, `outputs/` and `work/` are excluded by `.gitignore` in this snapshot. The adapter needs the author weights to produce predictions, so they have to be supplied alongside the checkout; the rest of the service starts without them.

## Requirements

- 64-bit Python 3.12 or 3.13. The deployment target is Linux x86_64; Windows is supported through the `.cmd` launchers.
- Enough free disk for the virtual environment. The adapter needs no torch wheel; only add one if you attach your own torch-based models.
- No GPU, no system database, no external service for the numeric endpoints.

### Linux (main path)

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-service.txt
```

`requirements-service.txt` pulls in `requirements.lock.txt`, the web stack, and `requirements-paper.txt` (CPU torch). Install and run with the interpreter inside `.venv` so the torch wheel matches the environment.

### Windows

One-time setup, or double-click:

```bat
setup.cmd
```

`setup.cmd` creates `.venv` if it is missing, installs `requirements-service.txt`, and copies `.env.example` to `.env` when `.env` does not exist yet.

Equivalent manual steps in a terminal at the repository root:

```bat
python -m venv .venv
".venv\Scripts\python.exe" -m pip install -r requirements-service.txt
copy /y ".env.example" ".env"
```

### Docker

```bash
docker compose up --build
```

`compose.yaml` requires `CHRONIC_API_TOKEN` to be set (the compose file fails fast when it is empty), sets `CHRONIC_TIMEZONE=Asia/Shanghai`, publishes the port on `127.0.0.1:8765`, and keeps state in the `chronic-data` volume mounted at `/data`. The image runs as an unprivileged user (uid 10001) and ships a `/healthz` health check.

## Model files

`service/server.py` refuses to start when a required model is missing and prints `Missing required model: <path>`. The complete list is:

```
artifacts/v2/survey.joblib
artifacts/v2/glucose.joblib
artifacts/table_heart_v1/table_model.joblib
artifacts/papers/hypertension/model.joblib
```

The Docker image copies exactly these files, which is why `.dockerignore` allowlists `artifacts/v2`, `artifacts/table_heart_v1` and `artifacts/papers` only.

## Running the service

### Linux

```bash
sh start_service.sh
```

The script changes into its own directory and executes `${CHRONIC_PYTHON:-.venv/bin/python} -m service.server`; set `CHRONIC_PYTHON` to use another interpreter. The equivalent explicit command, after activating the environment, is:

```bash
python -m service.server
```

On startup the server loads any allowlisted keys from `.env`, sets a restrictive umask (`0o077`), checks the models above, creates the storage directory, takes an exclusive lock on `<storage>/server.lock` (POSIX `flock`, so a second process on the same storage directory exits with `Another service already owns this storage directory.`), and then serves with uvicorn on `CHRONIC_HOST:CHRONIC_PORT`.

### Windows

```bat
start-api.cmd
```

`start-api.cmd` expects `.venv\Scripts\python.exe` and prints `Run setup.cmd first.` when it is absent, then runs `python -m service.server` from the repository root. The original launcher with a non-ASCII filename is kept in the repository; `start-api.cmd` is an identical English-named copy so that the documented command line stays ASCII.

### Standalone paper-only app

`paper_api.py` mounts only the paper router and preloads the paper models at startup:

```bash
uvicorn paper_api:app --host 127.0.0.1 --port 8765
```

This entry has no token middleware, no CORS configuration and no user storage; authentication, CORS and process management are then the responsibility of the integrating platform. `service.papers.router` can also be mounted into an existing FastAPI application.

### Where to look after startup

- `http://127.0.0.1:8765/docs` - interactive OpenAPI page, served from the bundled assets so it works offline. Expand a group, use **Try it out -> Execute**. If a token is configured, click **Authorize** first and paste the raw token (no `Bearer` prefix in that field).
- `http://127.0.0.1:8765/openapi.json` - machine-readable schema.
- `http://127.0.0.1:8765/healthz` - liveness probe.
- `http://127.0.0.1:8765/` - the small HTML validation page; it is a functional check page, not a production front end, and the API still runs when the `validation/` directory is not shipped.

### Demo and command-line tools

```bash
python demo.py
```

`demo.py` runs the completed local prediction flow on the bundled held-out examples in `artifacts/v2/`, prints glucose forecasts with risk flags, and writes `outputs/completion/demo_results.json`. It needs `artifacts/v2/` (model files plus the `*_input.json` / `*_truth.json` examples). On Windows use `run_demo.cmd`.

Other offline tools: `predict.py` (single JSON input, `diabetes` or `glucose`), `batch_predict.py` (whole table to `results.csv`, `issues.csv`, `summary.json`), `pipeline.py` (local train, held-out evaluation, batch prediction, optional advice), `run_batch.py` and `run_pipeline.py` (interactive launchers behind the `.cmd` files). These are analysis tools, not part of the HTTP service contract.

## Configuration

Configuration is environment variables, optionally supplied through a `.env` file in the repository root. `service/settings.py` reads only an allowlist of keys, strips optional matching quotes, and uses `os.environ.setdefault`, so variables already exported in the shell always win. A missing or malformed `.env` is ignored rather than fatal.

| Variable | Default | Purpose |
|---|---|---|
| `CHRONIC_HOST` | `127.0.0.1` | Bind address. The server refuses to bind anything other than loopback when `CHRONIC_API_TOKEN` is empty. |
| `CHRONIC_PORT` | `8765` | HTTP port. |
| `CHRONIC_STORAGE` | `<repo>/service_data` | SQLite database, uploads, reports, session key and process lock. Docker sets `/data`; the systemd template uses `/var/lib/chronic`. |
| `CHRONIC_API_TOKEN` | empty | Shared bearer token. When set, every request whose path starts with `/api/` needs `Authorization: Bearer <token>` (or an authenticated browser session), otherwise it returns 401. Empty means no token check, which is only allowed on loopback. |
| `CHRONIC_ALLOWED_ORIGINS` | empty | Comma-separated list of allowed CORS origins. An empty value means no cross-origin browser caller is allowed; wildcards are not used. |
| `CHRONIC_COOKIE_SECURE` | `0` | Set to `1` when the service sits behind an HTTPS reverse proxy; the session cookie is then marked HTTPS-only. |
| `CHRONIC_TIMEZONE` | `Asia/Shanghai` | Zone used for report timestamps and for localizing glucose measurements that carry no zone. |
| `PAPER_TORCH_THREADS` | `2` | Accepted for compatibility with deployments that add their own torch-based models; unused by the classifier adapter. |
| `KIMI_API_KEY` | empty | Optional. Enables the Kimi advice path; without it the service falls back to local rule-based advice. |
| `KIMI_MODEL` | `kimi-k2.5` | Optional. If the requested model is unavailable, the code checks the model list and accepts `kimi-k2.6`, otherwise it raises an explicit error. |

`.env.example` lists `CHRONIC_HOST`, `CHRONIC_PORT`, `CHRONIC_STORAGE`, `CHRONIC_API_TOKEN`, `CHRONIC_ALLOWED_ORIGINS`, `CHRONIC_COOKIE_SECURE`, `CHRONIC_TIMEZONE`, `PAPER_TORCH_THREADS`, `KIMI_API_KEY` and `KIMI_MODEL`; `deploy/chronic.env.example` covers the systemd deployment separately. Never commit a real `.env`.

Session handled by `SessionMiddleware`: `POST /session` with `{"password": "<token>"}` marks the browser session as authenticated and `DELETE /session` clears it. Both routes are excluded from the schema. Session-authenticated non-GET requests must also carry `X-Requested-With: chronic-workspace`, otherwise they are rejected with 403; that header check is a CSRF guard, not a second credential.

## API overview

Full request and response documentation, including units, validation rules, error codes and the legacy workflow, is in [docs/api-reference.md](docs/api-reference.md). The short version:

| Method and path | Input | Output |
|---|---|---|
| `GET /api/v2/paper-models` | none | Model id, input field list and task |
| `POST /api/v2/hypertension/predict` | JSON, `records` array with 1-1000 entries | Per-row `risk_label`, `risk_probability`, `imputed_fields` |
| `POST /api/v2/hypertension/predict-file` | multipart `file`, UTF-8 CSV/TSV, at most 1000 rows | Same predictions in original row order |
| `POST /api/v2/users`, `GET /api/v2/users`, `GET /api/v2/users/{id}` | Profile fields | User id and stored profile |
| `POST /api/v2/imports`, `GET /api/v2/imports/{id}/rows`, `POST /api/v2/records`, `GET /api/v2/templates/{kind}` | CSV/XLSX/XLS upload, glucose JSON, or template name | Import id, preview rows, stored record set, template download |
| `POST /api/v2/analyses`, `GET /api/v2/analyses/{id}`, `GET /api/v2/analyses/{id}/download` | `import_id` plus optional `row`, `unit`, `advice_engine` | Queued analysis id, status and report, per-row CSV |
| `GET /api/v2/users/{id}/analyses`, `GET /api/v2/users/{id}/integration` | User id | History list, or profile plus the latest 20 results |
| `GET /healthz` | none | Liveness JSON |

Error bodies are uniform: `{"error": "<message>"}`, with `details` added for 422 validation failures. Status codes used: 400 invalid input, 401 missing or wrong token, 403 invalid request origin, 404 unknown id, 409 result not finished, 413 upload above 32 MB, 422 missing or malformed parameters, 500 internal error. The human-readable messages in the current reference implementation are Chinese; the codes and the JSON shape are stable.

## Input formats

### Hypertension record

One row is one person's examination record. The CSV template is `service/templates/hypertension.csv`:

```csv
male,age,currentSmoker,cigsPerDay,BPMeds,diabetes,totChol,sysBP,diaBP,BMI,heartRate,glucose
1,50,0,0,0,0,200,130,80,25,70,90
```

| Field | Meaning and unit |
|---|---|
| `male` | Sex code, 1 male, 0 female |
| `age` | Years |
| `currentSmoker`, `BPMeds`, `diabetes` | Smoker now, on blood-pressure medication, diabetes history; all 0/1 |
| `cigsPerDay` | Cigarettes per day |
| `totChol`, `glucose` | Total cholesterol and glucose in mg/dL; mmol/L values are rejected by the declared ranges |
| `sysBP`, `diaBP` | Already measured systolic and diastolic pressure, mmHg; both required, and systolic must not be below diastolic |
| `BMI`, `heartRate` | Body mass index and heart rate in beats per minute |

All column names must be present; order does not matter. The JSON path forbids unknown fields. The file path rejects unknown columns with 422, except an optional `Risk` column, which is accepted and deliberately excluded from inference. Missing optional values may be `null` in JSON or empty in the table; the model imputes training medians and reports which fields were filled in `imputed_fields`. Declared ranges: `age` 0-130, `cigsPerDay` 0-200, `totChol` up to 2000, `sysBP` up to 400, `diaBP` up to 300, `BMI` up to 150, `heartRate` up to 350, `glucose` up to 2000.

### Other accepted data

For the legacy workflow the service accepts CSV, XLSX and XLS up to 32 MB and 100000 rows (first worksheet only), detects the dataset kind automatically, and needs `row` selection when an import holds more than one record. Templates for the three model kinds are downloadable from `GET /api/v2/templates/{kind}` with `kind` one of `heart`, `survey`, `glucose`.

## Model metrics and what the numbers mean

The classifier under `/api/v2/hypertension/*` is the authors' published model, and the figures below are the authors' reported test results, repeated here for context only. They are not a new evaluation performed for this repository:

- Hypertension tree: `model_id` `hypertension-dt-seed42-v1`, decision tree with `max_depth` 2, `min_samples_split` 20, `min_samples_leaf` 9, `selected_k` 6 features (`male`, `age`, `BPMeds`, `sysBP`, `diaBP`, `BMI`), decision threshold 0.5, split sizes 2544 / 848 / 424 / 424 (train / validation / calibration / test).

Two different sets of numbers exist and they should not be mixed:

- The authors' package metadata (`artifacts/papers/hypertension/metadata.json`, not redistributed) records a **single seed-42 split**: accuracy 0.9104, recall 0.9084, Brier 0.0696 on the 424-row test split.
- The published paper reports the **mean over 30 stratified random splits**, where the classifier's Brier score was 0.081 versus 0.090 for the clinical blood-pressure rule, with expected calibration error 0.027 versus 0.084.

Both are the authors' results. This repository reproduces neither: it loads their serialized model and does not re-evaluate it.

`risk_label` classifies the dataset's current `Risk` label. It is not the probability of developing the condition in coming years, and `risk_probability` is a calibrated model score, not a clinical risk estimate.

## Tests

`run_tests.py` is the shipped runner:

```bash
python run_tests.py
```

It runs `unittest` discovery restricted to the pattern `test_predict.py`, so only that module is executed. It writes `outputs/completion/test_results.txt` (human readable) and `outputs/completion/test_results.json` (tests run, pass flag, failures, errors), prints the log, and exits with code 1 when anything fails.

To run every test module, call discovery with the wider pattern:

```bash
python -m unittest discover -s . -p "test_*.py" -v
```

Module coverage, from the imports and test classes in each file:

| Module | What it exercises |
|---|---|
| `test_predict.py` | Input contracts (`validate_glucose`, `validate_survey`, glucose window building, unit constants), model artifacts through `ChronicPredictor`, metric helpers, and command-line entry points started in a subprocess |
| `test_batch.py` | Whole-table batch paths (`survey_batch`, `glucose_batch`, `custom_batch`, `run_batch`), table reading and preparation, local advice generation |
| `test_business.py` | The v2 business flow through a FastAPI test client: users, imports, analyses, reports |
| `test_fastapi.py` | HTTP layer behaviour of `create_app` |
| `test_service.py` | Service-level requests and lifecycle through a test client |
| `test_papers.py` | Contract tests for the published-classifier adapter; the artifact-dependent case is skipped when the author weights are absent |
| `test_workflow.py` | Dataset discovery, the local train/evaluate/predict workflow, `local_only` guard, advice isolation and response validation |
| `test_catalog.py` | Model catalog discovery and the advice session helper, including `.env` key filtering |
| `test_compact_advice.py` | Advice packet construction from a table row, terminal result rendering, saved-record loading |
| `test_selection.py` | Column selection helpers (`column_name`, `column_number`, `select_features`) |
| `test_terminal.py` | Console helpers: model matching, missing column reporting, paged result views |

Several modules are integration tests rather than pure unit tests: they load the trained artifacts and read the data files listed below. Run the suite from the repository root, in a checkout where `artifacts/` exists; modules whose artifacts are missing are skipped rather than failed.

## Data sources

The training candidates and the paper model inputs are all public datasets. `datasets/README.md` keeps the full audit notes; data files themselves are excluded from this repository and have to be obtained from the original publishers.

| Dataset | Size as checked | Role | Terms / caveats |
|---|---|---|---|
| ShanghaiT1DM and ShanghaiT2DM continuous glucose (Figshare, version 5) | 12 type-1 patients in 16 record files (15695 rows); 100 type-2 patients in 109 files (112475 rows) | Continuous glucose forecasting candidate; about 15-minute sampling with gaps | CC BY 4.0, attribution and modification notes required; no healthy controls, so it cannot support incidence prediction |
| CDC BRFSS 2015 diabetes health indicators (Kaggle `alexteboul`, UCI 891) | 253680 rows with 21 input fields; a balanced 70692-row variant | Survey classification candidate | Publisher marks CC0; treat row mapping carefully, the binary files map class 1-2 originals to 0 |
| Kaggle Diabetes Prediction (100000 rows) | 100000 rows, 8 input fields, 8500 positives | Alternative survey-style candidate | Provenance and licence not confirmed; not a first choice |
| Kaggle Cardiovascular Disease (70000 rows) | 70000 rows, 11 input fields, semicolon separated | Blood-pressure-related tabular candidate | Licence marked Unknown; visible outliers, e.g. systolic not above diastolic in 1236 rows |
| UCI Early Stage Diabetes Risk Prediction | 520 rows, 16 fields | Small pipeline demonstration only | Symptom questionnaire, 269 fully duplicated rows |
| UCI Heart Disease, Cleveland subset | 303 rows, 13 inputs | Classic tabular baseline for the heart model | 6 missing cells marked with a question mark; labels 0-4 mapped to a binary status |

The published classifier was trained by its authors on a public hypertension dataset of 4240 records (positive prevalence 31.1 percent); that dataset is not redistributed here and must be obtained from the original publisher.

## Storage and data flow

`CHRONIC_STORAGE` holds everything persistent: `records.sqlite3`, uploaded files, generated reports, `session.key` (generated on first start, mode 0600) and `server.lock`. The database keeps three logical record groups in separate tables linked by `user_id`: `users` (name, age, sex, height, weight), imports (metadata plus the stored original file), and analyses (task, request, status and the final report, including a profile snapshot taken at analysis time).

The adapter bypasses all of this: it returns JSON and stores nothing, so a platform can attach its own user identifiers to the response.

## Known limitations

- The service ships inference only. Training and evaluation scripts (`train.py`, `train_table.py`, `pipeline.py`, `chronic/training.py`, `evaluate.py`) belong to the offline modelling workflow and are not part of the HTTP contract.
- Continuous glucose forecasting still needs improvement and is documented as experimental; a working endpoint does not mean the accuracy target has been met.
- The published classifier keeps its authors' reported error levels. Nothing here re-validates its clinical accuracy.
- All labels describe the current state in the dataset, not future incidence. Nothing in this service is a diagnostic device or a substitute for clinical judgement.
- `artifacts/` and `datasets/` are not versioned in this snapshot, so a fresh clone cannot produce predictions until the model files are supplied.
- The author code package (`hypertension_source/`) is **not redistributed** here: [`ATTRIBUTION.md`](ATTRIBUTION.md) and `paper_models/SOURCES.md` document its provenance and how to obtain it.
- Authentication is a single shared token plus an optional browser session. There are no per-user accounts, scopes, rate limits or audit trail on the adapter endpoint.
- `validation/` is a manual check page. It is not the integration target, and the API works without it.
- The systemd unit, nginx snippet and Docker files are templates for the deploying party: adjust paths, user and environment file to the target host.
- Python 3.12 and 3.13 are the supported interpreters.

## License

The service code in this repository is released under the MIT licence (see `LICENSE`). Third-party material keeps its own terms, and the following separations stay visible in any distribution:

- Bundled third-party assets keep their own terms. The offline Swagger UI bundle under `service/docs_assets/` is Apache-2.0 (full text in `service/docs_assets/LICENSE`).
- Python dependencies remain under their upstream licences, as declared in their installed package metadata: FastAPI MIT; Starlette, uvicorn, itsdangerous, joblib, scikit-learn, pandas and numpy BSD-3-Clause (numpy additionally carries 0BSD, MIT, Zlib and CC0-1.0 components); python-multipart Apache-2.0.
- The published classifier, its parameters and its trained weights belong to the paper authors and are not relicensed here. Only the HTTP adapter was added; see [`ATTRIBUTION.md`](ATTRIBUTION.md).
- Datasets keep their own licences and agreements. Shanghai glucose data is CC BY 4.0 with attribution.

If you redistribute the service, also redistribute the model provenance notes in `paper_models/SOURCES.md` and the dataset notes in `datasets/README.md`.
