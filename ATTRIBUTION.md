# Attribution and provenance

This repository is an **engineering deliverable, not a research contribution**. Its purpose is to show how a trained classifier can be wrapped as a small, validated, deployable HTTP service. The science belongs to other people.

## The model this repository adapts

The classifier behind `POST /api/v2/hypertension/predict` and `POST /api/v2/hypertension/predict-file` was trained by the paper's authors, not by the maintainer of this repository.

> **Ligang Ren, Jinhua Liang, Zenan Shi, Yinhao Li, Yurong Wang, Ying Xing, Xiangfu Meng, Bin Yang, Jing Cao.**
> *Machine Learning for Hypertension Classification After Blood Pressure Measurement: An Interpretable and Calibrated Approach.*
> **Expert Systems** (Wiley), published 2026-09-06.
> DOI: [10.1111/exsy.70419](https://doi.org/10.1111/exsy.70419)

If you use the model, the method, or the reported numbers, cite that paper — not this repository.

## What this repository actually contains

- The FastAPI application, request validation, error handling and the OpenAPI documentation.
- A thin adapter (`paper_models/runtime.py`, `service/papers.py`) that loads the author's serialized model and maps HTTP input onto its expected feature order.
- Tests for the HTTP contract, the local analysis workflow, storage and deployment files.
- Documentation in English.

That is the whole of the maintainer's contribution, and it is what the MIT licence in `LICENSE` covers.

## What this repository does **not** contain

- The authors' training or inference code. It is not redistributed here and is not covered by this repository's licence.
- The authors' trained weights (`artifacts/papers/hypertension/model.joblib`). Obtain them from the authors. Without them the endpoint answers `503` with an explanatory message.
- The training data. The paper uses a public hypertension dataset of 4240 records; obtain it from its publisher.
- Any description of a second manuscript that was shared during the same collaboration but is **not publicly published**. Its method, weights and results are deliberately absent from this repository, because unpublished research is not ours to describe in public.

## Reported numbers

Every performance figure quoted in `README.md` belongs to the authors, and the two sets must not be confused:

- **Single split (authors' package metadata)**: accuracy 0.9104, recall 0.9084, Brier 0.0696 on the 424-row test split of the `seed=42` run. This comes from `artifacts/papers/hypertension/metadata.json`, which is not redistributed here.
- **Thirty-split means (published paper)**: Brier 0.081 versus 0.090 for the clinical rule, expected calibration error 0.027 versus 0.084.

Nothing in this repository was produced, reproduced or independently validated by the maintainer. No number here is a claim about clinical performance.

## If you are an author

If you would prefer a different wording, additional attribution, or removal of material, open an issue or contact the maintainer directly. Changes are made on request without argument.

## Scope of use

This adapter is a demonstration of service engineering. It is not a medical device, not a validated clinical tool, and must not be used for diagnosis or treatment decisions.
