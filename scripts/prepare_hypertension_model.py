"""Rebuild the author's seed-42 decision tree without another parameter search."""
import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import joblib
import pandas as pd
import sklearn
from sklearn.metrics import accuracy_score, recall_score, brier_score_loss
from sklearn.tree import DecisionTreeClassifier
from paper_models.hypertension_source.config import FEATURE_COLS
from paper_models.hypertension_source.pipeline_builder import build_pipeline, get_rfe_selected_features
from paper_models.hypertension_source.calibration import fit_calibrator
from paper_models.hypertension_source.splits import make_split


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=ROOT/'datasets/papers/hypertension.csv')
    parser.add_argument('--output', type=Path, default=ROOT/'artifacts/papers/hypertension')
    args = parser.parse_args()
    df = pd.read_csv(args.data, na_values=['NA', 'na', ''])
    split = make_split(df[FEATURE_COLS], df['Risk'].astype(int), seed=42)
    # Exact parameters in revision/outputs/predictions/model_params_seed42.csv.
    params = dict(max_depth=2, min_samples_split=20, min_samples_leaf=9, criterion='gini')
    pipeline = build_pipeline(DecisionTreeClassifier(random_state=42, **params), n_features_to_select=6)
    pipeline.fit(split.X_train, split.y_train)
    calibrated = fit_calibrator(pipeline, split.X_cal, split.y_cal)
    probabilities = calibrated.predict_proba(split.X_test)[:, 1]
    predictions = calibrated.predict(split.X_test)
    metadata = {
        'model_id': 'hypertension-dt-seed42-v1',
        'source_archive': 'hypertension-code-data.zip',
        'seed': 42, 'parameters': params, 'selected_k': 6,
        'selected_features': get_rfe_selected_features(pipeline, FEATURE_COLS),
        'input_features': FEATURE_COLS, 'target': 'Risk', 'threshold': 0.5,
        'sklearn_version': sklearn.__version__,
        'data_sha256': hashlib.sha256(args.data.read_bytes()).hexdigest(),
        'split_counts': {name: len(getattr(split, 'y_'+name)) for name in ['train','val','cal','test']},
        'test': {'accuracy': accuracy_score(split.y_test, predictions),
                 'recall': recall_score(split.y_test, predictions),
                 'brier': brier_score_loss(split.y_test, probabilities)},
        'scope': 'Post-measurement classification of dataset Risk; no future incidence horizon.',
    }
    args.output.mkdir(parents=True, exist_ok=True)
    joblib.dump(calibrated, args.output/'model.joblib')
    (args.output/'metadata.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding='utf-8')
    evaluation = split.X_test.copy()
    evaluation['truth'] = split.y_test
    evaluation['prediction'] = predictions
    evaluation['probability'] = probabilities
    # Research evaluation stays outside the share package; no raw records in artifacts.
    report_dir = ROOT/'work/paper_api_validation'
    report_dir.mkdir(parents=True, exist_ok=True)
    evaluation.to_csv(report_dir/'hypertension_holdout.csv', index=False)
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
