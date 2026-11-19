"""Package only the published-classifier adapter and its inference dependencies."""
import argparse
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=ROOT/'outputs/chronic_service_linux')
    dest = parser.parse_args().output.resolve()
    if dest.exists() and any(dest.iterdir()):
        raise SystemExit('Output must be empty to avoid shipping stale files.')
    files = {
        'paper_api.py': 'paper_api.py',
        'requirements.txt': 'requirements-paper-api.txt',
        'README.md': 'docs/paper_api_README.md',
        'ATTRIBUTION.md': 'ATTRIBUTION.md',
    }
    paths = [
        'service/__init__.py', 'service/papers.py',
        'paper_models/__init__.py', 'paper_models/runtime.py', 'paper_models/SOURCES.md',
        'paper_models/hypertension_source/__init__.py',
        'paper_models/hypertension_source/calibration.py',
        'paper_models/hypertension_source/config.py',
        'paper_models/hypertension_source/pipeline_builder.py',
        'artifacts/papers/hypertension/model.joblib',
        'artifacts/papers/hypertension/metadata.json',
    ]
    files.update({p:p for p in paths})
    for target, source in files.items():
        output = dest/target
        output.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT/source, output)
    total = sum((dest/p).stat().st_size for p in files)
    print(f'{len(files)} files; {total/1024/1024:.2f} MiB; {dest}')

if __name__ == '__main__':
    main()
