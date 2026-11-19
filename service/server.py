"""Linux production entry: one owner for the durable SQLite job queue."""
import os
from pathlib import Path
import uvicorn
from chronic.data import ROOT
from .app import create_app


def main():
    from .settings import load_local_env
    load_local_env(ROOT)
    os.umask(0o077)
    for model in ['artifacts/v2/survey.joblib', 'artifacts/v2/glucose.joblib',
                  'artifacts/table_heart_v1/table_model.joblib',
                  'artifacts/papers/hypertension/model.joblib']:
        if not (ROOT / model).is_file():
            raise SystemExit(f'Missing required model: {model}')
    base = Path(os.environ.get('CHRONIC_STORAGE', ROOT / 'service_data')).resolve()
    base.mkdir(parents=True, exist_ok=True)
    host = os.environ.get('CHRONIC_HOST', '127.0.0.1')
    if host not in ('127.0.0.1', 'localhost', '::1') and not os.environ.get('CHRONIC_API_TOKEN'):
        raise SystemExit('Set CHRONIC_API_TOKEN before binding a public interface.')
    # The lock is released by the OS on crash. Multiple HTTP threads share one task pool.
    lock = (base / 'server.lock').open('a+')
    if os.name == 'posix':
        import fcntl
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit('Another service already owns this storage directory.')
    app = create_app(base)
    uvicorn.run(app, host=host, port=int(os.environ.get('CHRONIC_PORT', '8765')))


if __name__ == '__main__':
    main()
