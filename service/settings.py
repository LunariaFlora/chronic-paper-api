"""Load optional local .env values without overriding the caller's environment."""
import os
from pathlib import Path


def load_local_env(root: Path):
    path = root/'.env'
    if not path.is_file():
        return
    allowed = {'CHRONIC_HOST','CHRONIC_PORT','CHRONIC_STORAGE','CHRONIC_API_TOKEN',
               'CHRONIC_ALLOWED_ORIGINS','CHRONIC_COOKIE_SECURE','CHRONIC_TIMEZONE',
               'KIMI_API_KEY','KIMI_MODEL','PAPER_TORCH_THREADS'}
    for line in path.read_text(encoding='utf-8-sig').splitlines():
        line = line.strip()
        if not line or line.startswith('#') or '=' not in line:
            continue
        key, value = line.split('=',1)
        key, value = key.strip(), value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
            value = value[1:-1]
        if key in allowed:
            os.environ.setdefault(key, value)
