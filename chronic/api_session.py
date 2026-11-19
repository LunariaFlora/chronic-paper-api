"""Load only documented API settings from the user's local .env file."""
import os
from pathlib import Path


def load_api_env(folder):
    path=Path(folder)/'.env'
    if not path.is_file():return
    for line in path.read_text(encoding='utf-8-sig').splitlines():
        line=line.strip()
        if not line or line.startswith('#') or '=' not in line:continue
        key,value=line.split('=',1);key=key.strip();value=value.strip()
        if key not in ('KIMI_API_KEY','KIMI_MODEL'):continue
        if len(value)>=2 and value[0]==value[-1] and value[0] in ('"',"'"):value=value[1:-1]
        if value:os.environ[key]=value


def request_recommendations():
    if input('预测后生成健康建议？[y/N]：').strip().lower()!='y':return False
    if os.environ.get('KIMI_API_KEY'):return True
    print('未配置API；按README填写.env后可生成建议。本次继续预测。')
    return False
