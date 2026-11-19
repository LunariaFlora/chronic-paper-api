# 调用示例：先启动API，再设置CHRONIC_API_TOKEN。依赖：pip install requests
import os, time, requests

BASE = "http://localhost:8765"          # 替换为实际部署地址
TOKEN = os.environ["CHRONIC_API_TOKEN"]
HEADERS = {"Authorization": f"Bearer {TOKEN}"}

# 1. 建立用户
r = requests.post(
    f"{BASE}/api/v2/users",
    headers=HEADERS,
    json={"name": "张三", "age": 58, "sex": "male", "height_cm": 172, "weight_kg": 70}
)
r.raise_for_status()
user = r.json()
user_id = user["id"]

# 2. 上传健康记录（以心脏检查模板为例）
with open("service/templates/heart.csv", "rb") as f:
    r = requests.post(
        f"{BASE}/api/v2/imports",
        headers=HEADERS,
        files={"file": f},
        data={"user_id": user_id}
    )
r.raise_for_status()
imp = r.json()
import_id = imp["id"]
# 若 needs_unit=true，在下一步 JSON 中加 unit="mmol/L" 或 "mg/dL"。
# 若 needs_row=true，在下一步加 row=0 等，选择当前用户的一行（从0开始）。

# 3. 发起分析
r = requests.post(
    f"{BASE}/api/v2/analyses",
    headers=HEADERS,
    json={"import_id": import_id}
)
r.raise_for_status()
task = r.json()
analysis_id = task["id"]

# 4. 轮询结果
while True:
    r = requests.get(
        f"{BASE}/api/v2/analyses/{analysis_id}",
        headers=HEADERS
    )
    r.raise_for_status()
    res = r.json()
    if res["status"] not in ("queued", "running"):
        break
    time.sleep(2)

# 返回样例（数值仅为示意，以实际返回为准）
report = res.get("report") or {}
print("状态:", res["status"])                       # completed / needs_data / failed
print("标题:", report.get("headline"))
print("指标:", report.get("metrics"))               # 如最近一次血糖
print("预测:", report.get("forecasts"))             # 30/60分钟血糖示意
print("建议:", report.get("recommendations"))       # 健康建议；查看 recommendation_engine 确认来源

r = requests.get(f"{BASE}/api/v2/users/{user_id}/integration", headers=HEADERS)
print(r.json())
