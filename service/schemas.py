from typing import Literal
from pydantic import BaseModel, ConfigDict, Field

class Input(BaseModel):
    model_config = ConfigDict(extra='forbid', allow_inf_nan=False)

class UserInput(Input):
    name: str = Field(min_length=1, max_length=100, description='姓名', examples=['张三'])
    age: float | None = Field(None, ge=0, le=130, description='实际年龄（岁）')
    sex: Literal['male','female','unknown'] = 'unknown'
    height_cm: float | None = Field(None, ge=20, le=260, description='身高，厘米')
    weight_kg: float | None = Field(None, ge=1, le=500, description='体重，千克')

class AnalysisInput(Input):
    import_id: str = Field(description='上传文件或JSON记录后返回的id')
    row: int | None = Field(None, ge=0, description='多行独立检查时选择当前用户对应行，从0开始')
    unit: Literal['mmol/L','mg/dL'] | None = Field(None, description='原始血糖未写明单位时补充')
    advice_engine: Literal['rules','kimi'] | None = Field(None, description='kimi使用Kimi生成建议；rules使用本地规则。有密钥时默认kimi')

class Measurement(Input):
    timestamp: str = Field(description='含时区的测量时间', examples=['2026-11-02T08:00:00+08:00'])
    value: float = Field(description='该时刻的血糖值')

class RecordInput(Input):
    user_id: str
    unit: Literal['mmol/L','mg/dL']
    measurements: list[Measurement] = Field(min_length=1, max_length=100000)

class LoginInput(Input):
    password: str
