"""Explicit column selection on top of code-only schema discovery."""
import copy


def column_number(value,columns):
    value=value.strip()
    if not value.isdigit() or not 1<=int(value)<=len(columns):raise ValueError('请输入表格左侧的列序号，例如17。')
    return columns[int(value)-1]


def select_features(schema, names):
    if names is None:
        return schema
    names=list(names)
    allowed=schema['numeric']+schema.get('categorical',[])
    if not names or len(set(names))!=len(names):
        raise ValueError('输入列不能为空或重复')
    bad=set(names)-set(allowed)
    if bad:
        raise ValueError(f'这些列不能作为自动流程的输入特征（可能是标签、ID或时间列）：{sorted(bad)}')
    result=copy.deepcopy(schema)
    for key in ['numeric','categorical','allow_missing','categorical_numeric']:
        if key in result:result[key]=[c for c in result[key] if c in names]
    for key in ['ranges','enums']:
        if key in result:result[key]={c:v for c,v in result[key].items() if c in names}
    result['selected_features']=names
    return result


def column_name(value,columns):
    """Accept an exact name or a one-based displayed index, never an Excel data row."""
    value=value.strip()
    if value in columns:return value
    if value.isdigit() and 1<=int(value)<=len(columns):return columns[int(value)-1]
    raise ValueError(f'没有这个列名或列序号：{value}')
