"""FastAPI HTTP entry. Models and stored user records remain unchanged."""
import json, logging, os, secrets, sqlite3, uuid
from pathlib import Path
from contextlib import closing, asynccontextmanager
from concurrent.futures import ThreadPoolExecutor
from fastapi import FastAPI, Request, HTTPException, Depends
from fastapi.security import HTTPBearer
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.sessions import SessionMiddleware
from starlette.middleware.cors import CORSMiddleware
from chronic.data import ROOT
from .schemas import UserInput, LoginInput


def create_app(storage=None):
    base=Path(storage or os.environ.get('CHRONIC_STORAGE',ROOT/'service_data')).resolve()
    base.mkdir(parents=True,exist_ok=True)
    pool=ThreadPoolExecutor(max_workers=2)
    @asynccontextmanager
    async def lifespan(app):
        from paper_models.runtime import get_models
        try:
            get_models()  # Load and validate supplied paper models once per process.
        except FileNotFoundError as exc:
            # The author weights are not redistributed with this repository.
            # The rest of the service works; only the adapter answers 503.
            logging.getLogger('uvicorn.error').warning('classifier adapter inactive: %s', exc)
        yield
        pool.shutdown(wait=True)
    app=FastAPI(title='慢病预测 API',version='3.1.0',lifespan=lifespan,
        docs_url=None, redoc_url=None, description='论文模型适配：提交检查表，直接返回结果。原有慢病分析：建立用户 → 上传记录 → 分析 → 查询结果。点击 Authorize 填入项目令牌，再用 Try it out 试调用。',
        dependencies=[Depends(HTTPBearer(auto_error=False))],
        swagger_ui_parameters={'displayRequestDuration':True,'defaultModelsExpandDepth':-1})
    secret=base/'session.key'
    if not secret.exists():secret.write_text(secrets.token_hex(32));secret.chmod(0o600)
    def sql(query,args=(),one=False):
        with closing(sqlite3.connect(base/'records.sqlite3',timeout=30)) as c:
            c.row_factory=sqlite3.Row
            rows=c.execute(query,args).fetchall();c.commit()
            return (dict(rows[0]) if rows else None) if one else [dict(r) for r in rows]
    sql('CREATE TABLE IF NOT EXISTS users(id TEXT PRIMARY KEY,data TEXT NOT NULL)')
    def require_user(uid):
        r=sql('SELECT * FROM users WHERE id=?',(uid,),True)
        if not r:raise HTTPException(404,'用户不存在')
        return dict(id=r['id'],**json.loads(r['data']))
    @app.middleware('http')
    async def auth(request:Request,call_next):
        token=os.environ.get('CHRONIC_API_TOKEN')
        if request.method!='OPTIONS':
            bearer=request.headers.get('Authorization','')
            if request.url.path.startswith('/api/') and token and not secrets.compare_digest(bearer,'Bearer '+token) and not request.session.get('authenticated'):
                return JSONResponse({'error':'请先登录或填写有效的项目令牌'},401)
            if request.session.get('authenticated') and request.method not in ['GET','HEAD'] and not bearer and request.headers.get('X-Requested-With')!='chronic-workspace':
                return JSONResponse({'error':'请求来源无效'},403)
        response=await call_next(request)
        response.headers['X-Content-Type-Options']='nosniff'
        if request.url.path.startswith('/api/'):response.headers['Cache-Control']='no-store'
        return response
    app.add_middleware(SessionMiddleware,secret_key=secret.read_text().strip(),same_site='strict',https_only=os.environ.get('CHRONIC_COOKIE_SECURE')=='1')
    app.add_middleware(CORSMiddleware,allow_origins=[s.strip() for s in os.environ.get('CHRONIC_ALLOWED_ORIGINS','').split(',') if s.strip()],allow_methods=['GET','POST','DELETE','OPTIONS'],allow_headers=['Authorization','Content-Type','X-Requested-With'])
    @app.exception_handler(HTTPException)
    async def http_error(request,exc):return JSONResponse({'error':exc.detail},exc.status_code)
    @app.exception_handler(RequestValidationError)
    async def validation_error(request,exc):
        errors=[{'field':'.'.join(map(str,e['loc'])),'message':e['msg']} for e in exc.errors()]
        return JSONResponse({'error':'请求参数不正确，请检查字段名称、类型和必填项','details':errors},422)
    @app.exception_handler(ValueError)
    async def value_error(request,exc):return JSONResponse({'error':str(exc)},400)
    @app.exception_handler(Exception)
    async def internal_error(request,exc):
        logging.getLogger(__name__).exception('Request failed',exc_info=exc)
        return JSONResponse({'error':'内部错误，请检查服务日志'},500)
    @app.post('/session',include_in_schema=False)
    def login(payload:LoginInput,request:Request):
        token=os.environ.get('CHRONIC_API_TOKEN')
        if token and not secrets.compare_digest(payload.password,token):raise HTTPException(401,'访问密码不正确')
        request.session.clear();request.session['authenticated']=True
        return {'status':'ok'}
    @app.delete('/session',include_in_schema=False)
    def logout(request:Request):request.session.clear();return {'status':'ok'}
    @app.get('/',include_in_schema=False)
    def page():
        path=ROOT/'validation/index.html'
        if not path.exists():raise HTTPException(404,'未安装验证网页，请打开/docs查看接口')
        return FileResponse(path)
    if (ROOT/'validation/assets').exists():app.mount('/assets',StaticFiles(directory=ROOT/'validation/assets'),name='validation-assets')
    @app.post('/api/v2/users',status_code=201,tags=['1. 用户档案'],summary='建立用户，返回用户编号')
    @app.post('/api/v1/users',status_code=201,include_in_schema=False)
    def users(payload:UserInput):
        d=payload.model_dump(exclude_none=True)
        if not d['name'].strip():raise ValueError('姓名不能为空')
        uid=uuid.uuid4().hex
        sql('INSERT INTO users VALUES(?,?)',(uid,json.dumps(d,ensure_ascii=False)))
        return require_user(uid)
    @app.get('/api/v2/users/{uid}',tags=['1. 用户档案'],summary='读取指定用户的基本信息')
    @app.get('/api/v1/users/{uid}',include_in_schema=False)
    def user(uid:str):return require_user(uid)
    from .business import register_business
    register_business(app,base,sql,pool,require_user)
    from .legacy import register_legacy
    register_legacy(app,base,sql,pool,require_user)
    from .papers import router as paper_router
    app.include_router(paper_router)
    from fastapi.openapi.docs import get_swagger_ui_html
    assets=Path(__file__).parent/'docs_assets'
    app.mount('/docs-assets',StaticFiles(directory=assets),name='docs-assets')
    @app.get('/docs',include_in_schema=False)
    def docs():
        return get_swagger_ui_html(openapi_url='/openapi.json',title='慢病 API · 在线试用',swagger_js_url='/docs-assets/swagger-ui-bundle.js',swagger_css_url='/docs-assets/swagger-ui.css',swagger_favicon_url='',swagger_ui_parameters={'displayRequestDuration':True,'defaultModelsExpandDepth':-1})
    generated_openapi=app.openapi
    contract_path=Path(__file__).parent/'openapi.json'
    response_contract=json.loads(contract_path.read_text(encoding='utf8')) if contract_path.exists() else {}
    def openapi_with_responses():
        schema=generated_openapi()
        for name,definition in response_contract.get('components',{}).get('schemas',{}).items():
            schema['components']['schemas'].setdefault(name,definition)
        for path,methods in response_contract.get('paths',{}).items():
            for method,definition in methods.items():
                target=schema.get('paths',{}).get(path,{}).get(method)
                if target:
                    for status,response in definition.get('responses',{}).items():
                        if status.startswith('2') and status in target['responses']:target['responses'][status]=response
        return schema
    app.openapi=openapi_with_responses
    return app

if __name__=='__main__':
    from .server import main
    main()
