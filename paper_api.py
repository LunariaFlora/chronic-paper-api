"""FastAPI entry for the two paper algorithms."""
from contextlib import asynccontextmanager

from fastapi import FastAPI

from paper_models.runtime import get_models
from service.papers import router


@asynccontextmanager
async def lifespan(app):
    get_models()
    yield


app = FastAPI(title="论文模型接口", version="3.1.0", lifespan=lifespan)
app.include_router(router)
