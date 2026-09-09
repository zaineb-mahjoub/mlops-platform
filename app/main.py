from fastapi import FastAPI
from app.api import health, predict
from fastapi.staticfiles import StaticFiles

app = FastAPI(title="MLOps Platform API", version="0.1.0")
app.mount("/ui", StaticFiles(directory="static", html=True), name="static")
app.include_router(health.router)
app.include_router(predict.router)




