from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import streamlit as st

from models.simple_regression import SimpleLinearRegression, training_loop_slr
from models.simple_classifier import SimpleClassifier, training_loop_classifier
from models.mlp_regression import MLPRegression, training_loop_mlp

app = FastAPI()

templates = Jinja2Templates(directory="../frontend/templates")
app.mount("/static", StaticFiles(directory="../frontend/static"), name="static")

models = {
    "linear-regression": SimpleLinearRegression(),
    "simple-classifier": SimpleClassifier(),
    "mlp-regression": MLPRegression(),
}

@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    return templates.TemplateResponse(
        "index.html", {"request": request}
    )

app.post("/train_simple_linear_regression")

async def train_simple_classifier():
    pass

app.post("/train_simple_classifier")
async def train_simple_classifier():
    pass

app.post("/train_mlp_regression")
async def train_mlp_regression():
    pass