import torch
import torch.nn as nn
import torch.optim as optim
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

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

class DataLR(BaseModel):
    x: float
    y: float

class DataClassifier(BaseModel):
    x: float
    y: float
    label: int

class TrainRequestLR(BaseModel):
    data: list[DataLR]
    learning_rate: float
    epochs: int

class TrainRequestMLP(BaseModel):
    data: list[DataLR]
    learning_rate: float
    epochs: int
    hidden_size: int

class TrainRequestClassifier(BaseModel):
    data: list[DataClassifier]
    learning_rate: float
    epochs: int


@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    return templates.TemplateResponse(
        "index.html", {"request": request}
    )

app.post("/train_simple_linear_regression")
async def train_simple_linear_regression(request: TrainRequestLR):
    try:
        x_data = torch.tensor([[i.x] for i in request.data], dtype=torch.float32)
        y_data = torch.tensor([[i.y] for i in request.data], dtype=torch.float32)

        model = SimpleLinearRegression()
        optimizer = optim.SGD(model.parameters(), lr=request.learning_rate)
        criterion = nn.MSELoss()

        final_loss = training_loop_slr(
            x_data, y_data,
            model, optimizer, criterion,
            request.epochs
        )

        models["linear-regression"] = model

        weights = model.fc.weight.item()
        bias = model.fc.bias.item()

        return JSONResponse(
            content={
            "message": "Model trained successfully",
            "final_loss": final_loss, "weights": [weights], "bias": bias},
            status_code=200
        )
    except Exception as e:
        return JSONResponse(
            content="ERROR: " + str(e),
            status_code=500
        )


app.post("/train_mlp_regression")
async def train_mlp_regression(request: TrainRequestMLP):
    try:
        x_data = torch.tensor([[i.x] for i in request.data], dtype=torch.float32)
        y_data = torch.tensor([[i.y] for i in request.data], dtype=torch.float32)

        model = MLPRegression(input_size=1, hidden_size=request.hidden_size, output_size=1)
        optimizer = optim.Adam(model.parameters(), lr=request.learning_rate)
        criterion = nn.MSELoss()

        final_loss = training_loop_mlp(
            x_data, y_data,
            model, optimizer, criterion,
            request.epochs
        )
        
        models["mlp-regression"] = model

        return JSONResponse(
            content={
            "message": "MLP trained successfully",
            "final_loss": final_loss},
            status_code=200
        )
    except Exception as e:
        return JSONResponse(
            content="ERROR: " + str(e),
            status_code=500
        )

app.post("/train_simple_classifier")
async def train_simple_classifier(request: TrainRequestClassifier):
    try:
        x_data = torch.tensor([[i.x, i.y] for i in request.data], dtype=torch.float32)
        y_data = torch.tensor([[i.label] for i in request.data], dtype=torch.float32)

        model = SimpleClassifier(input_f=2, output_f=1)
        optimizer = optim.SGD(model.parameters(), lr=request.learning_rate)
        criterion = nn.BCELoss()

        final_loss = training_loop_classifier(
            x_data, y_data,
            model, optimizer, criterion,
            request.epochs
        )
        
        models["simple-classifier"] = model

        weights = model.linear.weight.tolist()[0]
        bias = model.linear.bias.item()

        return JSONResponse(
            content=
            {"message": "Model trained successfully",
            "final_loss": final_loss, "weights": weights, "bias": bias},
            status_code=200
        )
            
    except Exception as e:
        return JSONResponse(
            content="ERROR: " + str(e),
            status_code=500
        )