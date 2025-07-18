import torch
import torch.nn as nn
import torch.optim as optim
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from models.simple_regression import SimpleLinearRegression, training_loop_slr
from models.mlp_regression import MLPRegression, training_loop_mlp
from models.simple_classifier import SimpleClassifier, training_loop_classifier
from models.mlp_classifier import MLPClassifier, training_loop_mlp_classifier

app = FastAPI()

templates = Jinja2Templates(directory="../frontend/templates")
app.mount("/static", StaticFiles(directory="../frontend/static"), name="static")

models = {
    "simple-linear-regression": SimpleLinearRegression(),
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

class TrainRequestMLPClassifier(BaseModel):
    data: list[DataClassifier]
    learning_rate: float
    epochs: int
    hidden_size: int

class PredictRequestMLP(BaseModel):
    inputs: list[float]

class PredictRequestClassifier(BaseModel):
    inputs: list[list[float]]

@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    return templates.TemplateResponse(
        "index.html", {"request": request}
    )

@app.get("/templates/{template_name}", response_class=HTMLResponse)
async def get_template(request: Request, template_name: str):
    return templates.TemplateResponse(f"{template_name}", {"request": request})

@app.post("/train_simple_linear_regression")
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
        print(str(e))
        return JSONResponse(
            content={"error": + str(e)},
            status_code=500
        )


@app.post("/train_mlp_regression")
async def train_mlp_regression(request: TrainRequestMLP):
    try:
        x_data = torch.tensor([[i.x] for i in request.data], dtype=torch.float32)
        y_data = torch.tensor([[i.y] for i in request.data], dtype=torch.float32)

        model = MLPRegression(input_f=1, hidden_size=request.hidden_size, output_f=1)
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
            content={"error": + str(e)},
            status_code=500
        )

@app.post("/train_simple_classifier")
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
            content={"error": + str(e)},
            status_code=500
        )

@app.post("/train_mlp_classifier")
async def train_mlp_classifier(request: TrainRequestMLPClassifier):
    try:
        x_data = torch.tensor([[i.x, i.y] for i in request.data], dtype=torch.float32)
        y_data = torch.tensor([[i.label] for i in request.data], dtype=torch.float32)

        model = MLPClassifier(input_f=2, hidden_size=request.hidden_size, output_f=1)
        optimizer = optim.Adam(model.parameters(), lr=request.learning_rate)
        criterion = nn.BCELoss()

        final_loss = training_loop_mlp_classifier(
            x_data, y_data,
            model, optimizer, criterion,
            request.epochs
        )

        models["mlp-classifier"] = model

        return JSONResponse(
            content={
            "message": "MLP Classifier trained successfully",
            "final_loss": final_loss},
            status_code=200
        )
    except Exception as e:
        return JSONResponse(
            content={"error": str(e)},
            status_code=500
        )

@app.post("/predict_mlp_regression")
async def predict_mlp_regression_endpoint(request_body: PredictRequestMLP):
    try:
        model = models["mlp-regression"]
        if not isinstance(model, MLPRegression):
            raise ValueError("MLP model not yet trained or not loaded correctly.")

        model.eval()
        with torch.no_grad():
            x_inputs = torch.tensor([[x_val] for x_val in request_body.inputs], dtype=torch.float32)
            predictions = model(x_inputs).squeeze().tolist()

        return JSONResponse(
            content=
            {"predictions": predictions},
            status_code=200
        )
    except Exception as e:
        return JSONResponse(
            content={"error": + str(e)},
            status_code=500
        )

@app.post("/predict_mlp_classifier")
async def predict_mlp_classifier_endpoint(request_body: PredictRequestClassifier):
    try:
        model = models["mlp-classifier"]
        if not isinstance(model, MLPClassifier):
            raise ValueError("MLP Classifier model not yet trained or not loaded correctly.")

        model.eval()
        with torch.no_grad():
            x_inputs = torch.tensor(request_body.inputs, dtype=torch.float32)
            predictions = model(x_inputs).squeeze().tolist()
            # Convert probabilities to classes
            class_predictions = [1 if p > 0.5 else 0 for p in predictions]

        return JSONResponse(
            content={"predictions": class_predictions},
            status_code=200
        )
    except Exception as e:
        return JSONResponse(
            content={"error": str(e)},
            status_code=500
        )