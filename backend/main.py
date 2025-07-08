from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles


app = FastAPI()

templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="../frontend"), name="static")


@app.get("/")
async def read_root():
    return JSONResponse(
        content={"message": "Welcome to ML-Playground."}, status_code=200
    )

