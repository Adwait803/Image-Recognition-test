import asyncio
import time
import uuid
from concurrent.futures import ProcessPoolExecutor
from functools import partial
import io

import uvicorn
from fastapi import File
from fastapi import FastAPI
from fastapi import UploadFile
import numpy as np
from PIL import Image

import config
from clf import predict


app = FastAPI()

def read_imagefile(file) -> Image.Image:
    image = Image.open(io.BytesIO(file))
    return image

@app.get("/")
def read_root():
    return {"message": "Welcome from the API"}





@app.post("/a")
async def predict_api(file: UploadFile = File(...)):
    extension = file.filename.split(".")[-1] in ("jpg", "jpeg", "png")
    if not extension:
        return "Image must be jpg or png format!"
    image = read_imagefile(await file.read())
    prediction = predict(image)
    return prediction



if __name__ == "__main__":
     uvicorn.run("main:app", host="localhost", port=8000)