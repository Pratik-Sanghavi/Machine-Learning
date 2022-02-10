from fastapi import (FastAPI,
                     Depends,
                     Request,
                     File,
                     UploadFile,
                     HTTPException)
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.templating import Jinja2Templates
import pathlib
import os
import io
import uuid
from PIL import Image
import tensorflow as tf
import asyncio
import numpy as np
import cv2

app = FastAPI()
BASE_DIR = pathlib.Path(__file__).parent
UPLOAD_DIR = BASE_DIR / 'uploads'
TEMPLATES = Jinja2Templates(directory=str(BASE_DIR / 'templates'))
MODEL = BASE_DIR / "Model"

LABEL_MAPPING = ['airplane', 'automobile','bird','cat', 'deer', 'dog','frog', 'horse', 'ship', 'truck']

@app.get('/', response_class=HTMLResponse)
@app.get('/home', response_class=HTMLResponse)
def home_view(request:Request):
    return TEMPLATES.TemplateResponse('home.html', {'request':request})

@app.get('/cifar_10', response_class=HTMLResponse)
def cifar_10(request:Request):
    return TEMPLATES.TemplateResponse('cifar_10.html', {'request':request})

async def load_model():
    model = tf.keras.models.load_model(MODEL / 'my_model.h5')
    return model

def preprocess_image(image_array):
    preprocessed_image = cv2.resize(image_array, (32,32))
    preprocessed_image = np.expand_dims(np.asarray(preprocessed_image), axis=0)
    return preprocessed_image

@app.post("/cifar_10_predict") #http POST
async def cifar_10_predict(file:UploadFile = File(...)):
    task = asyncio.create_task(load_model())
    UPLOAD_DIR.mkdir(exist_ok=True)
    bytes_str = io.BytesIO(await file.read())
    try:
        img = Image.open(bytes_str)
    except:
        raise HTTPException(detail="Invalid image", status_code=400)
    fname = pathlib.Path(file.filename)
    fext = fname.suffix #.jpg, .png etc
    dest = UPLOAD_DIR / f"{uuid.uuid1()}{fext}"
    img.save(dest)
    try:
        image_array = np.asarray(img)
        assert len(image_array.shape) == 3
    except:
        raise HTTPException(detail="Invalid shape", status_code=400)

    preprocessed_image = preprocess_image(image_array)
    model = await task
    res = model.predict(preprocessed_image)
    res = np.argmax(res, axis=-1)
    return {"predicted_label":LABEL_MAPPING[res[0]]}