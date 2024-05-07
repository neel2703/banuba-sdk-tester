import json
from fastapi import FastAPI, Path, Query
from fastapi.responses import JSONResponse
from mangum import Mangum
import ast
import uvicorn
import requests
from banuba_face_analysis import detect_face, detect_face_landmark

app = FastAPI()
handler = Mangum(app)

@app.get('/status')
async def health_check():
    return 'Success'

@app.get('/face_detection')
async def face_detection(data):
    b_box = detect_face(data)
    return {'coords': b_box}

@app.get('/landmark_detection')
async def landmark_detection(data):
    lmarks = detect_face_landmark(data)
    return {'lmarks': lmarks}

# if __name__ == "__main__":
#     uvicorn.run(app, host="127.0.0.1", port=80)