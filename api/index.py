from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import cv2
from PIL import Image
import io

app = FastAPI(title="Automated Weed Detection API")

# Allow CORS for the frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/api/health")
def health_check():
    return {"status": "ok"}

@app.post("/api/predict")
async def predict(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        img_array = np.array(image)
        
        # Convert grayscale or RGBA
        if len(img_array.shape) == 2:
            img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
        elif img_array.shape[2] == 4:
            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2RGB)
            
        # Normalize
        img_normalized = img_array.astype(np.float32) / 255.0
        
        # Calculate features (matching old logic)
        brightness = float(np.mean(img_normalized))
        greenness = float(np.mean(img_normalized[:, :, 1]) - np.mean(img_normalized[:, :, 0]) / 2)
        
        if greenness > brightness * 0.3:
            prediction_idx = 1  # Weed
            confidence_weed = min(0.95, 0.6 + greenness)
            confidence_crop = 1.0 - confidence_weed
        else:
            prediction_idx = 0  # Crop
            confidence_crop = min(0.95, 0.7 - greenness)
            confidence_weed = 1.0 - confidence_crop
            
        classes = ["Crop", "Weed"]
        
        return {
            "prediction": classes[prediction_idx],
            "confidence_crop": confidence_crop,
            "confidence_weed": confidence_weed,
            "brightness": brightness,
            "greenness": greenness,
            "size": [image.size[0], image.size[1]]
        }
    except Exception as e:
        return {"error": str(e)}

# For local development: Mount the public directory to serve index.html
import os
from fastapi.staticfiles import StaticFiles

if os.path.exists("public"):
    app.mount("/", StaticFiles(directory="public", html=True), name="public")

# Export the app for Vercel
# Vercel will look for an `app` object in this file because we point to `api/index.py:app`
