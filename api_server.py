#!/usr/bin/env python3
"""
FastAPI server for land cover classification inference.
Production-ready deployment with health checks and batch processing.
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import Response, JSONResponse
import torch
import numpy as np
from PIL import Image
import io
from pathlib import Path
import uvicorn

from models.unet import get_model
from inference import predict_single_image, colorize_mask, load_sentinel2_image


# Initialize FastAPI app
app = FastAPI(
    title="EO-AI Land Cover Classification API",
    description="Sentinel-2 land cover classification using quantized U-Net",
    version="1.0.0"
)

# Global model variable
model = None
device = None

CLASS_NAMES = [
    "Urban fabric",
    "Industrial or commercial units",
    "Arable land",
    "Permanent crops",
    "Pastures",
    "Complex cultivation patterns",
    "Forests",
    "Herbaceous vegetation",
    "Open spaces with little or no vegetation",
    "Wetlands and water bodies"
]


def load_model_checkpoint(checkpoint_path: str, device_name: str = 'cpu'):
    """Load model checkpoint on startup"""
    global model, device
    
    device = torch.device(device_name)
    model = get_model(n_channels=4, n_classes=10)
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model = model.to(device)
    model.eval()
    
    print(f"✅ Model loaded on {device}")


@app.on_event("startup")
async def startup_event():
    """Load model on server startup"""
    checkpoint_path = Path("checkpoints/quantized_model.pth")
    
    if not checkpoint_path.exists():
        checkpoint_path = Path("checkpoints/best_model.pth")
    
    if checkpoint_path.exists():
        load_model_checkpoint(str(checkpoint_path))
    else:
        print("⚠️ No model checkpoint found. Upload one to /checkpoints/")


@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "name": "EO-AI Land Cover Classification API",
        "version": "1.0.0",
        "status": "running",
        "model_loaded": model is not None,
        "endpoints": {
            "/predict": "POST - Upload Sentinel-2 image for classification",
            "/predict/json": "POST - Get prediction as JSON",
            "/health": "GET - Health check",
            "/info": "GET - Model information"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint for container orchestration"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return {
        "status": "healthy",
        "model_loaded": True,
        "device": str(device)
    }


@app.get("/info")
async def model_info():
    """Get model information"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    num_params = sum(p.numel() for p in model.parameters())
    
    return {
        "model": "U-Net",
        "input_bands": 4,
        "num_classes": 10,
        "parameters": num_params,
        "device": str(device),
        "classes": CLASS_NAMES
    }


@app.post("/predict")
async def predict_image(file: UploadFile = File(...)):
    """
    Predict land cover classification from Sentinel-2 image.
    Returns colored prediction as PNG.
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Read uploaded file
        contents = await file.read()
        
        # Handle different file formats
        if file.filename.endswith('.npy'):
            # Load NPY file
            image = np.load(io.BytesIO(contents))
            if image.ndim == 3 and image.shape[0] == 4:
                image = image.transpose(1, 2, 0)  # (4, H, W) -> (H, W, 4)
        else:
            raise HTTPException(
                status_code=400,
                detail="Unsupported file format. Please upload .npy file"
            )
        
        # Run prediction
        prediction, inference_time = predict_single_image(model, image, device)
        
        # Colorize prediction
        colored_pred = colorize_mask(prediction)
        
        # Convert to PNG
        pil_image = Image.fromarray(colored_pred)
        img_byte_arr = io.BytesIO()
        pil_image.save(img_byte_arr, format='PNG')
        img_byte_arr.seek(0)
        
        return Response(
            content=img_byte_arr.getvalue(),
            media_type="image/png",
            headers={
                "X-Inference-Time": f"{inference_time:.2f}ms"
            }
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/json")
async def predict_json(file: UploadFile = File(...)):
    """
    Predict land cover classification and return JSON with class distributions.
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Read uploaded file
        contents = await file.read()
        
        # Handle different file formats
        if file.filename.endswith('.npy'):
            image = np.load(io.BytesIO(contents))
            if image.ndim == 3 and image.shape[0] == 4:
                image = image.transpose(1, 2, 0)
        else:
            raise HTTPException(
                status_code=400,
                detail="Unsupported file format. Please upload .npy file"
            )
        
        # Run prediction
        prediction, inference_time = predict_single_image(model, image, device)
        
        # Calculate class distribution
        unique, counts = np.unique(prediction, return_counts=True)
        total_pixels = prediction.size
        
        class_distribution = {}
        for class_id, count in zip(unique, counts):
            class_distribution[CLASS_NAMES[class_id]] = {
                "pixels": int(count),
                "percentage": float(count / total_pixels * 100)
            }
        
        return JSONResponse(content={
            "prediction_shape": prediction.shape,
            "inference_time_ms": inference_time,
            "class_distribution": class_distribution,
            "total_pixels": int(total_pixels)
        })
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Run inference API server')
    parser.add_argument('--host', type=str, default='0.0.0.0',
                        help='Host to bind to')
    parser.add_argument('--port', type=int, default=8000,
                        help='Port to bind to')
    parser.add_argument('--reload', action='store_true',
                        help='Enable auto-reload for development')
    
    args = parser.parse_args()
    
    uvicorn.run(
        "api_server:app",
        host=args.host,
        port=args.port,
        reload=args.reload
    )
