"""
OncaScan Platform — Microservicio del Modelo IA
El backend llama este servicio via HTTP para obtener predicciones.
"""

import io
import torch
import torch.nn as nn
import torchvision.models as models
import numpy as np
from PIL import Image
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

# ── CONFIGURACIÓN ─────────────────────────────────────
MODELO_PATH = 'best_model_multimodal.pth'
IMSIZE      = 64
device      = torch.device('cpu')

FEAT_MEAN = np.array([3.74, 5.67, 3.80, 3.90, 1.60, 1.53, 4.35, 2.74], dtype=np.float32)
FEAT_STD  = np.array([1.18, 0.93, 0.94, 1.24, 0.97, 0.95, 1.23, 1.08], dtype=np.float32)

# ── MODELO MULTIMODAL ─────────────────────────────────
class MultimodalNet(nn.Module):
    def __init__(self, n_features=8):
        super().__init__()
        backbone = models.resnet18(weights=None)
        for name, param in backbone.named_parameters():
            if 'layer4' not in name and 'fc' not in name:
                param.requires_grad = False
        backbone.fc = nn.Identity()
        self.imagen_encoder   = backbone
        self.clinical_encoder = nn.Sequential(
            nn.Linear(n_features, 64), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64, 32),         nn.ReLU(),
        )
        self.fusion = nn.Sequential(
            nn.Linear(512 + 32, 128), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(128, 32),       nn.ReLU(),
            nn.Linear(32, 1),         nn.Sigmoid()
        )

    def forward(self, img, feat):
        return self.fusion(torch.cat([
            self.imagen_encoder(img),
            self.clinical_encoder(feat)
        ], dim=1))


# ── CARGAR MODELO AL INICIAR ──────────────────────────
print("Cargando modelo OncaScan...")
model = MultimodalNet(n_features=8)
model.load_state_dict(torch.load(MODELO_PATH, map_location=device))
model.eval()
print("✓ Modelo listo")

# ── FASTAPI ───────────────────────────────────────────
app = FastAPI(
    title="OncaScan AI Service",
    description="Microservicio de deteccion de nodulos pulmonares — OncaScan Platform",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── SCHEMAS ───────────────────────────────────────────
class FeaturesClinicas(BaseModel):
    subtlety:      float
    calcification: float
    sphericity:    float
    margin:        float
    lobulation:    float
    spiculation:   float
    texture:       float
    malignancy:    float

class PredictionResponse(BaseModel):
    score:          float
    nivel_riesgo:   str
    recomendacion:  str
    modelo_version: str

# ── HELPERS ───────────────────────────────────────────
def nivel_riesgo(score):
    if score < 0.33:   return "BAJO",  "Control rutinario"
    elif score < 0.66: return "MEDIO", "Seguimiento recomendado"
    else:              return "ALTO",  "Evaluacion urgente"

def preprocesar_imagen(image_bytes):
    img = Image.open(io.BytesIO(image_bytes)).convert('L').resize((IMSIZE, IMSIZE))
    img = np.array(img, dtype=np.float32) / 255.0
    return torch.tensor(np.stack([img, img, img], axis=0)).unsqueeze(0)

def preprocesar_features(features: FeaturesClinicas):
    feat = np.array([
        features.subtlety,      features.calcification,
        features.sphericity,    features.margin,
        features.lobulation,    features.spiculation,
        features.texture,       features.malignancy,
    ], dtype=np.float32)
    feat = (feat - FEAT_MEAN) / FEAT_STD
    return torch.tensor(feat).unsqueeze(0)

# ── ENDPOINTS ─────────────────────────────────────────
@app.get("/")
def root():
    return {
        "status":   "OncaScan AI Service corriendo",
        "version":  "1.0.0",
        "docs":     "http://localhost:8000/docs"
    }

@app.get("/health")
def health():
    return {
        "status":       "ok",
        "modelo":       "MultimodalNet ResNet18",
        "device":       str(device),
        "modelo_listo": True,
        "version":      "multimodal-v1.0"
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict(
    imagen:        UploadFile = File(...,  description="Imagen CT en PNG o JPG"),
    subtlety:      float      = Form(..., description="Sutileza del nodulo [1-5]"),
    calcification: float      = Form(..., description="Calcificacion [1-6]"),
    sphericity:    float      = Form(..., description="Esfericidad [1-5]"),
    margin:        float      = Form(..., description="Margen [1-5]"),
    lobulation:    float      = Form(..., description="Lobulacion [1-5]"),
    spiculation:   float      = Form(..., description="Espiculacion [1-5]"),
    texture:       float      = Form(..., description="Textura [1-5]"),
    malignancy:    float      = Form(..., description="Malignidad visual [1-5]"),
):
    # Validar imagen
    if not imagen.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        raise HTTPException(
            status_code=400,
            detail="Solo se aceptan imagenes PNG o JPG"
        )

    # Validar rangos de features
    validaciones = [
        (subtlety,      1, 5,  "subtlety"),
        (calcification, 1, 6,  "calcification"),
        (sphericity,    1, 5,  "sphericity"),
        (margin,        1, 5,  "margin"),
        (lobulation,    1, 5,  "lobulation"),
        (spiculation,   1, 5,  "spiculation"),
        (texture,       1, 5,  "texture"),
        (malignancy,    1, 5,  "malignancy"),
    ]
    for val, vmin, vmax, nombre in validaciones:
        if not (vmin <= val <= vmax):
            raise HTTPException(
                status_code=422,
                detail=f"{nombre} debe estar entre {vmin} y {vmax}. Recibido: {val}"
            )

    # Preprocesar imagen
    image_bytes = await imagen.read()
    img_tensor  = preprocesar_imagen(image_bytes)

    # Preprocesar features
    features = FeaturesClinicas(
        subtlety=subtlety,           calcification=calcification,
        sphericity=sphericity,       margin=margin,
        lobulation=lobulation,       spiculation=spiculation,
        texture=texture,             malignancy=malignancy
    )
    feat_tensor = preprocesar_features(features)

    # Inferencia
    with torch.no_grad():
        score = model(img_tensor, feat_tensor).item()

    nivel, recomendacion = nivel_riesgo(score)

    return PredictionResponse(
        score          = round(score, 4),
        nivel_riesgo   = nivel,
        recomendacion  = recomendacion,
        modelo_version = "multimodal-v1.0"
    )

# ── MAIN ──────────────────────────────────────────────
if __name__ == '__main__':
    print("\nOncaScan AI Service")
    print("Documentacion: http://localhost:8000/docs")
    print("Health check:  http://localhost:8000/health\n")
    uvicorn.run(app, host="0.0.0.0", port=8000)