# OncaScan AI Service

Microservicio de detección de nódulos pulmonares.
Modelo multimodal: imagen CT + features clínicas radiológicas.

## Resultados del modelo
- Accuracy: 85.2%
- AUC-ROC:  0.916

## API desplegada

El servicio está corriendo en:
https://luisdam-oncoscan-ai.hf.space

Documentación interactiva:
https://luisdam-oncoscan-ai.hf.space/docs

## Endpoint principal

POST /predict

Parámetros (form-data):
- imagen:        PNG o JPG de la CT pulmonar
- subtlety:      float [1-5]
- calcification: float [1-6]
- sphericity:    float [1-5]
- margin:        float [1-5]
- lobulation:    float [1-5]
- spiculation:   float [1-5]
- texture:       float [1-5]
- malignancy:    float [1-5]

Respuesta:
{
  "score":          0.9959,
  "nivel_riesgo":   "ALTO",
  "recomendacion":  "Evaluacion urgente",
  "modelo_version": "multimodal-v1.0"
}

## Correr localmente

pip install -r requirements.txt
python service.py

## Tecnologías
- PyTorch + ResNet18
- MONAI
- FastAPI
- Dataset: LIDC-IDRI