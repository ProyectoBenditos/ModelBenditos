# OncaScan AI Service

Microservicio de inferencia para deteccion de nodulos pulmonares a partir de:

- Imagen CT pulmonar
- Features clinicas/radiologicas ingresadas manualmente

El servicio expone una API HTTP con FastAPI para que otro backend pueda solicitar predicciones del modelo multimodal.

## Estado del proyecto

- Modelo principal: `best_model_multimodal.pth`
- Arquitectura: `ResNet18` para imagen + MLP para features clinicas
- Tipo de tarea: clasificacion binaria de riesgo
- Metrica reportada en la rama: `AUC-ROC 0.916`
- Ejecucion actual del servicio: `CPU`

## Contenido del repositorio

- `service.py`: API FastAPI para inferencia
- `demo.py`: demo local del modelo base de imagen
- `demo_endtoend.py`: demo interactivo multimodal con visualizacion
- `best_model_multimodal.pth`: pesos del modelo multimodal
- `best_model.pth`: pesos del modelo base de imagen
- `caso_*.png`: casos de prueba
- `resultado_*.png`: salidas generadas por demos locales

## API desplegada

Segun la documentacion de esta rama, el servicio esta desplegado en Hugging Face Spaces:

- URL base: `https://luisdam-oncoscan-ai.hf.space`
- Swagger UI: `https://luisdam-oncoscan-ai.hf.space/docs`

Si el Space sigue activo, desde ahi puedes probar el endpoint sin correr nada localmente.

## Requisitos

- Python 3.10 o superior recomendado
- `pip`
- Dependencias listadas en `requirements.txt`

Instalacion:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Ejecutar localmente

Levanta la API con:

```powershell
python service.py
```

Una vez arriba:

- Root: `http://localhost:8000/`
- Health check: `http://localhost:8000/health`
- Swagger UI: `http://localhost:8000/docs`

## Endpoints

### `GET /`

Devuelve estado basico del servicio y link de documentacion.

### `GET /health`

Devuelve informacion simple del modelo cargado:

- estado
- tipo de modelo
- dispositivo
- version

### `POST /predict`

Recibe `multipart/form-data`.

Campos requeridos:

- `imagen`: archivo `PNG`, `JPG` o `JPEG`
- `subtlety`: `float` entre `1` y `5`
- `calcification`: `float` entre `1` y `6`
- `sphericity`: `float` entre `1` y `5`
- `margin`: `float` entre `1` y `5`
- `lobulation`: `float` entre `1` y `5`
- `spiculation`: `float` entre `1` y `5`
- `texture`: `float` entre `1` y `5`
- `malignancy`: `float` entre `1` y `5`

Respuesta esperada:

```json
{
  "score": 0.9959,
  "nivel_riesgo": "ALTO",
  "recomendacion": "Evaluacion urgente",
  "modelo_version": "multimodal-v1.0"
}
```

## Ejemplo con curl

```powershell
curl -X POST "http://localhost:8000/predict" `
  -F "imagen=@caso_02_nodulo.png" `
  -F "subtlety=4" `
  -F "calcification=2" `
  -F "sphericity=3" `
  -F "margin=4" `
  -F "lobulation=3" `
  -F "spiculation=4" `
  -F "texture=5" `
  -F "malignancy=4"
```

## Logica de salida

El servicio devuelve un `score` entre `0` y `1` y lo traduce a categorias:

- `0.00 - 0.32`: `BAJO`
- `0.33 - 0.65`: `MEDIO`
- `0.66 - 1.00`: `ALTO`

Recomendaciones actuales:

- `BAJO`: control rutinario
- `MEDIO`: seguimiento recomendado
- `ALTO`: evaluacion urgente

## Notas de implementacion

- El preprocesamiento convierte la imagen a escala de grises, la redimensiona a `64x64` y replica el canal a 3 canales para alimentar `ResNet18`.
- Las features clinicas se normalizan con medias y desviaciones estandar fijas embebidas en el codigo.
- El servicio permite `CORS` abierto con `allow_origins=["*"]`, util para pruebas pero no ideal para produccion.
- El modelo se carga una vez al importar `service.py`, no durante un startup lifecycle explicito del framework.

## Limitaciones actuales

- No hay autenticacion ni rate limiting.
- No hay manejo avanzado de errores para archivos corruptos o muy grandes.
- El dispositivo esta fijado en `cpu`.
- Los pesos del modelo viven dentro del repositorio.
- No hay suite de tests para la API.

## Stack

- PyTorch
- Torchvision
- FastAPI
- Uvicorn
- Pillow
- NumPy

## Aviso

Este repositorio documenta un prototipo tecnico de inferencia. No debe considerarse una herramienta diagnostica clinica ni sustituye criterio medico profesional.
