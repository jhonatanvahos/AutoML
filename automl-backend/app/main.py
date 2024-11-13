# main.py
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sse_starlette import EventSourceResponse
import json
import os
import logging
from pathlib import Path
import pandas as pd
import shutil
import asyncio
import time

# Imports de otros módulos
from app.train import TrainModel
from app.predict import PredictModel
from app.models import (ConfigDataHome, ConfigData, ModelSelection, LinearRegressionParams, RidgeParams, 
                        RandomForestParams, AdaBoostParams, GradientBoostingParams, LightGBMParams, 
                        LogisticRegressionParams, RandomForestClassifierParams, SVMParams, KNNParams, PredictRequest)

# Configuración de logging
logging.basicConfig(level=logging.INFO)

# Inicialización de la aplicación FastAPI
app = FastAPI()

# Configuración de middleware CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Constantes y configuraciones de rutas
BASE_UPLOAD_DIRECTORY = "uploads"
PROJECTS_DIRECTORY = "projects"
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB
config_path = Path('app/config.json')

project_directory = None
config_project = "config_project.json"
status_file = Path("training_status.json")

# Funciones auxiliares
def create_project_directory(project_name: str) -> str:
    """Crea el directorio del proyecto si no existe y retorna su ruta."""
    project_directory = os.path.join(PROJECTS_DIRECTORY, project_name)
    os.makedirs(project_directory, exist_ok=True)
    return project_directory

def save_json(data: dict, file_path: Path):
    """Guarda un diccionario en un archivo JSON."""
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=4)
    logging.info(f"JSON data saved at: {file_path}")

# Endpoints de la API
@app.post("/save-config")
async def save_config(data: ConfigDataHome):
    """Guarda la configuración inicial del proyecto."""
    global project_directory
    try:
        project_directory = create_project_directory(data.project_name)
        final_file_path = os.path.join(project_directory, os.path.basename(data.dataset_path))
        shutil.move(data.dataset_path, final_file_path)
        config_data = data.dict()
        config_data["dataset_path"] = final_file_path
        save_json(config_data, config_path)
        return {"message": "Config saved successfully", "dataset_path": project_directory}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error saving config: {str(e)}")

@app.post("/update-config")
async def update_config(config_data: ConfigData):
    """Actualiza la configuración detallada del proyecto."""
    try:
        with open(config_path, "r") as config_file:
            current_config = json.load(config_file)
        current_config.update(config_data.dict())
        save_json(current_config, config_path)
        return {"message": "Config updated successfully!"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error updating config: {str(e)}")

@app.get("/get-config")
async def get_config():
    """Devuelve la configuración actual del proyecto."""
    try:
        with open(config_path, "r") as config_file:
            config_data = json.load(config_file)
        return config_data
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading config: {str(e)}")

@app.post("/upload-dataset")
async def upload_dataset(file: UploadFile = File(...)):
    """Sube y valida un archivo de dataset."""
    logging.info(f"Received file: {file.filename}")
    if file.size > MAX_FILE_SIZE:
        raise HTTPException(status_code=400, detail="File too large")
    try:
        os.makedirs(BASE_UPLOAD_DIRECTORY, exist_ok=True)
        temp_file_path = os.path.join(BASE_UPLOAD_DIRECTORY, file.filename)
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        logging.info(f"File saved at: {temp_file_path}")

        # Validación del archivo (detecta tipo y separador)
        df = None
        file_extension = file.filename.split('.')[-1].lower()
        if file_extension == 'csv':
            separators = [",", ";", "|"]
            for sep in separators:
                try:
                    df = pd.read_csv(temp_file_path, sep=sep, nrows=5)
                    if len(df.columns) > 1:
                        df = pd.read_csv(temp_file_path, sep=sep)
                        logging.info(f"CSV loaded with separator '{sep}'")
                        break
                except pd.errors.ParserError:
                    continue
            if df is None:
                raise HTTPException(status_code=400, detail="Invalid CSV format")
        elif file_extension in ['xls', 'xlsx']:
            df = pd.read_excel(temp_file_path)
            logging.info("Excel file loaded successfully")
        else:
            raise HTTPException(status_code=400, detail="Unsupported file type")
        
        return {"message": "File uploaded successfully", "file_path": temp_file_path, "columns": df.columns.tolist()}
    except Exception as e:
        logging.error(f"Error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error uploading file: {str(e)}")

#------------------------ Entrenamiento ----------------------
@app.post("/train")
async def train_models():
    """Inicia el entrenamiento del modelo basado en la configuración."""
    try:
        trainer = TrainModel(config_path)
        results = trainer.run()
        return results
    except Exception as e:
        logging.error(f"Error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error training model: {str(e)}")
"""
@app.get("/train/status")
async def get_training_status():
    if status_file.exists():
        with status_file.open("r") as f:
            return json.load(f)
    else:
        raise HTTPException(status_code=404, detail="Training status not found")
"""
# Función que lee el archivo y lo devuelve como un JSON
def get_training_status_from_file():
    if status_file.exists():
        with status_file.open("r") as f:
            return json.load(f)
    else:
        raise HTTPException(status_code=404, detail="Training status not found")

# Función que emite el estado del entrenamiento usando SSE
async def training_status_event():
    while True:
        try:
            # Obtener el estado de entrenamiento desde el archivo
            status_data = get_training_status_from_file()

            # Enviar los datos correctamente como un JSON, sin 'data:'
            yield f"{json.dumps(status_data)}\n\n"  # Asegúrate de enviar 'data:'

            # Esperar 1 segundo entre cada envío
            await asyncio.sleep(1)
        except asyncio.CancelledError:
            # Manejar el caso en el que el cliente se desconecta (al cancelar la conexión)
            print("Cliente desconectado. Cerrando la conexión SSE.")
            break
        except Exception as e:
            # Si ocurre un error, lo registramos y continuamos
            print(f"Error al enviar los datos: {e}")
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

@app.get("/train/status")
async def train_status():
    return EventSourceResponse(training_status_event())

@app.post("/save-model-selection")
async def save_model_selection(selection: ModelSelection):
    """Guarda el modelo seleccionado por el usuario en el archivo de configuración dentro del proyecto."""
    global project_directory  # Asegúrate de que use el valor de la variable global
    
    try:
        if project_directory is None:
            raise HTTPException(status_code=400, detail="Project directory not set.")

        # Ruta donde se guardará el nuevo archivo config_project.json dentro del proyecto
        config_model_file_path_save = Path(project_directory) / config_project
        # Leer la configuración actual de app/config.json
        try:
            with open(config_path, "r") as f:
                config_data = json.load(f)
        except FileNotFoundError:
            raise HTTPException(status_code=404, detail=f"Config file {config_path} not found.")
        
        # Agregar el modelo seleccionado a la configuración
        config_data["selected_model"] = selection.selectedModel
        
        # Guardar la configuración actualizada en config_project.json dentro del proyecto
        with open(config_model_file_path_save, "w") as f:
            json.dump(config_data, f, indent=4)

        return {"message": "Model selection and configuration saved successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save model selection: {str(e)}")

#------------------------ Predicciones ----------------------
@app.get("/api/projects")
async def get_projects():
    projects_path = 'projects'
    try:
        projects = [d for d in os.listdir(projects_path) if os.path.isdir(os.path.join(projects_path, d))]
        return {"projects": projects}
    except Exception as e:
        return {"error": str(e)}
    

@app.post("/api/predict")
async def predict_models(request: PredictRequest):
    """Ejecuta la predicción en el modelo entrenado."""
    try:
        project_name = request.project
        file_name = request.file

        # Configura la ruta del proyecto
        project_path = os.path.join(PROJECTS_DIRECTORY, project_name)
        config_file = os.path.join(project_path, config_project)
  
        # Inicializa el objeto de predicción
        predictor = PredictModel(config_file)
        
        # Ejecutar predicción (puedes ajustar esta parte según las necesidades de tu lógica)
        result = predictor.run()
        return JSONResponse(content=result)
    
    except Exception as e:
        logging.error(f"Error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error during prediction: {str(e)}")