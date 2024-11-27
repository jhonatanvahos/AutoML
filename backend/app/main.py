# Importar librerias
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import os
import logging
from pathlib import Path
import pandas as pd
import shutil
import json
import time

# Importar módulos propios
from app.models import (ConfigDataHome, ConfigData, ModelSelection, PredictRequest)
from app.train import TrainModel
from app.predict import PredictModel

#----------------------------------------------------------------------------------------------
#--------------------------------- Configuración del log --------------------------------------
#----------------------------------------------------------------------------------------------
log_dir = 'logs'
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

timestamp = time.strftime('%Y%m%d_%H%M%S')
log_filename = os.path.join(log_dir, f'log_{timestamp}.txt')

logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(), 
        logging.FileHandler(log_filename, mode='w') 
    ]
)

#----------------------------------------------------------------------------------------------
#-------------------------------- Configuración de FastAPI ------------------------------------
#----------------------------------------------------------------------------------------------
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

#----------------------------------------------------------------------------------------------
#-------------------------------- Variables globales ------------------------------------------
#----------------------------------------------------------------------------------------------
BASE_UPLOAD_DIRECTORY = "uploads"
PROJECTS_DIRECTORY = "projects"
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB
CONFIG_PATH = Path("app/config.json")
PROJECT = None
CONFIG_PROJECT = "config_project.json"
DATA = pd.DataFrame()

#----------------------------------------------------------------------------------------------
#--------------------------------- Funciones Auxiliares ---------------------------------------
#----------------------------------------------------------------------------------------------
def create_project_directory(project_name: str) -> str:
    """
    Crea el directorio del proyecto si no existe y retorna su ruta.

    Args:
        project_name (str): Nombre del proyecto.

    Returns:
        str: Ruta del directorio del proyecto.
    """
    PROJECT = os.path.join(PROJECTS_DIRECTORY, project_name)
    os.makedirs(PROJECT, exist_ok=True)
    logging.info(f"Directorio del proyecto creado en: {PROJECT}")
    return PROJECT

def save_json(data: dict, file_path: Path):
    """
    Guarda un diccionario en un archivo JSON.

    Args:
        data (dict): Datos a guardar.
        file_path (Path): Ruta del archivo JSON.
    """
    with open(file_path, "w") as f:
        json.dump(data, f, indent=4)
    logging.info(f"JSON guardado en: {file_path}")

def delete_directory(directory):
    """
    Borra un directorio y todo su contenido si existe.

    :param directorio: Ruta del directorio a eliminar.
    """
    if os.path.exists(directory):
        try:
            shutil.rmtree(directory)  # Borra el directorio y todo su contenido
            logging.info(f"Directorio '{directory}' eliminado exitosamente.")
        except Exception as e:
            logging.error(f"Error al intentar eliminar el directorio '{directory}': {e}")
    else:
        logging.warning(f"El directorio '{directory}' no existe.")

#----------------------------------------------------------------------------------------------
#----------------------------  Rutas API conexion con el Front --------------------------------
#----------------------------------------------------------------------------------------------
@app.post("/save-config")
async def save_config(data: ConfigDataHome):
    """
    Guarda la configuración inicial del proyecto.

    Args:
        data (ConfigDataHome): Datos de configuración inicial.

    Returns:
        dict: Mensaje de confirmación y ruta del dataset.
    """
    global PROJECT

    try:
        
        PROJECT = create_project_directory(data.project_name)
        final_file_path = os.path.join(PROJECT, os.path.basename(data.dataset_path))
        shutil.move(data.dataset_path, final_file_path)
        config_data = data.dict()
        config_data["dataset_path"] = final_file_path
        save_json(config_data, CONFIG_PATH)

        logging.info("--------------------------------------------------------------")
        logging.info(f"PROYECTO {data.project_name.upper()}")
        logging.info("--------------------------------------------------------------")
        return {"message": "Config saved successfully", "dataset_path": PROJECT}
    
    except Exception as e:
        logging.error(f"Error guardando la configuracion: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error saving config: {str(e)}")

@app.post("/upload-dataset")
async def upload_dataset(file: UploadFile = File(...)):
    """
    Sube y valida un archivo de dataset.

    Args:
        file (UploadFile): Archivo subido por el usuario.

    Returns:
        dict: Información del archivo subido, incluyendo columnas.
    """
    global DATA

    logging.info(f"Recibido el archivo: {file.filename}")

    if file.size > MAX_FILE_SIZE:
        raise HTTPException(status_code=400, detail="File too large")

    try:
        os.makedirs(BASE_UPLOAD_DIRECTORY, exist_ok=True)
        temp_file_path = os.path.join(BASE_UPLOAD_DIRECTORY, file.filename)
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        logging.info(f"Archivo guardado en: {temp_file_path}")

        # Detectar tipo de archivo y cargar
        df = None
        file_extension = file.filename.split(".")[-1].lower()

        if file_extension == "csv":
            separators = [",", ";", "|"]
            for sep in separators:
                try:
                    df = pd.read_csv(temp_file_path, sep=sep, nrows=5)
                    if len(df.columns) > 1:
                        df = pd.read_csv(temp_file_path, sep=sep)
                        logging.info(f"CSV cargado exitosamente con el separador: '{sep}'")
                        break
                except pd.errors.ParserError:
                    continue
            if df is None:
                raise HTTPException(status_code=400, detail="Invalid CSV format")
        elif file_extension in ["xls", "xlsx"]:
            df = pd.read_excel(temp_file_path)
            logging.info("Excel cargado exitosamente")
        else:
            raise HTTPException(status_code=400, detail="Unsupported file type")

        DATA = df.copy()

        return {"message": "File uploaded successfully", "file_path": temp_file_path, "columns": df.columns.tolist()}
    
    except Exception as e:
        logging.error(f"Error cargando el archivo: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error uploading file: {str(e)}")

@app.get("/preview-dataset")
async def preview_dataset():
    """
    Devuelve una vista previa del dataset cargado, con imputación básica para valores nulos.

    Returns:
        dict: Vista previa del dataset, columnas numéricas y categóricas.
    """
    global DATA

    try:
        # Identificar columnas numéricas y categóricas
        numeric_columns = DATA.select_dtypes(include=["number"]).columns.tolist()
        categorical_columns = DATA.select_dtypes(include=["object", "category"]).columns.tolist()

        # Crear una copia del dataset para evitar modificar el original
        preview_data = DATA.copy()

        # Imputación sencilla de valores nulos
        preview_data[numeric_columns] = preview_data[numeric_columns].fillna(0)
        preview_data[categorical_columns] = preview_data[categorical_columns].fillna("N/A")

        # Preparar la respuesta
        response = {
            "dataPreview": preview_data.to_dict(orient="records"),
            "numericColumns": numeric_columns,
            "categoricalColumns": categorical_columns,
        }
        logging.info("Datos para la previsualización cargados exitosamente")

        return response

    except Exception as e:
        logging.error(f"Error en la previsualización del dataset: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error previewing dataset: {str(e)}")


@app.post("/update-config")
async def update_config(config_data: ConfigData):
    """
    Actualiza la configuración detallada del proyecto.

    Args:
        config_data (ConfigData): Datos de configuración a actualizar.

    Returns:
        dict: Mensaje de confirmación de la actualización.
    """
    try:
        with open(CONFIG_PATH, "r") as config_file:
            current_config = json.load(config_file)
        
        # Actualizar la configuración existente
        current_config.update(config_data.dict())
        save_json(current_config, CONFIG_PATH)

        return {"message": "Config updated successfully"}
    
    except Exception as e:
        logging.error(f"Error actualizando la configuracion: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error updating configuration: {str(e)}")

#----------------------------------------------------------------------------------------------
#----------------------------- Entrenamiento de los modelos -----------------------------------
#----------------------------------------------------------------------------------------------
@app.post("/train")
async def train_models():
    """
    Inicia el entrenamiento del modelo basado en la configuración.

    Returns:
        dict: Resultados del entrenamiento.
    """
    try:
        logging.info("--------------------------------------------------------------")
        logging.info("--------------- ENTRENAMIENTO DE MODELOS ---------------------")
        logging.info("--------------------------------------------------------------")
        trainer = TrainModel(CONFIG_PATH)
        results = trainer.run()

        return results
    
    except Exception as e:
        logging.error(f"Error entrenando los modelos: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error training model: {str(e)}")

@app.post("/save-model-selection")
async def save_model_selection(selection: ModelSelection):
    """
    Guarda el modelo seleccionado en la configuración del proyecto.

    Args:
        selection (ModelSelection): Modelo seleccionado por el usuario.

    Returns:
        dict: Mensaje de confirmación.
    """
    global PROJECT
    
    logging.info("Actualizando los parámetros para guardar el modelo seleccionado...")
    try:
        if PROJECT is None:
            raise HTTPException(status_code=400, detail="Project directory is not set.")
        
        # Ruta del archivo de configuración dentro del proyecto
        config_model_file_path = Path(PROJECT) / CONFIG_PROJECT
       
        # Leer la configuración actual
        with open(CONFIG_PATH, "r") as f:
            config_data = json.load(f)

        # Actualizar la configuración con el modelo seleccionado
        config_data["selected_model"] = selection.selectedModel
        save_json(config_data, config_model_file_path)
        
        return {"message": "Model selection and configuration saved successfully"}
    
    except Exception as e:
        logging.error(f"Error guardando el modelo seleccionado: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error saving model selection: {str(e)}")

#----------------------------------------------------------------------------------------------
#------------------------- Predicción con modelos entrenados ----------------------------------
#----------------------------------------------------------------------------------------------
@app.get("/api/projects")
async def get_projects():
    """
    Lista todos los proyectos disponibles en el directorio de proyectos.

    Returns:
        dict: Lista de nombres de proyectos.
    """
    try:
        projects = [
            d for d in os.listdir(PROJECTS_DIRECTORY)
            if os.path.isdir(os.path.join(PROJECTS_DIRECTORY, d))
        ]
        logging.info("Listado de proyectos exitosa")

        return {"projects": projects}
    
    except Exception as e:
        logging.error(f"Error listando los proyectos: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error listing projects: {str(e)}")

@app.post("/api/predict")
async def predict_models(request: PredictRequest):
    """
    Ejecuta predicciones usando el modelo seleccionado.

    Args:
        request (PredictRequest): Información del proyecto y archivo de datos.

    Returns:
        dict: Resultados de la predicción.
    """
    try:
        project_name = request.project
        file_name = request.file

        # Configura la ruta del proyecto
        project_path = os.path.join(PROJECTS_DIRECTORY, project_name)
        config_file = os.path.join(project_path, CONFIG_PROJECT)
        
        logging.info("--------------------------------------------------------------")
        logging.info("---------------------- PREDICCION ----------------------------")
        logging.info("--------------------------------------------------------------")
        # Inicializa el objeto de predicción
        predictor = PredictModel(config_file, file_name)
        result = predictor.run()

        delete_directory(BASE_UPLOAD_DIRECTORY)

        return result
    
    except Exception as e:
        logging.error(f"Error durante la prediccion: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error during prediction: {str(e)}")