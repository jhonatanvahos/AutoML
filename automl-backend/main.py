from fastapi import FastAPI,File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Union
import json
from train import TrainModel
from predict import PredictModel
import shutil
import os
import logging
import pandas as pd
from pathlib import Path

logging.basicConfig(level=logging.INFO)

app = FastAPI()

# Crear un directorio para almacenar los archivos subidos, si no existe
# Directorio raíz para los archivos subidos
# Define directorios y tamaño máximo de archivo
project_directory = ""
BASE_UPLOAD_DIRECTORY = "uploads"
FINAL_DIRECTORY = "projects"  # Directorio para guardar los proyectos finales

MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB Peso máximo permitido 

# Habilitar CORS para permitir peticiones desde el frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Permitir todos los dominios
    allow_credentials=True,
    allow_methods=["*"],  # Permitir todos los métodos (GET, POST, etc.)
    allow_headers=["*"],  # Permitir todos los encabezados
)

# Path where config.json will be saved
config_path = Path('config.json')

# Estructura de datos esperados del Home congiruación inicial
class ConfigDataHome(BaseModel):
    project_name: str
    target_column: str
    dataset_path: str

# Estructura de datos esperada para la configuración detallada de cada proyecto
# Clase base común
class ModelParams(BaseModel):
    pass

# Parámetros para cada modelo de regresión
class LinearRegressionParams(ModelParams):
    fit_intercept: List[bool]

class RidgeParams(ModelParams):
    alpha: List[float]

class RandomForestParams(ModelParams):
    n_estimators: List[int]
    max_depth: List[int]
    max_features: List[str]
    criterion: List[str]

class AdaBoostParams(ModelParams):
    n_estimators: List[int]
    learning_rate: List[float]

class GradientBoostingParams(ModelParams):
    n_estimators: List[int]
    learning_rate: List[float]
    max_depth: List[int]

class LightGBMParams(ModelParams):
    n_estimators: List[int]
    max_depth: List[int]
    learning_rate: List[float]
    num_leaves: List[int]

# Parámetros para cada modelo de clasificación
class LogisticRegressionParams(ModelParams):
    multi_class: List[str]
    solver: List[str]
    class_weight: List[str]
    max_iter: List[int]

class RandomForestClassifierParams(ModelParams):
    n_estimators: List[int]
    max_features: List[int]
    max_depth: List[int]
    criterion: List[str]

class SVMParams(ModelParams):
    kernel: List[str]
    C: List[float]
    gamma: List[Union[str, float]]
    degree: List[int]
    coef0: List[float]

class KNNParams(ModelParams):
    n_neighbors: List[int]
    weights: List[str]
    metric: List[str]
    p: List[int]

class ConfigData(BaseModel):
    split: float
    missing_threshold: float
    numeric_imputer: str
    categorical_imputer: str
    imputer_n_neighbors_n: int
    imputer_n_neighbors_c: int
    scaling_method_features: str
    scaling_method_target: str
    threshold_outlier: float
    balance_method: str
    select_sampler: str
    balance_threshold: float
    k_features: float
    feature_selector_method: str
    pca_n_components: float
    delete_columns: List[str]
    model_type: str
    function: str
    n_jobs: int
    cv: int
    scoring_regression: str
    scoring_classification: str
    random_state: int
    model_competition: str
    models_regression: Dict[str, bool]
    models_classification: Dict[str, bool]
    params_regression: Dict[str, Union[LinearRegressionParams, RidgeParams, RandomForestParams, AdaBoostParams, GradientBoostingParams, LightGBMParams]]
    params_classification: Dict[str, Union[LogisticRegressionParams, RandomForestClassifierParams, SVMParams, KNNParams, ModelParams, ModelParams, ModelParams]]
    advanced_options: bool

# Guardar en configuracion aparte para las predicciones
config_model_file_path = "config_predict.json"
class ModelSelection(BaseModel):
    selectedModel: str

@app.post("/save-config")
async def save_config(data: ConfigDataHome):
    try:
        print(data)
        if not os.path.exists(FINAL_DIRECTORY):
            os.makedirs(FINAL_DIRECTORY)
        # Crear una carpeta específica para el proyecto
        project_directory = os.path.join(FINAL_DIRECTORY, data.project_name)
        os.makedirs(project_directory, exist_ok=True)
        
        # Mover el archivo a la nueva ubicación
        final_file_path = os.path.join(project_directory, os.path.basename(data.dataset_path))
        shutil.move(data.dataset_path, final_file_path)
        logging.info(f"File moved to: {final_file_path}")

        # Guardar la ruta del dataset en el archivo de configuración
        config_data = data.dict()
        config_data["dataset_path"] = final_file_path  # Actualizar la ruta al directorio del proyecto
        
        # Guardar la configuración en config.json
        with open(config_path, 'w') as f:
            json.dump(config_data, f, indent=4)
        
        logging.info("Configuration saved successfully")

        # Eliminar el directorio temporal
        if os.path.exists(BASE_UPLOAD_DIRECTORY):
            shutil.rmtree(BASE_UPLOAD_DIRECTORY)
            logging.info(f"Temporary upload directory '{BASE_UPLOAD_DIRECTORY}' removed successfully.")


        return {"message": "Config saved successfully", "dataset_path": project_directory}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error saving config: {str(e)}")
  
@app.post("/update-config")
async def update_config(config_data: ConfigData):
    try:
        # Leer la configuración actual desde el archivo config.json
        with open(config_path, "r") as config_file:
            current_config = json.load(config_file)

        # Convertir los datos entrantes a un diccionario
        new_config = config_data.dict()

        # Actualizar solo las partes que vengan en el request
        current_config.update(new_config)

        # Guardar los datos actualizados en config.json
        with open(config_path, "w") as config_file:
            json.dump(current_config, config_file, indent=4)

        return {"message": "Config updated successfully!"}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error updating config: {str(e)}")

@app.get("/get-config")
async def get_config():
    try:
        # Leer los datos desde el archivo config.json
        with open("config.json", "r") as config_file:
            config_data = json.load(config_file)
        return config_data
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading config: {str(e)}")

@app.post("/upload-dataset")
async def upload_dataset(file: UploadFile = File(...)):
    logging.info(f"Received file: {file.filename}")

    # Verificar el tamaño del archivo
    if file.size > MAX_FILE_SIZE:
        raise HTTPException(status_code=400, detail="File too large")

    try:
        if not os.path.exists(BASE_UPLOAD_DIRECTORY):
            os.makedirs(BASE_UPLOAD_DIRECTORY)
            
        # Guardar el archivo en el directorio de destino
        temp_file_path = os.path.join(BASE_UPLOAD_DIRECTORY, file.filename)
        
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        logging.info(f"File saved at: {temp_file_path}")
        # Verificación del tipo de archivo
        file_extension = file.filename.split('.')[-1].lower()
        if file_extension == 'csv':
            separators = [",", ";", "|"]
            for sep in separators:
                try:
                    # Leer las primeras filas para probar el separador
                    temp_df = pd.read_csv(temp_file_path, sep=sep, nrows=5)
                    
                    # Validar que el archivo está correctamente separado (más de 1 columna)
                    if len(temp_df.columns) > 1:
                        df = pd.read_csv(temp_file_path, sep=sep)
                        print(f"Archivo CSV cargado correctamente con separador '{sep}'")
                        break  # Si se carga correctamente, salir del bucle
                    else:
                        print(f"Separador '{sep}' no parece ser el correcto. Intentando con otro.")
                
                except pd.errors.ParserError:
                    print(f"Separador '{sep}' no funcionó, probando con otro.")
            else:
                raise ValueError("Ninguno de los separadores funcionó para el archivo CSV.")
                
        elif file_extension in ['.xls', '.xlsx']:
            df = pd.read_excel(temp_file_path)
            logging.info("XLSX file loaded successfully")
        else:
            raise HTTPException(status_code=400, detail="Unsupported file type")
        

        columns = df.columns.tolist()
        print(columns)
        return {
            "message": "File uploaded successfully",
            "file_path": temp_file_path,
            "columns": columns
        } 

    except Exception as e:
        logging.error(f"Error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error uploading file: {str(e)}")
    
@app.post("/train")
async def train_models():
    try:
        trainer = TrainModel("config.json")
        results = trainer.run()
        print("-"*100)
        print(results)
        print("-"*100)
        return results

    except Exception as e:
        print(f"Error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error training model: {str(e)}")

@app.post("/save-model-selection")
async def save_model_selection(selection: ModelSelection):
    try:
        # Completar la ruta  para que se guarde en el proyecto el modelo seleccionado
        print("-" *100 )
        print(project_directory , config_model_file_path)
        print()
        config_model_file_path_save= os.path.join(project_directory, config_model_file_path)
        # Guardar el modelo seleccionado en el archivo config_predict.json
        with open(config_model_file_path_save, "w") as f:
            json.dump({"selected_model": selection.selectedModel}, f)
        return {"message": "Model selection saved successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail="Failed to save model selection")

@app.post("/predict")
async def predict_models():
    try:
        predict = PredictModel("config.json")
        result = predict.run()
        print(result)
        return result

    except Exception as e:
        print(f"Error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error training model: {str(e)}")