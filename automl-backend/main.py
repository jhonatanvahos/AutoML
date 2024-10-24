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
UPLOAD_DIRECTORY = "./uploaded_files"
if not os.path.exists(UPLOAD_DIRECTORY):
    os.makedirs(UPLOAD_DIRECTORY)
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
    variable_imputer: str
    imputer_n_neighbors: int
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

@app.post("/save-config")
async def save_config(data: ConfigDataHome):
    try:
        # Save the received data into config.json
        with open(config_path, 'w') as f:
            json.dump(data.dict(), f, indent=4)
        return {"message": "Config saved successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail="Error saving config") from e
    
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
        # Guardar el archivo en el directorio de destino
        file_path = os.path.join(UPLOAD_DIRECTORY, file.filename)
        
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        logging.info(f"File saved at: {file_path}")

        # Verificación del tipo de archivo
        file_extension = file.filename.split('.')[-1].lower()
        if file_extension == 'csv':
            try:
                df = pd.read_csv(file_path, delimiter=',')
                logging.info("CSV loaded with ',' separator")
            except pd.errors.ParserError:
                try:
                    df = pd.read_csv(file_path, delimiter=';')
                    logging.info("CSV loaded with ';' separator")
                except pd.errors.ParserError:
                    try:
                        df = pd.read_csv(file_path, delimiter='|')
                        logging.info("CSV loaded with '|' separator")
                    except pd.errors.ParserError:
                        raise HTTPException(status_code=400, detail="Unsupported CSV format")
        elif file_extension == 'xlsx':
            df = pd.read_excel(file_path)
            logging.info("XLSX file loaded successfully")
        else:
            raise HTTPException(status_code=400, detail="Unsupported file type")
        
        return {"message": "File uploaded successfully", "file_path": file_path}

    except Exception as e:
        logging.error(f"Error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error uploading file: {str(e)}")
    
@app.post("/train")
async def train_models():
    try:
        trainer = TrainModel("config.json")
        model_name, score = trainer.run()

        model_results = {"model_name": model_name,
                         "score": score}

        return model_results

    except Exception as e:
        print(f"Error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error training model: {str(e)}")
    
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