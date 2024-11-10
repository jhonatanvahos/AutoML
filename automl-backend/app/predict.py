import json
import logging
from pathlib import Path
from app.src.RegressionModule import GridSearchModelRegression
from app.src.ClassificationModule import GridSearchModelClassification
from app.src.PreprocessingModule import DataPreprocessor

class PredictModel:
    def __init__(self, config_file):
        self.config_file = Path(config_file)

    def load_params(self):
        """Carga los parámetros del archivo de configuración JSON."""
        logging.info(f"Cargando el archivo de configuración: {self.config_file}")
        if not self.config_file.exists():
            logging.error(f"Error: El archivo de configuración '{self.config_file}' no se encontró.")
            return None
        
        try:
            with self.config_file.open('r') as f:
                config = json.load(f)
                logging.info("Archivo de configuración cargado correctamente.")
                return config
        except json.JSONDecodeError:
            logging.error(f"Error: No se pudo decodificar el archivo JSON '{self.config_file}'.")
            return None

    def create_project_directories(self, config):
        """Crea las rutas necesarias para el proyecto y devuelve las rutas de los datos y modelos."""
        project_path = Path(f'projects/{config.get("project_name")}')
        path_predict = project_path / 'predict'
        path_models = project_path / 'models'
        path_transforms = project_path / 'transforms'

        return path_predict, path_models, path_transforms / 'transform'

    def load_and_transform_data(self, preprocessor, path_predict, target, path_transforms):
        """Carga y transforma los datos usando los transformadores ya entrenados."""
        logging.info("Cargando datos para predicción.")
        X, y = preprocessor.load_dataset_prediction(path_predict, target)
        X_origin = X.copy()
        y_origin = y.copy()

        logging.info("Cargando transformadores.")
        transformers = preprocessor.load_transformers(path_transforms)

        logging.info("Aplicando transformadores.")
        X, y = preprocessor.apply_transformers(transformers, X, y)
        return X_origin, y_origin, X, y, transformers

    def run(self):
        """Ejecuta el flujo de predicción del modelo."""
        # Cargar parámetros de configuración
        config = self.load_params()
        if not config:
            return {"message": "Error loading configuration."}

        # Crear las rutas necesarias
        path_predict, path_models, path_transforms = self.create_project_directories(config)

        # Instanciar clases
        preprocessor = DataPreprocessor(config)
        grid_search_regression = GridSearchModelRegression(config)
        grid_search_classification = GridSearchModelClassification(config)
        
        model_type = config.get("model_type")
        selected_model = config.get("selected_model") 
        target = config.get("target_column")
        path_models = path_models / selected_model

        if model_type == 'Regression':
            # Cargar datos y realizar predicciones para regresión
            X_origin, y_origin, X, y, tranforms = self.load_and_transform_data(preprocessor, path_predict, target, path_transforms)

            # Cargar modelo y realizar predicción
            logging.info("Cargando el modelo de regresión.")
            model = grid_search_regression.load_model(path_models)
            logging.info("Realizando predicciones.")
            result = grid_search_regression.prediction(X_origin, y_origin, X, y, tranforms, model)

        elif model_type == 'Classification':
            # Cargar datos y realizar predicciones para clasificación
            X_origin, y_origin, X, y, tranforms= self.load_and_transform_data(preprocessor, path_predict, target, path_transforms)

            # Cargar modelo y realizar predicción
            logging.info("Cargando el modelo de clasificación.")
            model = grid_search_classification.load_model(path_models)
            logging.info("Realizando predicciones.")
            result = grid_search_classification.prediction(X_origin, y_origin, X, y, model)

        else:
            logging.error(f"Tipo de modelo '{model_type}' no reconocido.")
            return {"message": f"Model type '{model_type}' not recognized."}

        logging.info("Predicciones realizadas con éxito.")
        return result
