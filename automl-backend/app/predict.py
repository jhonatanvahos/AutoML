import json
import logging
from pathlib import Path
from app.src.RegressionModule import GridSearchModelRegression
from app.src.ClassificationModule import GridSearchModelClassification
from app.src.PreprocessingModule import DataPreprocessor

class PredictModel:
    def __init__(self, config_file, file_name):
        self.config_file = Path(config_file)
        self.file_name = file_name

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
        path_models = project_path / 'models'
        path_transforms = project_path / 'transforms'

        if self.file_name == "predict":
            path_predict = project_path / 'predict.csv'
        else:
            path_predict = Path(self.file_name)

        return path_predict, path_models, path_transforms / 'transform'

    def load_and_transform_data(self, preprocessor, path_predict, target, path_transforms, trained_features):
        """Carga y transforma los datos usando los transformadores ya entrenados."""
        logging.info("Cargando datos para predicción.")
        logging.info(f"path datos: {path_predict}")
        logging.info(f"target: {target}")
        df = preprocessor.load_dataset_prediction(path_predict)
        logging.info(f"tamaño de los datos {df.shape}")
        
        if self.file_name == "predict":
            y = df[target]
            X = df.drop(columns=[target])
    
            X_origin = X.copy()
            y_origin = y.copy()

            logging.info("Cargando transformadores.")
            transformers = preprocessor.load_transformers(path_transforms)

            logging.info("Aplicando transformadores.")
            X = preprocessor.apply_transformers(transformers, X, trained_features)
        
        else:
            X = df    
            y = None
            X_origin = X.copy()
            y_origin = None

            logging.info("Cargando transformadores.")
            transformers = preprocessor.load_transformers(path_transforms)

            logging.info("Aplicando transformadores.")
            X = preprocessor.apply_transformers(transformers, X, trained_features)
       
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
        trained_features = config.get("trained_features")
        path_models = path_models / selected_model

        if model_type == 'Regression':
            # Cargar datos y realizar predicciones para regresión
            X_origin, y_origin, X, y, tranforms = self.load_and_transform_data(preprocessor, path_predict, target, path_transforms, trained_features)

            # Cargar modelo y realizar predicción
            logging.info("Cargando el modelo de regresión.")
            model = grid_search_regression.load_model(path_models)
            
            if self.file_name == "predict":
                logging.info("Realizando predicciones.")
                result = grid_search_regression.prediction(X_origin, y_origin, X, y, tranforms, model)
            else:
                logging.info("Realizando predicciones.")
                result = grid_search_regression.predict_real_data(X_origin, X, tranforms, model)

        elif model_type == 'Classification':
            # Cargar datos y realizar predicciones para clasificación
            X_origin, y_origin, X, y, tranforms= self.load_and_transform_data(preprocessor, path_predict, target, path_transforms, trained_features)

            # Cargar modelo y realizar predicción
            logging.info("Cargando el modelo de clasificación.")
            model = grid_search_classification.load_model(path_models)
            
            if self.file_name == "predict":
                logging.info("Realizando predicciones.")
                result = grid_search_classification.prediction(X_origin, y_origin, X, y, model)
            else:
                logging.info("Realizando predicciones.")
                result = grid_search_classification.predict_real_data(X_origin, X, model)

        else:
            logging.error(f"Tipo de modelo '{model_type}' no reconocido.")
            return {"message": f"Model type '{model_type}' not recognized."}

        logging.info("Predicciones realizadas con éxito.")
        return result
