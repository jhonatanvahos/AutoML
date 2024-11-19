import json
import sys
import logging
from pathlib import Path
from app.src.RegressionModule import GridSearchModelRegression
from app.src.ClassificationModule import GridSearchModelClassification
from app.src.PreprocessingModule import DataPreprocessor

class TrainModel:
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
        """Crea los directorios necesarios para el proyecto."""
        project_path = Path(f'projects/{config.get("project_name")}')
        path_predict = project_path / 'predict.csv'
        path_models = project_path / 'models'
        path_transforms = project_path / 'transforms'

        # Crear directorios si no existen
        path_models.mkdir(parents=True, exist_ok=True)
        path_transforms.mkdir(parents=True, exist_ok=True)
        
        return path_predict, path_models, path_transforms / 'transform'

    def run(self):
        """Ejecuta el flujo de trabajo de entrenamiento del modelo."""
        # Cargar parámetros de configuración
        config = self.load_params()
        if not config:
            return {"message": "Error loading configuration."}

        # Crear los directorios necesarios
        path_predict, path_models, path_transforms = self.create_project_directories(config)
    
        # Instanciar clases
        preprocessor = DataPreprocessor(config)
        grid_search_regression = GridSearchModelRegression(config)
        grid_search_classification = GridSearchModelClassification(config)
        
        # Preprocesamiento de datos
        preprocessor.load_dataset()
        preprocessor.split_data_for_predictions(path_predict)
        preprocessor.remove_outliers_zscore()
        #preprocessor.remove_outliers_adjusted_zscore()
        preprocessor.fit()
        preprocessor.transform()
        preprocessor.select_features()

        # Guardar transformadores
        preprocessor.save_transformers(path_transforms)
        X, y = preprocessor.get_processed_dataframe()
        print("Diccionario actualizado: ", config)
        sys.stdout.flush()

        try:
            with open(self.config_file, "w") as file:
                json.dump(config, file, indent=4)  # Guarda con formato legible
            print(f"Configuración guardada en {self.config_file}.")
        except Exception as e:
            print(f"Error al guardar la configuración: {e}")

        # Selección del modelo
        model_type = config.get("model_type")
        if model_type == 'Regression':
            results = grid_search_regression.grid_search(X, y, path_models)
        elif model_type == 'Classification':
            results = grid_search_classification.grid_search(X, y, path_models)
        else:
            logging.error(f"Tipo de modelo '{model_type}' no reconocido.")
            return {"message": f"Model type '{model_type}' not recognized."}

        logging.info("Entrenamiento completado con éxito.")
        return results

    def save_config_to_file(self, file_path="config.json"):
        """Guarda el diccionario de configuración en un archivo JSON."""
        try:
            with open(file_path, "w") as file:
                json.dump(self.config_file, file, indent=4)  # Guarda con formato legible
            print(f"Configuración guardada en {file_path}.")
        except Exception as e:
            print(f"Error al guardar la configuración: {e}")