import json
import logging
from pathlib import Path
from app.src.RegressionModule import GridSearchModelRegression
from app.src.ClassificationModule import GridSearchModelClassification
from app.src.PreprocessingModule import DataPreprocessor

class TrainModel:
    """
    Clase para manejar el flujo de entrenamiento de modelos de regresión y clasificación.
    """
    def __init__(self, config_file):
        """
        Inicializa la clase con el archivo de configuración.
        
        Args:
            config_file (str): Ruta al archivo de configuración JSON.
        """
        self.config_file = Path(config_file)

    def load_params(self):
        """
        Carga los parámetros del archivo de configuración JSON.

        Returns:
            dict: Diccionario con los parámetros de configuración cargados.
        """
        logging.info(f"Cargando el archivo de configuración: {self.config_file}")
        if not self.config_file.exists():
            logging.error(f"El archivo de configuración '{self.config_file}' no se encontró.")
            return None

        try:
            with self.config_file.open("r") as f:
                config = json.load(f)
                logging.info("Archivo de configuración cargado correctamente.")
                return config
        except json.JSONDecodeError as e:
            logging.error(f"No se pudo decodificar el archivo JSON '{self.config_file}': {str(e)}")
            return None

    def create_project_directories(self, config):
        """
        Crea los directorios necesarios para el proyecto.

        Args:
            config (dict): Configuración del proyecto.

        Returns:
            tuple: Rutas de los directorios de predicción, modelos y transformaciones.
        """
        project_path = Path(f'projects/{config.get("project_name")}')
        path_predict = project_path / 'predict.csv'
        path_models = project_path / 'models'
        path_transforms = project_path / 'transforms'

        path_models.mkdir(parents=True, exist_ok=True)
        path_transforms.mkdir(parents=True, exist_ok=True)

        logging.info(f"Directorios creados en {project_path}")
        return path_predict, path_models, path_transforms / 'transform.pkl'

    def save_config_to_file(self, config):
        """Guarda el diccionario de configuración en un archivo JSON."""
        logging.info("Actualizando diccionario...")
        try:
            with open(self.config_file, "w") as file:
                json.dump(config, file, indent=4) 
            logging.info(f"Configuración guardada en {self.config_file}.")
        except Exception as e:
            logging.error(f"Error al guardar la configuración: {e}")

    def run(self):
        """
        Ejecuta el flujo de trabajo de entrenamiento del modelo.

        Returns:
            dict: Resultados del entrenamiento o mensaje de error.
        """
        config = self.load_params()
        if not config:
            return {"message": "Error al cargar la configuración."}

        path_predict, path_models, path_transforms = self.create_project_directories(config)

        # Instanciar clases necesarias
        preprocessor = DataPreprocessor(config)
        grid_search_regression = GridSearchModelRegression(config)
        grid_search_classification = GridSearchModelClassification(config)

        logging.info("--------------------------------------------------------------")
        logging.info("------------ Preprocesamiento de los datos -------------------")
        logging.info("--------------------------------------------------------------")
        try:
            preprocessor.load_dataset()
            preprocessor.split_data_for_predictions(path_predict)
            preprocessor.remove_outliers_zscore()
            preprocessor.fit_transform()
            preprocessor.apply_transform()
            preprocessor.select_features()
            preprocessor.save_transformers(path_transforms)
        except Exception as e:
            logging.error(f"Error en el preprocesamiento de datos: {str(e)}")
            return {"message": "Error en el preprocesamiento de datos."}

        X, y = preprocessor.get_processed_dataframe()
        self.save_config_to_file(config)
        logging.info("Preprocesamiento completado con éxito.")

        logging.info("--------------------------------------------------------------")
        logging.info("--------------- Competencia de Modelos -----------------------")
        logging.info("--------------------------------------------------------------")
        model_type = config.get("model_type")
        if model_type == "Regression":
            logging.info("Inicia entrenamiento de Modelos de Regresión")
            results = grid_search_regression.grid_search(X, y, path_models)
        elif model_type == "Classification":
            logging.info("Inicia entrenamiento de Modelos de Clasificación")
            results = grid_search_classification.grid_search(X, y, path_models)
        else:
            logging.error(f"Tipo de modelo '{model_type}' no reconocido.")
            return {"message": f"Tipo de modelo '{model_type}' no reconocido."}

        logging.info("Entrenamiento completado con éxito.")
        return results
