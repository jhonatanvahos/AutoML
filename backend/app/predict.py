import json
import logging
from pathlib import Path
from app.src.RegressionModule import GridSearchModelRegression
from app.src.ClassificationModule import GridSearchModelClassification
from app.src.PreprocessingModule import DataPreprocessor

class PredictModel:
    """
    Clase para manejar el flujo de predicción de modelos de regresión y clasificación.
    """
    def __init__(self, config_file, file_name):
        """
        Inicializa la clase con el archivo de configuración y el nombre del archivo de predicción.
        
        Args:
            config_file (str): Ruta al archivo de configuración JSON.
            file_name (str): Nombre del archivo para realizar predicciones.
        """
        self.config_file = Path(config_file)
        self.file_name = file_name

    def load_params(self):
        """
        Carga los parámetros desde el archivo de configuración JSON.

        Returns:
            dict: Diccionario con la configuración cargada.
        """
        logging.info(f"Cargando el archivo de configuración: {self.config_file}")
        if not self.config_file.exists():
            logging.error(f"El archivo de configuración '{self.config_file}' no se encontró.")
            return None
        
        try:
            with self.config_file.open('r') as f:
                config = json.load(f)
                logging.info("Archivo de configuración cargado correctamente.")
                return config
        except json.JSONDecodeError as e:
            logging.error(f"No se pudo decodificar el archivo JSON '{self.config_file}': {e}")
            return None

    def create_project_directories(self, config):
        """
        Crea las rutas necesarias para el proyecto.

        Args:
            config (dict): Configuración del proyecto.

        Returns:
            tuple: Rutas para predicciones, modelos y transformadores.
        """
        project_path = Path(f'projects/{config.get("project_name")}')
        path_models = project_path / 'models'
        path_transforms = project_path / 'transforms'
        path_predict = project_path / 'predict.csv' if self.file_name == "predict" else Path(self.file_name)

        return path_predict, path_models, path_transforms / 'transform.pkl'

    def load_and_transform_data(self, preprocessor, path_predict, target, path_transforms, trained_features):
        """
        Carga y transforma los datos usando los transformadores entrenados.

        Args:
            preprocessor (DataPreprocessor): Instancia del preprocesador de datos.
            path_predict (Path): Ruta al archivo de predicciones.
            target (str): Columna objetivo en el dataset.
            path_transforms (Path): Ruta a los transformadores entrenados.
            trained_features (list): Lista de características entrenadas.

        Returns:
            tuple: Datos originales (X, y), datos transformados (X, y) y transformadores cargados.
        """
        logging.info("Cargando datos para predicción.")
        df = preprocessor.load_dataset_prediction(path_predict)
        logging.info(f"Tamaño de los datos cargados: {df.shape}")

        if self.file_name == "predict":
            y = df[target]
            X = df.drop(columns=[target])
        else:
            y = None
            X = df

        X_origin, y_origin = X.copy(), y.copy() if y is not None else None

        logging.info("Cargando y aplicando transformadores.")
        transformers = preprocessor.load_transformers(path_transforms)
        X_transformed, y_transformed = preprocessor.apply_transformers(transformers, X, y, trained_features)

        return X_origin, y_origin, X_transformed, y_transformed, transformers

    def run(self):
        """
        Ejecuta el flujo de predicción según el tipo de modelo.

        Returns:
            dict: Resultados de las predicciones.
        """
        config = self.load_params()
        if not config:
            return {"message": "Error al cargar la configuración."}

        path_predict, path_models, path_transforms = self.create_project_directories(config)
        preprocessor = DataPreprocessor(config)

        grid_search = (
            GridSearchModelRegression(config)
            if config.get("model_type") == 'Regression'
            else GridSearchModelClassification(config)
        )
        
        selected_model = config.get("selected_model")
        target = config.get("target_column")
        trained_features = config.get("trained_features")
        path_model = path_models / f"{selected_model}.pkl"

        logging.info("--------------------------------------------------------------")
        logging.info("---------------- Adecuacion de los datos ---------------------")
        logging.info("--------------------------------------------------------------")
        X_origin, y_origin, X_transformed, y_transformed, transformers = self.load_and_transform_data(
            preprocessor, path_predict, target, path_transforms, trained_features
        )

        logging.info(f"Cargando el modelo {selected_model}.")
        model = grid_search.load_model(path_model)

        logging.info("--------------------------------------------------------------")
        logging.info("------------------ Realizar predicción -----------------------")
        logging.info("--------------------------------------------------------------")
        if self.file_name == "predict":
            logging.info("Realizando predicciones con datos etiquetados.")
            result = grid_search.prediction(X_origin, X_transformed, y_transformed, transformers, target, model)
        else:
            logging.info("Realizando predicciones con datos reales.")
            result = grid_search.predict_real_data(X_origin, X_transformed, transformers, target, model)

        logging.info("Predicciones realizadas con éxito.")
        return result
