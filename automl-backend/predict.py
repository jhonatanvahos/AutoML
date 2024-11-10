from src.RegressionModule import GridSearchModelRegression
from src.ClassificationModule import GridSearchModelClassification
from src.PreprocessingModule import DataPreprocessor
import json
import os

class PredictModel:
    def __init__(self, config_file):
        self.config_file = config_file

    def load_params(self):

        print(f"Cargando el archivo de configuración: {self.config_file}")
        config_path = os.path.join(os.path.dirname(__file__), self.config_file)
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
                print("Archivo de configuración cargado correctamente.")
                return config
        except FileNotFoundError:
            print(f"Error: El archivo de configuración '{self.config_file}' no se encontró.")
            return None
        except json.JSONDecodeError:
            print(f"Error: No se pudo decodificar el archivo JSON '{self.config_file}'.")
            return None

    def run(self):
        # Cargar parámetros
        config = self.load_params()
        path_predict = 'projects/' + config.get('project_name') + '/predict' 
        path_models = 'projects/' + config.get('project_name') + '/model'
        path_transforms = 'projects/' + config.get('project_name') + '/transform'

        # Instanciar clases
        preprocessor = DataPreprocessor(config)
        grid_search_regression = GridSearchModelRegression(config)
        grid_search_classification = GridSearchModelClassification(config)
        
        model_type = config.get("model_type")
        target = config.get("target_column")

        if model_type == 'Regression':
            # cargar dataset para realizar predicciones
            X, y = preprocessor.load_dataset_prediction(path_predict, target)
            # cargar y aplicar las transformaciones de los datos
            transformers = preprocessor.load_transformers(path_transforms)
            X = preprocessor.apply_transformers(transformers, X)

            print("Cargar Modelo")
            model = grid_search_regression.load_model(path_models)

            print("Predicciones realizadas")
            grid_search_regression.prediction(X, y, transformers, model)
            
            return "OK"
        
        elif model_type == 'Classification':
            print("Inicia predicción")

            # cargar dataset para realizar predicciones
            X, y = preprocessor.load_dataset_prediction(path_predict, target)
            print(X, y)
            # cargar y aplicar las transformaciones de los datos
            
            transformers = preprocessor.load_transformers(path_transforms)
            X, y = preprocessor.apply_transformers(transformers, X , y)

            print("Cargar Modelo")
            model = grid_search_classification.load_model(path_models)
            print(model)
            print("Predicciones realizadas")
            grid_search_classification.prediction(X, y, model)
            
            print("OK")
            
            return "OK"