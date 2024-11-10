from src.RegressionModule import GridSearchModelRegression
from src.ClassificationModule import GridSearchModelClassification
from src.PreprocessingModule import DataPreprocessor
import json
import os

class TrainModel:
    def __init__(self, config_file):
        self.config_file = config_file
        print(self.config_file)

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
        path_models = 'projects/' + config.get('project_name') + '/models'
        path_transforms = 'projects/' + config.get('project_name') + '/transforms'

        if not os.path.exists(path_models):
            os.makedirs(path_models)

        if not os.path.exists(path_transforms):
            os.makedirs(path_transforms)
        path_transforms += '/transform'

        # Instanciar clases
        preprocessor = DataPreprocessor(config)
        grid_search_regression = GridSearchModelRegression(config)
        grid_search_classification = GridSearchModelClassification(config)
        
        # Preprocesamiento de datos
        preprocessor.load_dataset()
        #preprocessor.descriptive_analysis()
        preprocessor.split_data_for_predictions(path_predict)
        preprocessor.remove_outliers_adjusted_zscore()
        preprocessor.fit()
        preprocessor.transform()
        preprocessor.select_features()

        # Guardar transformadores 
        preprocessor.save_transformers(path_transforms) 
        X, y = preprocessor.get_processed_dataframe()
        print(X, y)
        print("TERMINA OK todas las transformaciones")
        
        model_type = config.get("model_type")

        if model_type == 'Regression':
            results = grid_search_regression.grid_search(X, y, path_models)
            return results
        
        elif model_type == 'Classification':
            results = grid_search_classification.grid_search(X, y, path_models)
            return results
        