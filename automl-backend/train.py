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
        path_predict = 'uploaded_files/predict_' + config.get('project_name')
        path_models = 'models/' +config.get('project_name') 
        path_transforms = 'transforms/' +config.get('project_name')

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
        scorinf_regression = config.get('scoring_regression', 'neg_mean_absolute_error') 
        scoring_classification = config.get('scoring_classification', 'f1')

        if model_type == 'Regression':
            results = grid_search_regression.grid_search(X, y)
            best_model_name = grid_search_regression.compete_models(results)
            score = results[best_model_name][f'score_{scorinf_regression}']
            
            #Guardar mejor modelo
            best_model = results[best_model_name]['mejor_modelo']
            grid_search_regression.save_best_model(best_model,path_models)
            
            return best_model_name, score
        
        elif model_type == 'Classification':
            results = grid_search_classification.grid_search(X, y)
            best_model_name = grid_search_classification.compete_models(results)
            score = results[best_model_name][f'score_{scoring_classification}']     
            
            #Guardar mejor modelo
            best_model = results[best_model_name]['mejor_modelo']
            grid_search_classification.save_best_model(best_model, path_models)

            return best_model_name, score
        