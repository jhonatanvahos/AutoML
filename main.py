from src.RegressionModule import GridSearchModelRegression
from src.ClassificationModule import GridSearchModelClassification
from src.PreprocessingModule import DataPreprocessor
from sklearn.pipeline import Pipeline
import pandas as pd
import json
import sys

if __name__ == "__main__":
    #---------------------------------------------------------------#
    #----------------- Cargar archivo de parametros ----------------#
    #---------------------------------------------------------------#
    # Función para cargar los parámetros
    def load_params(config_file):
        with open(config_file, 'r') as f:
            config = json.load(f)
        return config
    
    # carga de parametros necesarios para orquestar el proyecto.
    config_file = 'config.json'
    config = load_params(config_file)
    function = config.get("function")
    model_type = config.get("model_type")
    target_column = config.get("target_column")

    path_predict = 'data/predict_' + config.get('project_name')
    path_models = 'models/' +config.get('project_name') 
    path_transforms = 'transforms/' +config.get('project_name')
    
    #---------------------------------------------------------------#
    #---------------------- Cargar instancias ----------------------#
    #---------------------------------------------------------------#
    preprocessor = DataPreprocessor(config)
    grid_search_regression = GridSearchModelRegression(config)
    grid_search_classification = GridSearchModelClassification(config)

    #---------------------------------------------------------------#
    #----------------------- Regresión -----------------------------#
    #---------------------------------------------------------------#
    if model_type == 'Regression':
        #---------------------------------------------------------------#
        #------------------ Sección de Entrenamiento -------------------#
        #---------------------------------------------------------------#
        if function == 'training':
            print("---------------------------------------------------")
            sys.stdout.flush()
            print("------- Preprocesamiento de los datos -------------")
            sys.stdout.flush()
            print("---------------------------------------------------")
            sys.stdout.flush() 

            # Carga del dataset y configuraciones
            preprocessor.load_dataset()
            # Descripción de los datos
            preprocessor.descriptive_analysis()
            # Guardar un porcentaje de datos para predicciones
            preprocessor.split_data_for_predictions(path_predict)
            # Eliminar datos atipicos de las variables numericas
            preprocessor.remove_outliers_adjusted_zscore()
            # Ajustar el preprocesador a los datos
            preprocessor.fit()
            # Transformar los datos de entrenamiento
            preprocessor.transform()
            # Seleccion de caracteristicas representativas 
            preprocessor.select_features()
            # Guardar transformadores
            preprocessor.save_transformers(path_transforms)

            # Obtener las variables predictoras X y a predecir y procesadas.
            X,y = preprocessor.get_processed_dataframe()
            
            print("---------------------------------------------------")
            sys.stdout.flush()
            print("----------- Entrenamiento de modelos --------------")
            sys.stdout.flush()
            print("---------------------------------------------------")
            sys.stdout.flush()
 
            # Realizar busqueda de hiperparámetros y obtener el mejor modelo según la métrica seleccionada.
            results = grid_search_regression.grid_search(X, y)

            # Competir entre los modelos para seleccionar el mejor según la métrica seleccionada.
            best_model_name = grid_search_regression.compete_models(results)

            # Modelo ganador
            print("\nMejor Modelo: ", best_model_name)
            sys.stdout.flush() 
            best_model = results[best_model_name]['mejor_modelo']
            best_hiperparameters = results[best_model_name]['mejores_hiperparametros']
            print(best_model)
            sys.stdout.flush()

            # Guardar modelo ganador.
            grid_search_regression.save_best_model(best_model,path_models)

        #---------------------------------------------------------------#
        #------------------ Sección de Predicción ----------------------#
        #---------------------------------------------------------------#
        elif function == 'predict':
            print("---------------------------------------------------")
            sys.stdout.flush()
            print("--------- Procesamiento de los datos --------------")
            sys.stdout.flush()
            print("---------------------------------------------------")
            sys.stdout.flush() 

            # cargar dataset para realizar predicciones
            df= pd.read_csv(path_predict)
            #cargar transformaciones de los datos
            transformers = preprocessor.load_transformers(path_transforms)
            #Aplicar transformaciones 
            y = df[target_column]
            X = df.drop(columns=[target_column])
            X = preprocessor.apply_transformers(transformers, X)

            print("---------------------------------------------------")
            sys.stdout.flush()
            print("------------------- Predicción  -------------------")
            sys.stdout.flush()
            print("---------------------------------------------------")
            sys.stdout.flush() 

            # Cargar el modelo
            model = grid_search_regression.load_model(path_models)
            # Usar el modelo cargado para predecir
            grid_search_regression.prediction(X, y, transformers, model)

    #---------------------------------------------------------------#
    #----------------------- Clasificación -------------------------#
    #---------------------------------------------------------------#
    elif model_type == 'Classification':
        #---------------------------------------------------------------#
        #------------------ Sección de Entrenamiento -------------------#
        #---------------------------------------------------------------#
        if function == 'training':
            print("---------------------------------------------------")
            sys.stdout.flush()
            print("------- Preprocesamiento de los datos -------------")
            sys.stdout.flush()
            print("---------------------------------------------------")
            sys.stdout.flush() 
            
            # Carga del dataset y configuraciones
            preprocessor.load_dataset()
            # Descripción de los datos
            preprocessor.descriptive_analysis()
            # Guardar un porcentaje de datos para predicciones
            preprocessor.split_data_for_predictions(path_predict)
            # Eliminar datos atipicos de las variables numericas
            preprocessor.remove_outliers_adjusted_zscore()            
            # Ajustar el preprocesador a los datos
            preprocessor.fit()
            # Transformar los datos de entrenamiento
            preprocessor.transform()
            # Seleccion de caracteristicas representativas 
            preprocessor.select_features()
            # Guardar transformadores
            preprocessor.save_transformers(path_transforms)
       
            # Obtener las variables predictoras X y a predecir y procesadas. 
            X,y = preprocessor.get_processed_dataframe() 

            print("---------------------------------------------------")
            sys.stdout.flush()
            print("----------- Entrenamiento de modelos --------------")
            sys.stdout.flush()
            print("---------------------------------------------------")
            sys.stdout.flush()

            # Realizar busqueda de hiperparámetros y obtener el mejor modelo según la métrica seleccionada.
            results = grid_search_classification.grid_search(X, y)

            # Competir entre los modelos para seleccionar el mejor según la métrica seleccionada.
            best_model_name = grid_search_classification.compete_models(results)

            print("\nMejor Modelo")
            sys.stdout.flush() 
            # Modelo ganador
            best_model = results[best_model_name]['mejor_modelo']
            best_hiperparameters = results[best_model_name]['mejores_hiperparametros']
            print(best_model)
            sys.stdout.flush()  
            
            # Guardar modelo ganador.
            grid_search_classification.save_best_model(best_model,path_models)
            
        #---------------------------------------------------------------#
        #------------------ Sección de Predicción ----------------------#
        #---------------------------------------------------------------#    
        elif function == 'predict':
            print("---------------------------------------------------")
            sys.stdout.flush()
            print("--------- Procesamiento de los datos --------------")
            sys.stdout.flush()
            print("---------------------------------------------------")
            sys.stdout.flush() 

            # cargar dataset para realizar predicciones
            df= pd.read_csv(path_predict)
            # cargar transformaciones de los datos
            transformers = preprocessor.load_transformers(path_transforms)
            # Aplicar transformaciones 
            y = df[target_column]
            X = df.drop(columns=[target_column])
            X = preprocessor.apply_transformers(transformers, X)
         
            print("---------------------------------------------------")
            sys.stdout.flush()
            print("------------------- Predicción  -------------------")
            sys.stdout.flush()
            print("---------------------------------------------------")
            sys.stdout.flush() 

            print("\nCargara Modelo")
            sys.stdout.flush()
            model = grid_search_classification.load_model(path_models)

            print("\nPredicciones realizadas")
            sys.stdout.flush() 
            # Usar el modelo cargado para predecir
            grid_search_classification.prediction(X, y, model)
    