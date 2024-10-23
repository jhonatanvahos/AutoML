from time import time
import numpy as np
import pandas as pd
import json
import sys
import joblib

#Modelos
from sklearn.linear_model import LinearRegression, Ridge
import lightgbm as lgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import GridSearchCV

class GridSearchModelRegression:
    def __init__(self, config):
        self.config = config
        self.models = {}
        self.cv = None
        self.scoring = None
        self.n_jobs = None
        self.random_state = self.config.get('random_state' , 1234)

        self.cv = self.config.get('cv', 5)
        self.scoring = self.config.get('scoring_regression', 'neg_mean_absolute_error')
        self.n_jobs = self.config.get('n_jobs', -1)

        for model_name, modelFlag in self.config['models_regression'].items():
            if modelFlag:
                if model_name in self.config["params_regression"]:
                    hiperparameters = self.config["params_regression"][model_name] 
                    self.models[model_name] = {'model': model_name, 'hiperparameters': hiperparameters}
                else:
                    print(f"No se encontraron hiperparámetros para el modelo {model_name}.")

    # Función para la busqueda de grilla. 
    def grid_search(self, X, y): 
        print("Busqueda de grilla: ")
        sys.stdout.flush()    

        results = {}
        # Ciclo para recorrer cada uno de los modelos seleccionados en los parámetros
        for model_name, config in self.models.items():
            start_time = time()
            model = config['model']
            hiperparameters = config['hiperparameters']

            print(f'hiperparametros a probar para {model} son: {hiperparameters}')
            sys.stdout.flush()

            if model == 'linearRegression': 
                estimator = LinearRegression()
            elif model == 'ridge':
                estimator = Ridge()
            elif model == 'random_forest':
                estimator = RandomForestRegressor(random_state = self.random_state)
            elif model == 'ada_boost':
                estimator = AdaBoostRegressor(random_state = self.random_state)
            elif model == 'gradient_boosting':
                estimator = GradientBoostingRegressor(random_state = self.random_state)
            elif model == 'lightGBM':
                estimator = lgb(random_state = self.random_state)

            # Grid_search
            grid_search = GridSearchCV(estimator = estimator, param_grid = hiperparameters, scoring = self.scoring,
                                        cv = self.cv, n_jobs = self.n_jobs)
            
            grid_search.fit(X, y)
            mejores_hiperparametros = grid_search.best_params_
            mejor_modelo = grid_search.best_estimator_
            score = -grid_search.best_score_
            results[model_name] = {'mejor_modelo': mejor_modelo, 'mejores_hiperparametros': mejores_hiperparametros, f'score_{self.scoring}': score}
            
            print(f'score_{self.scoring}: ', score)
            sys.stdout.flush()

            end_time = time()
            elapsed_time = end_time - start_time
            print("Tiempo transcurrido durante el entrenamiento:", elapsed_time/60, "minutos")
            sys.stdout.flush()

        return results

    # Función para la selección del mejor modelo
    def compete_models(self, results):
        print("\nCompetencia de Modelos")
        sys.stdout.flush() 

        # Comparación del Score seleccionado (MSE)
        best_model_name = None
        best_score = float('inf') 

        for model_name, result in results.items():
            score = result[f'score_{self.scoring}']
            if score < best_score:
                best_score = score
                best_model_name = model_name

        return best_model_name

    # Funcion para guardar el mejor modelo
    def save_best_model(self, best_model, filename):
        print("\nGuardar modelo ganador: ")
        sys.stdout.flush()

        try:
            # Guarda el diccionario en un archivo usando joblib
            joblib.dump(best_model, filename)
            print(f"El mejor modelo se guardó en '{filename}'.")
        except Exception as e:
            print(f"Error al guardar el mejor modelo: {e}")

    # Función para cargar modelo.
    def load_model(self, filename):
        print("\nCargar Modelo: ")
        sys.stdout.flush() 
        try:
            # Carga el diccionario del archivo usando joblib
            model = joblib.load(filename)
            print(f"El modelo se cargó desde '{filename}'.")
            return model
        except Exception as e:
            print(f"Error al cargar el modelo: {e}")

    # Fución para realizar la predicción.
    def prediction(self, X, y, transformers, model):
        print("\nRelizar predicciones: ")
        sys.stdout.flush() 
        
        # Si se aplicó el escalamiento de datos de y invertir el transformador para obtener el valor real.
        if 'scaler_y' in transformers:
            transformer = transformers['scaler_y']
            result = model.predict(X)
            result_real = transformer.inverse_transform(result.reshape(-1, 1))
        else:
            result = model.predict(X)
            result_real = result

        # Calcular métricas
        mse = mean_squared_error(y, result_real.ravel())
        mae = mean_absolute_error(y, result_real.ravel())
        r2 = r2_score(y, result_real.ravel())

        print("Error Cuadrático Medio (MSE):", mse)
        sys.stdout.flush() 
        print("Error Absoluto Medio (MAE):", mae)
        sys.stdout.flush() 
        print("Coeficiente de Determinación (R^2):", r2)
        sys.stdout.flush() 

        # Crear un DataFrame con la variable objetivo y las predicciones
        df_result = pd.DataFrame({'y': y, 'result_predict': result_real.ravel()})
        df_result['difference'] = df_result['y'] - df_result['result_predict']
        # Formatear las dos columnas en el DataFrame
        df_result = df_result.applymap(lambda x: '{:,.2f}'.format(x))
        print(df_result)
        sys.stdout.flush() 