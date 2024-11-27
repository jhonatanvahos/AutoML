import logging
import numpy as np
import pandas as pd
import joblib
from datetime import datetime
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import GridSearchCV
from skopt import BayesSearchCV

class GridSearchModelRegression:
    """
    Clase para la búsqueda de hiperparámetros y evaluación de modelos de regresión.
    """
    def __init__(self, config):
        """
        Inicializa la clase con la configuración proporcionada.

        Args:
            config (dict): Diccionario de configuración con los modelos y parámetros.
        """
        self.config = config
        self.models = {}
        self.random_state = self.config.get('random_state' , 1234)
        self.cv = self.config.get('cv', 5)
        self.scoring = self.config.get('scoring_regression', 'neg_mean_absolute_error')
        self.n_jobs = self.config.get('n_jobs', -1)
        self.model_competition = config.get('model_competition', 'Grid_Search')
 
        # Configuración de modelos y parámetros
        for model_name, model_flag in self.config['models_regression'].items():
            if model_flag:
                logging.info(f"Modelo activado: {model_name}")
                if model_name in self.config["params_regression"]:
                    hiperparameters = self.config["params_regression"][model_name] 
                    logging.info(f"Hiperparametros: {hiperparameters}")
                    self.models[model_name] = {'model': model_name, 'hiperparameters': hiperparameters}
                else:
                   logging.warning(f"No se encontraron hiperparámetros para el modelo {model_name}.")

    def grid_search(self, X, y, path_models):
        """
        Realiza la búsqueda de hiperparámetros utilizando GridSearchCV o BayesSearchCV.

        Args:
            X (pd.DataFrame): Datos de entrada.
            y (pd.Series): Etiquetas de salida.
            path_models (Path): Ruta para guardar los modelos entrenados.

        Returns:
            dict: Resultados del modelo con los mejores hiperparámetros y métricas.
        """
        logging.info("Iniciando búsqueda de hiperparámetros.")
        results = {}
        for model_name, config in self.models.items():
            start_time = datetime.now()
            hiperparameters = config['hiperparameters']
            logging.info("--------------------------------------------------------------")
            logging.info(f"Probando hiperparámetros para {model_name}: {hiperparameters}")

            # Selección del modelo
            estimator = self._select_model(model_name)

            # Selección del método de búsqueda
            search = self._select_search_method(estimator, hiperparameters)

            # Entrenamiento y evaluación
            search.fit(X, y)
            mejores_hiperparametros = search.best_params_
            mejor_modelo = search.best_estimator_
            score = search.best_score_

            elapsed_time = (datetime.now() - start_time).total_seconds() / 60

            # Guardar modelo y resultados
            self.save_model(mejor_modelo, path_models / f"{model_name}.pkl")
            results[model_name] = {
                'mejor_modelo': type(mejor_modelo).__name__,
                'mejores_hiperparametros': mejores_hiperparametros,
                'score': score,
                'elapsed_time_minutes': elapsed_time
            }

            logging.info(f"Modelo: {model_name}")
            logging.info(f"Mejores hiperparámetros: {mejores_hiperparametros}")
            logging.info(f"Score_{self.scoring}: {score}")
            logging.info(f"Tiempo transcurrido: {elapsed_time:.2f} minutos")

        return results
        
    def _select_model(self, model_name):
        """
        Selecciona el modelo correspondiente.

        Args:
            model_name (str): Nombre del modelo.

        Returns:
            sklearn.base.BaseEstimator: Instancia del modelo.
        """
        if model_name == 'linearRegression': 
            return LinearRegression()
        elif model_name == 'ridge':
            return Ridge()
        elif model_name == 'random_forest':
            return RandomForestRegressor(random_state = self.random_state)
        elif model_name == 'ada_boost':
            return AdaBoostRegressor(random_state = self.random_state)
        elif model_name == 'gradient_boosting':
            return GradientBoostingRegressor(random_state = self.random_state)
        else:
            raise ValueError(f"Modelo no reconocido: {model_name}")

    def _select_search_method(self, estimator, hiperparameters):
        """
        Selecciona el método de búsqueda según la configuración.

        Args:
            estimator: Modelo estimador.
            hiperparameters (dict): Espacio de búsqueda de hiperparámetros.

        Returns:
            GridSearchCV or BayesSearchCV: Objeto de búsqueda configurado.
        """
        if self.model_competition == 'Grid_Search':
            return GridSearchCV(estimator=estimator, param_grid=hiperparameters, scoring=self.scoring, cv=self.cv, n_jobs=self.n_jobs)
        elif self.model_competition == 'Bayes_Search':
            return BayesSearchCV(estimator=estimator, search_spaces=hiperparameters, scoring=self.scoring, cv=self.cv, n_jobs=self.n_jobs)
        else:
            raise ValueError(f"Método de búsqueda no soportado: {self.model_competition}")
    
    def save_model(self, model, filename):
        """
        Guarda el modelo en un archivo.

        Args:
            model: Modelo a guardar.
            filename (Path): Ruta del archivo.
        """
        try:
            joblib.dump(model, filename)
            logging.info(f"Modelo guardado en {filename}.")
        except Exception as e:
            logging.error(f"Error al guardar el modelo: {e}")

    def load_model(self, filename):
        """
        Carga un modelo desde un archivo.

        Args:
            filename (Path): Ruta del archivo.

        Returns:
            Modelo cargado.
        """
        try:
            model = joblib.load(filename)
            logging.info(f"Modelo cargado desde {filename}.")
            return model
        except Exception as e:
            logging.error(f"Error al cargar el modelo: {e}")
            return None

    def prediction(self, X_origin, X, y, transformers, target, model):
        """
        Realiza predicciones en un conjunto de datos de prueba y calcula métricas de evaluación.

        Args:
            X_origin (pd.DataFrame): Datos originales (sin escalar ni transformar).
            y_origin (array-like): Valores originales de la variable objetivo.
            X (array-like): Datos preprocesados (escalados o transformados).
            y (array-like): Valores de la variable objetivo transformados (numéricos).
            model: Modelo previamente entrenado.

        Returns:
            dict: Resultados de la predicción, métricas y detalles del desempeño.
        """
        logging.info("Iniciando predicción en datos de testeo...")
        try:    
            #  Realizar prediccioness
            # Si se aplicó el escalamiento de datos de y invertir el transformador para obtener el valor real.
            if 'scaler_y' in transformers:
                transformer = transformers['scaler_y']['scaler']
                predictions = model.predict(X)
                predictions_original_scale = transformer.inverse_transform(predictions.reshape(-1, 1)).ravel()
            else:
                predictions_original_scale = model.predict(X)

            logging.info("Predicciones realizadas.")

            # Calcular métricas
            metrics = {
                "Error cuadrático medio": mean_squared_error(y, predictions_original_scale),
                "Error absoluto medio": mean_absolute_error(y, predictions_original_scale),
                "R2": r2_score(y, predictions_original_scale)
            }
            logging.info("Cálculo de métricas completado.")

            # Crear DataFrame de resultados
            df_result = pd.DataFrame({
                target: y, 
                f"prediccion_{target}": predictions_original_scale 
            })
            df_result['difference'] = df_result[target] - df_result[f"prediccion_{target}"]
            df_result = df_result.applymap(lambda x: '{:,.2f}'.format(x))

            # Identificar columnas numéricas y categóricas en X_origin
            numeric_columns = X_origin.select_dtypes(include=["number"]).columns.tolist()
            categorical_columns = X_origin.select_dtypes(include=["object", "category"]).columns.tolist()

            # Imputación básica
            X_origin[numeric_columns] = X_origin[numeric_columns].fillna(0)
            X_origin[categorical_columns] = X_origin[categorical_columns].fillna("N/A")

            for col in X_origin.columns:
                df_result[col] = X_origin[col].values

            # Construir el diccionario con los resultados
            result_predict = {
                "data" : "test",
                "model_type" : "regression",
                **metrics,
                "total_predictions": df_result.shape[0],
                "actual_values": y.tolist(),  # Valores reales
                "predicted_values": predictions_original_scale.tolist(),  # Valores predichos
                "predictions": df_result.to_dict(orient="records")  # Convertir el DataFrame a lista de diccionarios
            }

            # Convertir resultados a tipos serializables
            result_predict = self._convert_to_serializable(result_predict)
            logging.info("Resultados de predicción procesados exitosamente.")
            logging.info("Resultados de predicción procesados exitosamente.")
            logging.info("Detalles de las predicciones y métricas:")
            logging.info(f"Modelo: {result_predict['model_type']}")
            logging.info(f"Total de predicciones: {result_predict['total_predictions']}")
            logging.info(f"Métricas de rendimiento:")
            logging.info(f"  Error cuadrático medio: {metrics['Error cuadrático medio']}")
            logging.info(f"  Error absoluto medio: {metrics['Error absoluto medio']}")
            logging.info(f"  R2: {metrics['R2']}")
            logging.info("Primeras 5 predicciones:")
            for record in result_predict['predictions'][:5]:  
                logging.info(record)

            return result_predict
        
        except Exception as e:
            logging.error(f"Error durante la predicción: {str(e)}", exc_info=True)
            raise
    
    def predict_real_data(self, X_origin, X, transformers, target, model):
        """
        Realiza predicciones en un conjunto de datos reales sin valores objetivo.

        Args:
            X_origin (pd.DataFrame): Datos originales (sin escalar ni transformar).
            X (array-like): Datos preprocesados (escalados o transformados).
            model: Modelo previamente entrenado.

        Returns:
            dict: Resultados de las predicciones incluyendo los datos originales.
        """
        logging.info("Iniciando predicción en datos reales.")

        try:    
            # Si se aplicó el escalamiento de datos de y invertir el transformador para obtener el valor real.
            if 'scaler_y' in transformers:
                transformer = transformers['scaler_y']['scaler']
                predictions = model.predict(X)
                predictions_original_scale = transformer.inverse_transform(predictions.reshape(-1, 1)).ravel()
            else:
                predictions_original_scale = model.predict(X)

            # Crear DataFrame con resultados    
            df_result = pd.DataFrame({f"prediccion_{target}": predictions_original_scale })   
            df_result = df_result.applymap(lambda x: '{:,.2f}'.format(x))

            # Identificar columnas numéricas y categóricas en X_origin
            numeric_columns = X_origin.select_dtypes(include=["number"]).columns.tolist()
            categorical_columns = X_origin.select_dtypes(include=["object", "category"]).columns.tolist()

            # Imputación básica
            X_origin[numeric_columns] = X_origin[numeric_columns].fillna(0)
            X_origin[categorical_columns] = X_origin[categorical_columns].fillna("N/A")
            
            for col in X_origin.columns:
                df_result[col] = X_origin[col].values
    
            # Construir el diccionario con los resultados
            result_predict = {
                "data" : "real",
                "model_type": "regression",  
                "total_predictions": df_result.shape[0],
                "predictions": df_result.to_dict(orient="records")  # Convertir el DataFrame a lista de diccionarios
            }

            # Convertir resultados a tipos serializables
            result_predict = self._convert_to_serializable(result_predict)
            logging.info("Resultados de predicción procesados exitosamente.")
            logging.info("Detalles de las predicciones y métricas:")
            logging.info(f"Modelo: {result_predict['model_type']}")
            logging.info(f"Total de predicciones: {result_predict['total_predictions']}")
            logging.info("Primeras 5 predicciones:")
            for record in result_predict['predictions'][:5]:
                logging.info(record)

            return result_predict
        
        except Exception as e:
            logging.error(f"Error durante la predicción: {str(e)}", exc_info=True)
            raise

    def _convert_to_serializable(self, obj):
        """
        Convierte objetos a tipos serializables para JSON.

        Args:
            obj: Objeto a convertir.

        Returns:
            Objeto convertido a tipos básicos serializables.
        """
        if isinstance(obj, (np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, list):
            return [self._convert_to_serializable(item) for item in obj]
        elif isinstance(obj, dict):
            return {key: self._convert_to_serializable(value) for key, value in obj.items()}
        return obj