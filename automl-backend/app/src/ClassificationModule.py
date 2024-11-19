from time import time
import numpy as np
import pandas as pd
import json
import sys
import joblib
from pathlib import Path
from datetime import datetime

#Modelos
from sklearn.linear_model import LinearRegression, Ridge, LogisticRegression
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score, precision_recall_curve, auc
from sklearn.model_selection import GridSearchCV
from skopt import BayesSearchCV
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB

class GridSearchModelClassification:
    def __init__(self, config):
        self.config = config
        self.models = {}
        self.cv = None
        self.scoring = None
        self.n_jobs = None
        self.random_state = self.config.get('random_state' , 1234)

        self.cv = self.config.get('cv', 5)
        self.scoring = self.config.get('scoring_classification', 'f1')
        self.n_jobs = self.config.get('n_jobs', -1)
        self.model_competition = self.config.get('model_competition', 'Grid_Search')

        for model_name, modelFlag in self.config['models_classification'].items():
            if modelFlag:
                if model_name in self.config["params_classification"]:
                    hiperparameters = self.config["params_classification"][model_name] 
                    self.models[model_name] = {'model': model_name, 'hiperparameters': hiperparameters}
                else:
                    print(f"No se encontraron hiperparámetros para el modelo {model_name}.")
    
    # Función para la busqueda de grilla. 
    def grid_search(self, X, y, path_models): 
        print("Busqueda de grilla: ")
        sys.stdout.flush() 

        results = {}
        results_format = {}
        # Archivo temporal para almacenar el estado del entrenamiento
        status_file = Path("training_status.json")
        # Ciclo para recorrer cada uno de los modelos seleccionados en los parámetros
        for model_name, config in self.models.items():
            print(model_name)
            start_time = self.start_model_training(model_name, status_file)
            
            model = config['model']
            hiperparameters = config['hiperparameters']
            print(f'hiperparametros a probar para {model} son: {hiperparameters}')
            sys.stdout.flush()

            if model == 'random_forest':
                estimator = RandomForestClassifier(random_state = self.random_state)
            if model == 'logisticRegression':
                estimator = LogisticRegression(random_state = self.random_state)
            elif model == 'SVM':
                estimator = SVC(random_state = self.random_state)
            elif model == 'KNN':
                estimator = KNeighborsClassifier()
            elif model == "GaussianNB":
                estimator = GaussianNB()
            elif model == "MultinomialNB":
                estimator = MultinomialNB()
            elif model == "BernoulliNB":
                estimator = BernoulliNB()

            print('Model competition : ', self.model_competition)
            if self.model_competition == 'Grid_Search':
                # Grid_search
                grid_search = GridSearchCV(estimator = estimator, param_grid = hiperparameters, scoring = self.scoring,
                                            cv = self.cv, n_jobs = self.n_jobs)
                
                grid_search.fit(X, y)
                mejores_hiperparametros = grid_search.best_params_
                mejor_modelo = grid_search.best_estimator_
                score = grid_search.best_score_

            elif self.model_competition == 'Bayes_Search':
                # Bayes_search
                bayes_search = BayesSearchCV(estimator = estimator, search_spaces = hiperparameters, scoring = self.scoring,
                                            cv = self.cv, n_jobs = self.n_jobs)
                bayes_search.fit(X, y)
                mejores_hiperparametros = bayes_search.best_params_
                mejor_modelo = bayes_search.best_estimator_
                score = bayes_search.best_score_
            else: 
                pass
            results[model_name] = {'mejor_modelo': mejor_modelo, 'mejores_hiperparametros': mejores_hiperparametros, f'score_{self.scoring}': score}
            
            print(f'score_{self.scoring}: ', score)
            sys.stdout.flush()

            elapsed_time = self.finish_model_training( model_name, start_time, status_file)
            print("Tiempo transcurrido durante el entrenamiento:", elapsed_time, "minutos")
            sys.stdout.flush()

            filename = path_models / model_name
            
            self.save_model(results[model_name]['mejor_modelo'],filename)
            
            # Modifica el diccionario para que solo incluya el nombre del modelo y no el objeto completo
            results_format[model_name] = {
                'mejor_modelo': type(mejor_modelo).__name__,  # Solo el nombre del modelo
                'mejores_hiperparametros': mejores_hiperparametros,
                'score': score,
                'elapsed_time_minutes': round(elapsed_time / 60, 2)
            }
                
        return results_format
    
    def start_model_training(self, model_name, status_file):
        # Inicia el registro del modelo actual y la hora de inicio
        start_time = datetime.now()
        with status_file.open("w") as f:
            json.dump({
                "timestamp": start_time.isoformat(),
                "current_model": model_name,
                "progress": "Entrenamiento iniciado",
                "elapsed_time_minutes": 0
            }, f)
        return start_time

    def finish_model_training(self, model_name, start_time, status_file):
        # Calcula el tiempo transcurrido y guarda el estado final
        elapsed_time = (datetime.now() - start_time).total_seconds() / 60  # en minutos
        with status_file.open("w") as f:
            json.dump({
                "timestamp": datetime.now().isoformat(),
                "current_model": model_name,
                "progress": "Entrenamiento completedo",
                "elapsed_time_minutes": round(elapsed_time, 2)
            }, f)
        return elapsed_time
    
    # Función para la selección del mejor modelo
    def compete_models(self, results):
        print("\nCompetencia de Modelos")
        sys.stdout.flush() 

        # Comparación del Score seleccionado (F1 score).
        best_model_name = None
        best_score = 0.0  
        
        for model_name, result in results.items():
            score = result[f'score_{self.scoring}']
            if score > best_score:
                best_score = score
                best_model_name = model_name

        return best_model_name

    # Funcion para guardar el mejor modelo
    def save_model(self, best_model, filename):
        try:
            # Guarda el diccionario en un archivo usando joblib
            joblib.dump(best_model, filename)
            print(f"El modelo se guardó en '{filename}'.")
        except Exception as e:
            print(f"Error al guardar el modelo: {e}")

    # Función para cargar modelo.
    def load_model(self, filename):
        print("\nCargar Modelo: ")
        sys.stdout.flush() 

        try:
            # Carga el diccionario del archivo usando joblib
            model = joblib.load(filename)
            print(f"El modelo se cargaron desde '{filename}'.")
            return model
        except Exception as e:
            print(f"Error al cargar el modelo: {e}")

    # Fución para realizar la predicción.
    def prediction(self, X_origin, y_origin, X, y, model):
        print("\nRealizar predicciones: ")
        sys.stdout.flush()

        # Realizar predicción
        result = model.predict(X)
        #proba = model.predict_proba(X)

        # Transformar y (asegurarse de que los valores estén en formato numérico)
        y = [0 if label == 'no' else 1 for label in y]

        # Calcular métricas
        ac = accuracy_score(y, result)
        pc = precision_score(y, result)
        f1 = f1_score(y, result)
        cm = confusion_matrix(y, result)

        print("Accuracy: ", ac)
        sys.stdout.flush()
        print("Precision: ", pc)
        sys.stdout.flush()
        print("F1_Score: ", f1)
        sys.stdout.flush()
        print("Matriz de confusion: ", cm.tolist())
        sys.stdout.flush()

        # Crear DataFrame con resultados
        df_result = pd.DataFrame({'y': y, 'prediccion': result.ravel()})
        
        # Incluir las variables originales (X_origin)
        for idx, col in enumerate(X_origin.columns):
            df_result[col] = X_origin.iloc[:, idx].values

        df_result['match'] = df_result['y'] == df_result['prediccion']
        print(df_result)
        sys.stdout.flush()

        count_true = df_result["match"].astype(int).sum()
        count_false = df_result.shape[0] - df_result["match"].astype(int).sum()
        print('Acertó:', count_true)
        sys.stdout.flush()
        print('Se equivocó:', count_false)
        sys.stdout.flush()
        print('%: ', count_true / df_result.shape[0])
        sys.stdout.flush()

        # Construir el diccionario con los resultados
        result_predict = {
            "data" : "test",
            "model_type": "classification",
            "accuracy": ac,
            "precision": pc,
            "f1_score": f1,
            "total_predictions": df_result.shape[0],
            "correct_predictions": count_true,
            "incorrect_predictions": count_false,
            "prediction_accuracy": count_true / df_result.shape[0],
            "confusion_matrix": cm.tolist(),  # Matriz de confusión como lista para JSON
            "actual_values": y,  # Valores reales
            "predicted_values": result.tolist(),  # Valores predichos
            "predictions": df_result.to_dict(orient="records")  # Convertir el DataFrame a lista de diccionarios
        }

        # Convertir a tipos serializables
        def convert_to_serializable(obj):
            if isinstance(obj, (np.int64, np.int32)):
                return int(obj)
            elif isinstance(obj, (np.float64, np.float32)):
                return float(obj)
            elif isinstance(obj, list):
                return [convert_to_serializable(item) for item in obj]
            elif isinstance(obj, dict):
                return {key: convert_to_serializable(value) for key, value in obj.items()}
            return obj

        result_predict = convert_to_serializable(result_predict)

        return result_predict
    
    def predict_real_data(self, X_origin, X, model):
        """
        Realiza predicciones en datos reales (sin variable objetivo).
        
        Args:
            X_origin (pd.DataFrame): Datos originales (sin la columna objetivo).
            model: Modelo previamente entrenado.
        
        Returns:
            dict: Resultados de las predicciones incluyendo los datos originales y las predicciones.
        """
        print("\nRealizando predicciones en datos reales:")
        sys.stdout.flush()

        # Realizar predicción
        predictions = model.predict(X)
        
        # Si el modelo tiene probabilidad, puedes agregarlo opcionalmente
        # probabilities = model.predict_proba(X_origin)

        # Crear DataFrame con resultados
        df_result = pd.DataFrame({'prediccion': predictions.ravel()})
        
        # Incluir las variables originales (X_origin)
        for idx, col in enumerate(X_origin.columns):
            df_result[col] = X_origin.iloc[:, idx].values

        print("Predicciones completadas.")
        sys.stdout.flush()
        print(df_result)
        sys.stdout.flush()

        # Construir el diccionario con los resultados
        result_predict = {
            "data" : "real",
            "model_type": "classification",  # Cambia según el tipo de modelo
            "total_predictions": df_result.shape[0],
            "predictions": df_result.to_dict(orient="records")  # Convertir el DataFrame a lista de diccionarios
        }

        # Convertir a tipos serializables
        def convert_to_serializable(obj):
            if isinstance(obj, (np.int64, np.int32)):
                return int(obj)
            elif isinstance(obj, (np.float64, np.float32)):
                return float(obj)
            elif isinstance(obj, list):
                return [convert_to_serializable(item) for item in obj]
            elif isinstance(obj, dict):
                return {key: convert_to_serializable(value) for key, value in obj.items()}
            return obj

        result_predict = convert_to_serializable(result_predict)

        return result_predict
