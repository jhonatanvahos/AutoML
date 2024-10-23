from time import time
import numpy as np
import pandas as pd
import json
import sys
import joblib

#Modelos
from sklearn.linear_model import LinearRegression, Ridge
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score, precision_recall_curve, auc
from sklearn.model_selection import GridSearchCV
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
        self.scoring = self.config.get('scoring_classification', 'f1_score')
        self.n_jobs = self.config.get('n_jobs', -1)

        for model_name, modelFlag in self.config['models_classification'].items():
            if modelFlag:
                if model_name in self.config["params_classification"]:
                    hiperparameters = self.config["params_classification"][model_name] 
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

            if model == 'random_forest':
                estimator = RandomForestClassifier(random_state = self.random_state)
            elif model == 'ada_boost':
                estimator = AdaBoostClassifier(random_state = self.random_state, algorithm = 'SAMME')
            elif model == 'gradient_boosting':
                estimator = GradientBoostingClassifier(random_state = self.random_state)
            elif model == 'lightGBM':
                estimator = LGBMClassifier(verbose=-1,random_state = self.random_state)
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

            # Grid_search
            grid_search = GridSearchCV(estimator = estimator, param_grid = hiperparameters, scoring = self.scoring,
                                        cv = self.cv, n_jobs = self.n_jobs)
            
            grid_search.fit(X, y)
            mejores_hiperparametros = grid_search.best_params_
            mejor_modelo = grid_search.best_estimator_
            score = grid_search.best_score_
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
            print(f"El modelo se cargaron desde '{filename}'.")
            return model
        except Exception as e:
            print(f"Error al cargar el modelo: {e}")

    # Fución para realizar la predicción.
    def prediction(self, X, y, model):
        print("\nRelizar predicciones: ")
        sys.stdout.flush() 

        result = model.predict(X)
        proba = model.predict_proba(X)
        
        # Calcular metricas
        ac = accuracy_score(y, result)
        pc = precision_score(y, result)
        f1 = f1_score(y, result)

        print("Accuracy: ", ac)
        sys.stdout.flush() 
        print("Precission: ", pc)
        sys.stdout.flush() 
        print("F1_Score: ", f1)
        sys.stdout.flush() 

        # Crear un DataFrame con la variable objetivo y las predicciones
        df_result = pd.DataFrame({'y': y, 'result_predict': result.ravel()})
        # Agregar las probabilidades para cada clase como nuevas columnas en el DataFrame
        for i in range(proba.shape[1]):
            df_result[f'probability_class_{i}'] = proba[:, i]

        # Comparar si 'y' y 'result_predict' son iguales
        df_result['match'] = df_result['y'] == df_result['result_predict']
        print(df_result)
        sys.stdout.flush() 

        count_true = df_result["match"].astype(int).sum()
        count_false = df_result.shape[0] - df_result["match"].astype(int).sum()
        print('acertó:' , count_true)
        sys.stdout.flush() 
        print('se equivoco:', count_false)
        sys.stdout.flush() 
        print('% : ', count_true/df_result.shape[0])
        sys.stdout.flush() 