# Manipulación de dataframes
import pandas as pd
import numpy as np
from scipy.stats import zscore

# Sistema
import sys
import os 

# Imputacion
from sklearn.impute import SimpleImputer
from sklearn.impute import KNNImputer

# Escalado y codificacion
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MinMaxScaler
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder

# Seleccion caracteristicas
from sklearn.feature_selection import SelectKBest, f_regression, RFE, RFECV, mutual_info_classif, mutual_info_regression
from sklearn.linear_model import LinearRegression  # Para RFE y RFECV

# Reducción de dimensionalidad
from sklearn.decomposition import PCA

# balanceo de datos
from imblearn.over_sampling import SMOTE, ADASYN , RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler, ClusterCentroids, TomekLinks
from imblearn.combine import SMOTEENN, SMOTETomek

# Guardar objetos (Transformadores y Modelos)
import joblib

# graficas
import matplotlib.pyplot as plt
import seaborn as sns

class DataPreprocessor:
    def __init__(self, config):
        self.config = config
        self.path_file = self.config.get('dataset_path', None)
        self.delete_columns = self.config.get('delete_columns')
        self.split = self.config.get('split')
        self.target = self.config.get('target_column', None)
        self.model_type = self.config.get('model_type', None)

        # Variables para datos atípicos
        self.threshold_outlier = self.config.get('threshold_outlier', 3)

        # Variables para Codificadores
        self.label_encoder = LabelEncoder()
        self.one_hot_encoder = OneHotEncoder(handle_unknown='ignore')

        # Variables para imputar datos
        self.missing_threshold = self.config.get('missing_threshold', 0.1)
        self.numeric_imputer =  self.config.get("numeric_imputer")
        self.categorical_imputer = self.config.get("categorical_imputer")
        self.imputer_n_neighbors = self.config.get("imputer_n_neighbors")

        self.balance_threshold = self.config.get('balance_thershold',0.5)
        self.balance_method = self.config.get('balance_method', None)
        self.select_sampler = self.config.get('select_sampler', None)

        if self.balance_method == 'over_sampling':
            if self.select_sampler == 'SMOTE':
                self.sampler = SMOTE()
            elif self.select_sampler == 'ADASYN':
                self.samplter = ADASYN()
            elif self.select_sampler == 'RandomOverSampler':
                self.sampler = RandomOverSampler()
        elif self.balance_method == 'under_sampling':
            if self.select_sampler == 'RandomUnderSampler':
                self.sampler = RandomUnderSampler()
            elif self.select_sampler == 'ClusterCentroids':
                self.sampler = ClusterCentroids()
            elif self.select_sampler == 'TomekLinks':
                self.sampler = TomekLinks()
        elif self.balance_method == 'combine':
            if self.select_sampler == 'SMOTEENN':
                self.sampler = SMOTEENN()
            elif self.select_sampler == 'SMOTETomek':
                self.sampler = SMOTETomek()
        else:
            self.sampler = None

        # Variables para escalar datos
        self.scaling_method_target = self.config.get('scaling_method_target')
        if self.scaling_method_target == 'standard':
            self.scaler_y = StandardScaler()
        elif self.scaling_method_target == 'minmax':
            self.scaler_y = MinMaxScaler()
        else: 
            self.scaler_y = None

        self.scaling_method_features = self.config.get('scaling_method_features')
        if self.scaling_method_features == 'standard':
            self.scaler_X = StandardScaler()
        elif self.scaling_method_features == 'minmax':
            self.scaler_X = MinMaxScaler()
        else:
            self.scaler_X = None

        # Variables para selección de caracteristicas
        self.k = self.config.get('k_features')
        self.feature_selector_method = self.config.get('feature_selector_method')

        if self.feature_selector_method == "select_k_best":
            self.feature_selector = SelectKBest(score_func = f_regression)
        elif self.feature_selector_method == "rfe":
            model = LinearRegression() 
            self.feature_selector = RFE(model)
        elif self.feature_selector_method == "rfecv":
            model = LinearRegression() 
            self.feature_selector = RFECV(model, step=1, cv=5)
        elif self.feature_selector_method == "mutual_info_classif":
           self.feature_selector = SelectKBest(score_func=mutual_info_classif)  
        elif self.feature_selector_method == "mutual_info_regression":
            self.feature_selector = SelectKBest(score_func=mutual_info_regression)
        else:
            raise ValueError(f"Método de selección de características no reconocido: {self.feature_selector_method}")

        # Variables creadas 
        self.transformers = {} # para guardar las transformaciones 
        self.seed = 11 # semilla para la separacion de los datos
        self.unprocessed_columns = [] # columnas no procesadas por no cumplir criterios de nulos

    # Función para cargar los datos y hacer depuración. 
    def load_dataset(self):
        print("---------------------------------------------------")
        sys.stdout.flush()
        print("--------------- Carga de datos -------------------")
        sys.stdout.flush()
        print("---------------------------------------------------")
        sys.stdout.flush()
        try:
            # Obtener la extensión del archivo
            file_extension = os.path.splitext(self.path_file)[1].lower()
            try:
                # Si el archivo es un CSV, probar con diferentes separadores
                if file_extension == '.csv':
                    separators = [",", ";", "|"]
                    for sep in separators:
                        try:
                            # Leer las primeras filas para probar el separador
                            temp_df = pd.read_csv(self.path_file, sep=sep, nrows=5)
                            
                            # Validar que el archivo está correctamente separado (más de 1 columna)
                            if len(temp_df.columns) > 1:
                                self.df = pd.read_csv(self.path_file, sep=sep)
                                print(f"Archivo CSV cargado correctamente con separador '{sep}'")
                                break  # Si se carga correctamente, salir del bucle
                            else:
                                print(f"Separador '{sep}' no parece ser el correcto. Intentando con otro.")
                        
                        except pd.errors.ParserError:
                            print(f"Separador '{sep}' no funcionó, probando con otro.")
                    else:
                        raise ValueError("Ninguno de los separadores funcionó para el archivo CSV.")
                
                # Si el archivo es un Excel
                elif file_extension in ['.xls', '.xlsx']:
                    self.df = pd.read_excel(self.path_file)
                    print("Archivo Excel cargado correctamente.")
                
                # Si el archivo no es ni CSV ni Excel, lanzar un error
                else:
                    raise ValueError(f"Tipo de archivo no soportado: {file_extension}")

            except Exception as e:
                print(f"Error al cargar el archivo: {e}")

            print(self.df)
            self.df.columns = self.df.columns.str.strip()
            self.df.columns = self.df.columns.str.replace(' ', '_')

            print("Cantidad de registros cargados: ", self.df.shape[0])
            sys.stdout.flush()
            print("Cantidad de columnas cargadas: ", self.df.shape[1])
            sys.stdout.flush()
            print("Eliminar conlumnas indicadas por el usuario: ", self.delete_columns)
            sys.stdout.flush()
            self.df.drop(columns = self.delete_columns, inplace = True)
            
            print("Eliminar conlumnas con valores unicos, todos los valores diferentes y duplicados: ")
            sys.stdout.flush()
            # Columnas con un único valor.
            unique_columns = self.df.columns[self.df.nunique() == 1]
            self.df.drop(columns = unique_columns, inplace = True)

            # Columnas con todos los valores diferentes.
            unique_columns = self.df.columns[self.df.nunique() == len(self.df)]
            self.df.drop(columns = unique_columns, inplace = True)

            # Valores duplicados
            self.df.drop_duplicates(inplace = True)
            self.df.reset_index(drop = True, inplace = True) 

            print("Cantidad de datos nuevos ", self.df.shape)
            sys.stdout.flush()    

            # Selección de la variable objetivo
            target_column = self.target
            if target_column is None:
                raise ValueError("Variable objetivo no identificada en los parametros")
            
            # Asignación de 'y' varialbe objetivo y 'X' variables predictora 
            self.y = self.df[target_column]
            self.X = self.df.drop(columns=[target_column])

            # Identificar variables numéricas y categóricas
            self.numeric_columns = self.X.select_dtypes(include=['number','float64','float64','int32','int64']).columns
            self.categorical_columns = self.X.select_dtypes(include=['object', 'category']).columns

            return self.X, self.y

        except Exception as e:
            print("Error cargando el dataset:", e)

    # Función para realizar un analisis descriptivo y una visualización de los datos
    def descriptive_analysis(self):
        print(" Información de los datos: ")
        sys.stdout.flush() 
        print(self.df.info())
        sys.stdout.flush() 

        # Análisis descriptivo de variables numéricas
        if self.numeric_columns is not None:
            numeric_data = self.df[self.numeric_columns]
            if not numeric_data.empty:
                print("Análisis descriptivo de variables numéricas: ")
                sys.stdout.flush() 
                print(numeric_data.describe())
                sys.stdout.flush() 

                # Visualización de variables numéricas
                print("Visualización de variables numéricas: ")
                sys.stdout.flush() 
                numeric_data.hist(figsize=(12, 8))
                plt.tight_layout()
                plt.show()
        
        # Análisis descriptivo de variables categóricas
        if self.categorical_columns is not None:
            categorical_data = self.df[self.categorical_columns]
            if not categorical_data.empty: 
                print("Análisis descriptivo de variables categóricas: ")
                sys.stdout.flush() 
                print(categorical_data.describe())
                sys.stdout.flush() 

                # Visualización de variables categóricas
                print("Visualización de variables categóricas:")
                sys.stdout.flush() 
                num_plots = len(self.categorical_columns)
                num_groups = (num_plots + 5) // 6  # Calcular el número de grupos de 6
                for group in range(num_groups):
                    start_index = group * 6
                    end_index = min(start_index + 6, num_plots)
                    num_variables = end_index - start_index
                    num_rows = (num_variables + 1) // 3
                    fig, axes = plt.subplots(num_rows, 3, figsize=(15, num_rows*5), squeeze=False)
                    for i in range(start_index, end_index):
                        row_index = (i - start_index) // 3
                        col_index = (i - start_index) % 3
                        col = self.categorical_columns[i]
                        sns.countplot(data=categorical_data, x=col, hue=col, palette='viridis', ax=axes[row_index, col_index], legend=False)
                        axes[row_index, col_index].set_title(f"Distribución de {col}")
                        axes[row_index, col_index].tick_params(axis='x', rotation=45)
                    plt.tight_layout()
                    plt.show()
                    sys.stdout.flush()
    
    #Función para tratamiento de variables categóricas 
    def process_categorical_variables(self):
        limite = 20
        if self.categorical_columns is not None:
            categorical_data = self.df[self.categorical_columns]
            if not categorical_data.empty: 
                # Convertir todas las categorías a minúsculas
                categorical_data = categorical_data.apply(lambda x: x.str.lower() if x.dtype == 'object' else x)
                
                # Contar cuántos valores únicos existen
                valores_unicos = categorical_data.nunique()
                
                # Encontrar columnas que superan el límite de categorías
                columnas_con_limite = []
                for columna, n_valores_unicos in valores_unicos.items():
                    if n_valores_unicos > limite:
                        columnas_con_limite.append(columna)

                return categorical_data, valores_unicos, columnas_con_limite

    # Función para separar un % de los datos para realizar predicciones después de crear el modelo
    def split_data_for_predictions(self, save_path):

        print("Separación de datos para predecir: ")
        sys.stdout.flush()  
        # Seleccionar datos aleatorios
        np.random.seed(self.seed)
        num_rows_to_predict = int(len(self.df) * self.split)
        random_indices = np.random.choice(self.df.index, num_rows_to_predict, replace=False)
        prediction_data = self.df.loc[random_indices]#??

        # Eliminar los datos seleccionados del DataFrame original
        self.df.drop(random_indices, inplace=True)
        self.df.reset_index(drop=True, inplace=True)#??

        # Guardar los datos para predicciones en un nuevo archivo CSV
        try:
            prediction_data.to_csv(save_path, index=False)
            print(f"Datos para predicciones guardados en '{save_path}'")
            sys.stdout.flush() 
        except Exception as e:
            print("Error al guardar los datos para predicciones:", e)
            sys.stdout.flush() 
 
    # Función para remover los datos atipicos a través del z_score
    def remove_outliers_zscore(self):
        print("Eliminar valores atipicos")
        sys.stdout.flush()
        # Calcular z-scores para las columnas numéricas
        z_scores = zscore(self.X[self.numeric_columns])

        # Identificar filas con valores atípicos
        outlier_rows = (np.abs(z_scores) > self.threshold_outlier).any(axis=1)

        # Eliminar filas con valores atípicos
        self.X = self.X[~outlier_rows]
        self.y = self.y[~outlier_rows]

        self.X.reset_index(drop=True, inplace=True)
        self.y.reset_index(drop=True, inplace=True)
        
        print("Cantidad de datos nuevos ", self.X.shape, self.y.shape)
        sys.stdout.flush()   
        return self.X, self.y
    
    # Función para remover datos atípicos con el z_core ajustado.
    def remove_outliers_adjusted_zscore(self):
        print("Eliminar valores atípicos con z-score ajustado")
        sys.stdout.flush()

        # Calcular la mediana y MAD (desviación absoluta mediana) para las columnas numéricas
        median_vals = np.median(self.X[self.numeric_columns], axis=0)
        mad_vals = np.median(np.abs(self.X[self.numeric_columns] - median_vals), axis=0)

        # Evitar divisiones por cero
        mad_vals[mad_vals == 0] = 1e-6

        # Calcular el z-score ajustado
        z_scores_adjusted = 0.6745 * (self.X[self.numeric_columns] - median_vals) / mad_vals

        # Identificar filas con valores atípicos (usando el umbral de outlier)
        outlier_rows = (np.abs(z_scores_adjusted) > self.threshold_outlier).any(axis=1)

        # Eliminar filas con valores atípicos
        self.X = self.X[~outlier_rows]
        self.y = self.y[~outlier_rows]

        # Reiniciar los índices
        self.X.reset_index(drop=True, inplace=True)
        self.y.reset_index(drop=True, inplace=True)

        print("Cantidad de datos nuevos ", self.X.shape, self.y.shape)
        sys.stdout.flush()

        return self.X, self.y
   
    # Función para reducción de dimensionalidad
    def apply_pca(self):
        print("Aplicando PCA: ")
        sys.stdout.flush()

        # Validar que sea un problema de regresión o clasificación
        if self.model_type != 'Regression' and self.model_type != 'Classification':
            raise ValueError("PCA solo puede ser aplicado en problemas de regresión o clasificación.")

        # Verificar si se definió un número de componentes o porcentaje de varianza
        if 'pca_n_components' in self.config:
            n_components = self.config.get('pca_n_components')

            # Si es un valor entre 0 y 1, se interpreta como varianza explicada
            if 0 < n_components < 1:
                print(f"Aplicando PCA para explicar al menos el {n_components*100}% de la varianza.")
                sys.stdout.flush()
                self.pca = PCA(n_components=n_components)

            # Si es mayor que 1, se asume que es el número de componentes
            elif n_components >= 1:
                print(f"Aplicando PCA para reducir a {n_components} componentes.")
                sys.stdout.flush()
                self.pca = PCA(n_components=int(n_components))
            
            else:
                raise ValueError("El valor de 'pca_n_components' debe ser un número positivo mayor que 0.")
        else:
            raise ValueError("Por favor, defina 'pca_n_components' en el archivo de configuración.")
        
        # Ajustar PCA al conjunto de datos
        self.pca.fit(self.X)

        # Transformar los datos
        self.X = self.pca.transform(self.X)

        # Guardar el objeto PCA en los transformadores
        self.transformers['pca'] = self.pca

        print("PCA aplicado con éxito. Nueva forma de los datos: ", self.X.shape)
        sys.stdout.flush()

        return self.X

    # Función para entrenar los transformadores: Imputar, Escalar, Codificar
    def fit(self):
        print("---------------------------------------------------")
        sys.stdout.flush()
        print("------------ Creando transformadores  -------------")
        sys.stdout.flush()
        print("---------------------------------------------------")
        sys.stdout.flush()#

        print("Imputar datos numéricos.")
        sys.stdout.flush()
        numeric_data = self.X[self.numeric_columns]
        # Verificar si hay datos faltantes en las variables numéricas
        if numeric_data.isnull().any().any():
            # Verificar si el porcentaje de datos faltantes es menor que el umbral
            if numeric_data.isnull().mean().mean() < self.missing_threshold:
                if self.numeric_imputer != "knn":
                    # Ajustar el imputador a todas las variables numéricas
                    numeric_simple_imputer = SimpleImputer(strategy = self.numeric_imputer)
                    numeric_simple_imputer.fit(numeric_data)
                    # Guardar el imputador en el diccionario de transformadores
                    self.transformers['numeric_imputer'] = numeric_simple_imputer
                else: 
                    knn_imputer_numeric = KNNImputer(n_neighbors = self.imputer_n_neighbors)
                    knn_imputer_numeric.fit(numeric_data)
                    # Guardar el imputador en el diccionario de transformadores
                    self.transformers['numeric_imputer'] = knn_imputer_numeric
            else:
                print(f"Al menos una de las columnas numéricas tiene un porcentaje de datos faltantes mayor al {self.missing_threshold * 100}%")
            
        else:
            print("No hay datos faltantes en las columnas numéricas, no se requiere imputación.")

        print("Imputar datos categóricos.")
        sys.stdout.flush()
        categorical_data = self.X[self.categorical_columns]
        # Verificar si hay datos faltantes en las variables categóricas
        if categorical_data.isnull().any().any():
            # Verificar si el porcentaje de datos faltantes es menor que el umbral
            if categorical_data.isnull().mean().mean() < self.missing_threshold:
                if self.categorical_imputer != "knn":
                    # Ajustar el imputador a todas las variables categóricas
                    categorical_simple_imputer = SimpleImputer(strategy = self.categorical_imputer)
                    categorical_simple_imputer.fit(categorical_data)
                    # Guardar el imputador en el diccionario de transformadores
                    self.transformers['categorical_imputer'] = categorical_simple_imputer
                else:
                    # Codificar variables categóricas con OrdinalEncoder
                    ordinal_encoder = OrdinalEncoder()
                    X_categorical_encoded = ordinal_encoder.fit_transform(categorical_data)

                    # Crear el imputador para las columnas categóricas
                    knn_imputer_categorical = KNNImputer(n_neighbors = self.imputer_n_neighbors)
                    knn_imputer_categorical.fit(X_categorical_encoded)
                    # Ajustar el imputador y transformar los datos
                    X_categorical_imputed = knn_imputer_categorical.fit_transform(X_categorical_encoded)

                    # Volver a decodificar las variables imputadas
                    X_categorical_imputed = ordinal_encoder.inverse_transform(X_categorical_imputed)

                    # Guardar los valores imputados en el DataFrame original
                    self.X[self.categorical_columns] = X_categorical_imputed

                    # Guardar el imputador en el diccionario de transformadores
                    self.transformers['categorical_imputer'] = knn_imputer_categorical
            else:
                print(f"Al menos una de las columnas categóricas tiene un porcentaje de datos faltantes mayor al {self.missing_threshold * 100}%")
        else:
            print("No hay datos faltantes en las columnas categóricas, no se requiere imputación.")

        print("Escalar datos numéricas.")
        sys.stdout.flush()
        # Escalar variables numéricas
        self.scaler_X.fit(self.X[self.numeric_columns])
        self.transformers['scaler_X'] = self.scaler_X

        # Si el modelo es de regresión también se escala la variable objetivo
        if self.model_type == 'Regression':
            # Convertir la variable objetivo a un array de numpy
            self.y = np.array(self.y).reshape(-1, 1)
            
            if self.scaling_method_target == 'standard':
                self.scaler_y.fit(self.y)
                # Guardar en los transformadores la información del escalador
                self.transformers['scaler_y'] = {'method': 'standard', 'scaler': self.scaler_y}       
            elif self.scaling_method_target == 'minmax':
                self.scaler_y.fit(self.y)
                # Guardar en los transformadores la información del escalador
                self.transformers['scaler_y'] = {'method': 'minmax', 'scaler': self.scaler_y}
            elif self.scaling_method_target == 'log':
                # Guardar en los transformadores la información de la transformación
                self.transformers['scaler_y'] = {'method': 'log'}
            
            elif self.scaling_method_target == 'sqrt':
                # Guardar en los transformadores la información de la transformación
                self.transformers['scaler_y'] = {'method': 'sqrt'}
            
            elif self.scaling_method_target == 'cbrt':
                # Guardar en los transformadores la información de la transformación
                self.transformers['scaler_y'] = {'method': 'cbrt'}
            
            else:
                raise ValueError(f"Método de escalado no reconocido: {self.scaling_method_target}")

            print("Escalado de la variable objetivo completado.")
            sys.stdout.flush()

        if self.model_type == 'Classification':
            self.label_encoder.fit_transform(self.y)
            self.transformers['label_encoder']  = self.label_encoder

        print("Codificar datos categóricos.")
        sys.stdout.flush()
        if len(self.categorical_columns) > 0:
            categorical_data = self.X[self.categorical_columns]
            # Ajustar el codificador OneHotEncoder
            self.one_hot_encoder.fit(categorical_data)
            # Guardar el codificador en los transformadores
            self.transformers['one_hot_encoder'] = self.one_hot_encoder
        else:
            print("No hay columnas categóricas para codificar.")
        
    # Función para aplicar las transformaciones a los datos.
    def transform(self):
        print("---------------------------------------------------")
        sys.stdout.flush()
        print("---------- Aplicando transformadores  -------------")
        sys.stdout.flush()
        print("---------------------------------------------------")
        sys.stdout.flush()

        print("Imputar datos nulos.")
        sys.stdout.flush()
        # Imputar datos nulos en variables numéricas
        if 'numeric_imputer' in self.transformers:
            self.X[self.numeric_columns] = self.transformers['numeric_imputer'].transform(self.X[self.numeric_columns])
   
        # Imputar datos nulos en variables categóricas
        if 'categorical_imputer' in self.transformers:
            if self.categorical_imputer != "knn":
                self.X[self.categorical_columns] = self.transformers['categorical_imputer'].transform(self.X[self.categorical_columns])
            else:
                # Codificar variables categóricas con OrdinalEncoder
                categorical_data = self.X[self.categorical_columns]
                ordinal_encoder = OrdinalEncoder()
                X_categorical_encoded = ordinal_encoder.fit_transform(categorical_data)

                # Ajustar el imputador y transformar los datos
                X_categorical_imputed = self.transformers['categorical_imputer'].transform(X_categorical_encoded)

                # Volver a decodificar las variables imputadas
                X_categorical_imputed = ordinal_encoder.inverse_transform(X_categorical_imputed)

                # Guardar los valores imputados en el DataFrame original
                self.X[self.categorical_columns] = X_categorical_imputed

        print("Escalar datos numéricos.")
        sys.stdout.flush()
        self.X[self.numeric_columns] = self.transformers['scaler_X'].transform(self.X[self.numeric_columns])
        if self.model_type == 'Regression':
            print(f"Aplicando el método de escalado '{self.scaling_method_target}' a la variable objetivo.")
            sys.stdout.flush()
            if self.scaling_method_target == 'standard' or self.scaling_method_target == 'minmax':
                self.y = np.array(self.y).reshape(-1, 1)
                self.y = self.transformers['scaler_y']['scaler'].transform(self.y).ravel()
            elif self.scaling_method_target == 'log':
                self.y = np.log1p(self.y)
            elif self.scaling_method_target == 'sqrt':
                self.y = np.sqrt(self.y)
            elif self.scaling_method_target == 'cbrt':
                self.y = np.cbrt(self.y)
            else:
                raise ValueError(f"Método de escalado no reconocido: {self.scaling_method_target}")

            print("Escalado de la variable objetivo completado.")
            sys.stdout.flush()

        print("Codificar datos categóricos.")
        sys.stdout.flush()
        encoded_features = self.transformers['one_hot_encoder'].transform(self.X[self.categorical_columns])
        encoded_feature_names = self.one_hot_encoder.get_feature_names_out(input_features=self.categorical_columns)
        encoded_df = pd.DataFrame(encoded_features.toarray(), columns=encoded_feature_names)
        self.X.drop(columns=self.categorical_columns, inplace=True)
        self.X = pd.concat([self.X, encoded_df], axis=1)
  
        # Si el modelo es de clasificación realizar Balanceo de datos
        if self.model_type == "Classification":
            print("Codificación de variable a predecir.")
            sys.stdout.flush()
            self.y = pd.Series(self.transformers['label_encoder'].transform(self.y))
            # Mostramos las etiquetas codificadas
            print("Etiquetas codificadas:", self.y)

            # Mostramos el mapeo de etiquetas originales a códigos numéricos
            print("Mapeo de etiquetas originales a códigos numéricos:")
            for label, code in zip(self.transformers['label_encoder'].classes_, self.transformers['label_encoder'].transform(self.transformers['label_encoder'].classes_)):
                print(f"{label}: {code}")

            print("Balanceo de datos: ")
            sys.stdout.flush()
            print(f"Datos balanceados usando {self.balance_method} con {self.sampler}")
            sys.stdout.flush()   
            print(f"cantidad clases antes del balanceo: {self.y.value_counts()}")
            sys.stdout.flush()   
            X_resampled, y_resampled = self.sampler.fit_resample(self.X, self.y)
            print("y_resampled")
            sys.stdout.flush()
            self.X = X_resampled
            self.y = y_resampled  
            print(f"cantidad clases despues del balanceo: {self.y.value_counts()}")
            sys.stdout.flush()  
        
        return self.X

    # Función para seleccionar las caracteristicas mas representativas
    def select_features(self):
        print("Selección de características: ")
        sys.stdout.flush()
        # Ajustar el objeto SelectKBest al conjunto de datos
        n_features = int(self.X.shape[1]*self.k)
        print('Cantidad caracteristicas a seleccionar: ', n_features)
        sys.stdout.flush()
        print('Cantidad caracteristicas inicial: ', self.X.shape[1])
        sys.stdout.flush()

        if self.feature_selector_method == "rfe":
            self.feature_selector.n_features_to_select = n_features
            self.feature_selector.fit(self.X, self.y)
            selected_features = self.X.iloc[:, self.feature_selector.support_]

        elif self.feature_selector_method == "rfecv":
            self.feature_selector.fit(self.X, self.y)
            selected_features = self.X.iloc[:, self.feature_selector.support_]
        else:
            self.feature_selector.k = n_features
            self.feature_selector.fit(self.X, self.y)
            selected_features = self.X.iloc[:, self.feature_selector.get_support()]


        # Obtener las características más representativas
        self.X = selected_features

        print('Caracteristicas seleccionadas: ',self.X.columns)
        sys.stdout.flush() 

        # Guargar caracteristicas seleccionadas en los transformadores
        self.transformers['feature_selector'] = self.X.columns
        return self.X
    
    # Función para obtener 'y' varialbe objetivo y 'X' variables predictoras
    def get_processed_dataframe(self):
        return self.X , self.y

    # Función para guardar los transformadores.
    def save_transformers(self, filename):
        print("Guardando transformadores: ")
        sys.stdout.flush() 

        try:
            # Guarda el diccionario en un archivo usando joblib
            joblib.dump(self.transformers, filename)
            print(f"Las transformaciones se guardaron en '{filename}'.")
        except Exception as e:
            print(f"Error al guardar las transformaciones: {e}")

    # Función para cargar los datos y hacer depuración. 
    def load_dataset_prediction(self, path_file, target):
        print(path_file)
        print("---------------------------------------------------")
        sys.stdout.flush()
        print("--------------- Carga de datos -------------------")
        sys.stdout.flush()
        print("---------------------------------------------------")
        sys.stdout.flush()
        try:
            df = pd.read_csv(path_file)
            # Asignación de 'y' varialbe objetivo y 'X' variables predictora 
            y = df[target]
            X = df.drop(columns=[target])

            return X, y

        except Exception as e:
            print("Error cargando el dataset:", e)

    # Función para cargar los transformadores.
    def load_transformers(self, filename):
        print("Cargando transformadores: ")
        sys.stdout.flush()

        try:
            # Carga el diccionario del archivo usando joblib
            transformers = joblib.load(filename)
            print(f"Las transformaciones se cargaron desde '{filename}'.")
            return transformers
        except Exception as e:
            print(f"Error al cargar las transformacioens: {e}")

    # Función para aplicar los transformadores a los datos a predecir
    def apply_transformers(self, transformers, X, y):
        print("Aplicando transformadores: ")
        sys.stdout.flush()
        # Obtener columnas numericas y categoricas.
        numeric_columns = X.select_dtypes(include=['number']).columns
        categorical_columns = X.select_dtypes(include=['object', 'category']).columns

        # Revisar los transformadores, de acuerdo a las llaves aplicar sobre los datos
        for name, transformer in transformers.items():
            if 'numeric_imputer' in name:
                X[numeric_columns] = transformers['numeric_imputer'].transform(X[numeric_columns])
            elif 'categorical_imputer' in name:
               X[categorical_columns] = transformers['categorical_imputer'].transform(X[categorical_columns])
            elif name == 'scaler_X':
                X[numeric_columns] = transformer.transform(X[numeric_columns])
            elif name == 'one_hot_encoder':
                encoded_feature_names = []
                for i, column in enumerate(categorical_columns):
                    encoded_feature_names.extend([f"{column}_{category}" for category in transformer.categories_[i]])

                encoded_features = transformer.transform(X[categorical_columns])
                encoded_df = pd.DataFrame(encoded_features.toarray(), columns=encoded_feature_names)
                X.drop(columns=categorical_columns, inplace=True)
                X = pd.concat([X, encoded_df], axis=1)
            elif name == 'feature_selector':
                X = X[transformers['feature_selector']]

        return X, y