import pandas as pd
import numpy as np
from pathlib import Path
import joblib
import os
import logging
from scipy.stats import zscore
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MinMaxScaler, LabelEncoder, OrdinalEncoder
from sklearn.feature_selection import SelectKBest, f_regression, RFE, RFECV, mutual_info_classif, mutual_info_regression
from sklearn.linear_model import LinearRegression
from imblearn.over_sampling import SMOTE, ADASYN, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler, ClusterCentroids, TomekLinks
from imblearn.combine import SMOTEENN, SMOTETomek
from sklearn.decomposition import PCA

class DataPreprocessor:
    """
    Clase para preprocesamiento de datos con funcionalidades como carga, imputación, eliminación de atípicos y escalado.
    """

    def __init__(self, config):
        """
        Inicializa el preprocesador de datos con la configuración proporcionada.

        Args:
            config (dict): Configuración que incluye rutas, columnas objetivo y opciones de preprocesamiento.
        """
        self.config = config
        self.path_file = self.config.get('dataset_path')
        self.delete_columns = self.config.get('delete_columns', [])
        self.split = self.config.get('split', 0.2)
        self.target = self.config.get('target_column')
        self.model_type = self.config.get('model_type')
        self.threshold_outlier = self.config.get('threshold_outlier', 3)

        # Inicialización de variables para codificación
        self.label_encoder = LabelEncoder()
        self.one_hot_encoder = OneHotEncoder(handle_unknown='ignore')

        # Configuración de imputadores
        self.missing_threshold = self.config.get('missing_threshold', 0.1)
        self.numeric_imputer = self.config.get("numeric_imputer", SimpleImputer(strategy="mean"))
        self.categorical_imputer = self.config.get("categorical_imputer", SimpleImputer(strategy="most_frequent"))
        self.imputer_n_neighbors_n = self.config.get("imputer_n_neighbors_n", 5)
        self.imputer_n_neighbors_c = self.config.get("imputer_n_neighbors_c", 5)

        # Configuración de balanceo
        self.balance_method = self.config.get('balance_method')
        self.select_sampler = self.config.get('select_sampler')
        self._initialize_sampler()

        # Configuración de escaladores
        self.scaling_method_target = self.config.get('scaling_method_target')
        self.scaling_method_features = self.config.get('scaling_method_features')
        self._initialize_scalers()

        # Configuración de selección de características
        self.k = self.config.get('k_features', 10)
        self.feature_selector_method = self.config.get('feature_selector_method')
        self._initialize_feature_selector()

        # Variables adicionales
        self.transformers = {}
        self.seed = 11
        self.unprocessed_columns = []

    def _initialize_sampler(self):
        """
        Inicializa el método de balanceo según la configuración.
        """
        samplers = {
            'over_sampling': {'SMOTE': SMOTE(), 'ADASYN': ADASYN(), 'RandomOverSampler': RandomOverSampler()},
            'under_sampling': {'RandomUnderSampler': RandomUnderSampler(), 'ClusterCentroids': ClusterCentroids(),
                               'TomekLinks': TomekLinks()},
            'combine': {'SMOTEENN': SMOTEENN(), 'SMOTETomek': SMOTETomek()}
        }
        self.sampler = samplers.get(self.balance_method, {}).get(self.select_sampler, None)

    def _initialize_scalers(self):
        """
        Inicializa los escaladores para características y la variable objetivo.
        """
        scalers = {'standard': StandardScaler, 'minmax': MinMaxScaler}  # No instanciamos todavía

        # Crear nuevas instancias para cada uno
        self.scaler_X = scalers.get(self.scaling_method_features)()  # Instanciamos el escalador para X
        self.scaler_y = scalers.get(self.scaling_method_target)()     # Instanciamos el escalador para y

    def _initialize_feature_selector(self):
        """
        Inicializa el método de selección de características según la configuración.
        """
        if self.feature_selector_method == "select_k_best":
            self.feature_selector = SelectKBest(score_func=f_regression)
        elif self.feature_selector_method == "rfe":
            self.feature_selector = RFE(LinearRegression())
        elif self.feature_selector_method == "rfecv":
            self.feature_selector = RFECV(LinearRegression(), step=1, cv=5)
        elif self.feature_selector_method == "mutual_info_classif":
            self.feature_selector = SelectKBest(score_func=mutual_info_classif)
        elif self.feature_selector_method == "mutual_info_regression":
            self.feature_selector = SelectKBest(score_func=mutual_info_regression)
        else:
            raise ValueError(f"Método de selección de características no reconocido: {self.feature_selector_method}")

    def load_dataset(self):
        """
        Carga el dataset desde la ruta especificada en la configuración.
        """
        logging.info("Cargando dataset desde %s", self.path_file)
        try:
            file_extension = os.path.splitext(self.path_file)[1].lower()
            if file_extension == '.csv':
                separators = [",", ";", "|"]
                for sep in separators:
                    try:
                        temp_df = pd.read_csv(self.path_file, sep=sep, nrows=5)
                        if len(temp_df.columns) > 1:
                            self.df = pd.read_csv(self.path_file, sep=sep)
                            logging.info("Archivo CSV cargado correctamente con separador '%s'", sep)
                            break
                    except pd.errors.ParserError:
                        continue
                else:
                    raise ValueError("No se pudo determinar el separador del archivo CSV.")
            elif file_extension in ['.xls', '.xlsx']:
                self.df = pd.read_excel(self.path_file)
                logging.info("Archivo Excel cargado correctamente.")
            else:
                raise ValueError(f"Tipo de archivo no soportado: {file_extension}")

            # Limpieza inicial de columnas
            self.df.columns = self.df.columns.str.strip().str.replace(' ', '_')
            self.df.drop(columns=self.delete_columns, inplace=True)
            self.df = self.df.apply(lambda x: x.str.lower() if x.dtype == 'object' else x)

            # Eliminar columnas irrelevantes
            self.df.drop(columns=self.df.columns[self.df.nunique() == 1], inplace=True)
            self.df.drop(columns=self.df.columns[self.df.nunique() == len(self.df)], inplace=True)
            self.df.drop_duplicates(inplace=True)
            self.df.reset_index(drop=True, inplace=True)

            # Selección de variable objetivo
            if self.target not in self.df.columns:
                raise ValueError("Variable objetivo no encontrada en el dataset.")
            self.y = self.df[self.target]
            self.X = self.df.drop(columns=[self.target])

            logging.info("Columnas de entranamiento a guardar: ", list(self.X.columns))
            # Actualizando el diccionario self.config
            self.config["trained_features"] = list(self.X.columns)

            # Identificación de tipos de columnas
            self.numeric_columns = self.X.select_dtypes(include=['number']).columns
            self.categorical_columns = self.X.select_dtypes(include=['object', 'category']).columns

            logging.info("Dataset cargado y procesado. Dimensiones: %s", self.df.shape)
        except Exception as e:
            logging.error("Error al cargar el dataset: %s", e)
            raise

    def split_data_for_predictions(self, save_path):
        """
        Separa un porcentaje de los datos para realizar predicciones después de crear el modelo
        y guarda esos datos en un archivo CSV.

        :param save_path: Ruta donde se guardarán los datos para predicciones.
        """
        logging.info("Separando datos para predicciones...")

        # Seleccionar datos aleatorios
        np.random.seed(self.seed)
        num_rows_to_predict = int(len(self.df) * self.split)
        random_indices = np.random.choice(self.df.index, num_rows_to_predict, replace=False)
        prediction_data = self.df.loc[random_indices]

        # Eliminar los datos seleccionados del DataFrame original
        self.df.drop(random_indices, inplace=True)
        self.df.reset_index(drop=True, inplace=True)

        # Guardar los datos para predicciones en un nuevo archivo CSV
        try:
            prediction_data.to_csv(save_path, index=False)
            logging.info(f"Datos para predicciones guardados en '{save_path}'")
        except Exception as e:
            logging.error(f"Error al guardar los datos para predicciones: {e}")

    def remove_outliers_zscore(self):
        """
        Elimina los valores atípicos utilizando Z-score.

        :return: Los DataFrames `X` y `y` sin los valores atípicos.
        """
        logging.info("Eliminando valores atípicos usando Z-score...")

        # Calcular z-scores para las columnas numéricas
        z_scores = zscore(self.X[self.numeric_columns])

        # Identificar filas con valores atípicos
        outlier_rows = (np.abs(z_scores) > self.threshold_outlier).any(axis=1)

        # Eliminar filas con valores atípicos
        self.X = self.X[~outlier_rows]
        self.y = self.y[~outlier_rows]

        self.X.reset_index(drop=True, inplace=True)
        self.y.reset_index(drop=True, inplace=True)

        logging.info(f"Cantidad de datos después de eliminar atípicos: {self.X.shape}")
        logging.info("Finaliza eliminación de valores atípicos exitosamente!")

    def remove_outliers_adjusted_zscore(self):
        """
        Elimina los valores atípicos utilizando un Z-score ajustado basado en la mediana y MAD (desviación absoluta mediana).

        :return: Los DataFrames `X` y `y` sin los valores atípicos.
        """
        logging.info("Eliminando valores atípicos usando Z-score ajustado...")

        # Calcular la mediana y MAD para las columnas numéricas
        median_vals = np.median(self.X[self.numeric_columns], axis=0)
        mad_vals = np.median(np.abs(self.X[self.numeric_columns] - median_vals), axis=0)

        # Evitar divisiones por cero
        mad_vals[mad_vals == 0] = 1e-6

        # Calcular el z-score ajustado
        z_scores_adjusted = 0.6745 * (self.X[self.numeric_columns] - median_vals) / mad_vals

        # Identificar filas con valores atípicos
        outlier_rows = (np.abs(z_scores_adjusted) > self.threshold_outlier).any(axis=1)

        # Eliminar filas con valores atípicos
        self.X = self.X[~outlier_rows]
        self.y = self.y[~outlier_rows]

        # Reiniciar los índices
        self.X.reset_index(drop=True, inplace=True)
        self.y.reset_index(drop=True, inplace=True)

        logging.info(f"Cantidad de datos después de eliminar atípicos con z-score ajustado: {self.X.shape}")
        logging.info("Finaliza eliminación de valores atípicos exitosamente!")

    def apply_pca(self):
        """
        Aplica PCA (Análisis de Componentes Principales) para la reducción de dimensionalidad.

        :return: El DataFrame transformado por PCA.
        """
        logging.info("Inicia la reducción de dimensionalidad PCA...")

        # Validar que sea un problema de regresión o clasificación
        if self.model_type not in ['Regression', 'Classification']:
            raise ValueError("PCA solo puede ser aplicado en problemas de regresión o clasificación.")

        # Verificar si se definió un número de componentes o porcentaje de varianza
        if 'pca_n_components' in self.config:
            n_components = self.config.get('pca_n_components')

            # Si es un valor entre 0 y 1, se interpreta como varianza explicada
            if 0 < n_components < 1:
                logging.info(f"Aplicando PCA para explicar al menos el {n_components*100}% de la varianza.")
                self.pca = PCA(n_components=n_components)
            # Si es mayor que 1, se asume que es el número de componentes
            elif n_components >= 1:
                logging.info(f"Aplicando PCA para reducir a {n_components} componentes.")
                self.pca = PCA(n_components=int(n_components))
            else:
                raise ValueError("El valor de 'pca_n_components' debe ser un número positivo mayor que 0.")
        else:
            raise ValueError("Por favor, defina 'pca_n_components' en el archivo de configuración.")
        
        # Ajustar PCA al conjunto de datos
        self.pca.fit(self.df[self.numeric_columns])

        # Transformar los datos
        transformed_data = self.pca.transform(self.df[self.numeric_columns])

        # Guardar el objeto PCA en los transformadores
        self.transformers['pca'] = self.pca

        logging.info(f"Nueva forma de los datos: {transformed_data.shape}")
        logging.info("Finaliza la reducción de dimensionalidad PCA exitosamente!")

        return transformed_data

    def fit_transform(self):
        """
        Ajusta los transformadores (imputación, escalado y codificación) a los datos.

        Esta función realiza la imputación de datos numéricos y categóricos, escala las características y la variable
        objetivo, y codifica las variables categóricas.
        """
        logging.info("Inicia el entrenamiento y ajuste de transformadores...")

        # Imputación de datos numéricos y categóricos
        self._impute_numerical_data_fit()
        self._impute_categorical_data_fit()
        logging.info("Imputanción de datos realizada exitosamente!")

        # Escalado de datos numéricos 
        self._scale_numerical_data_fit()
        logging.info("Escalado de datos realizado exitosamente!")

        # Codificación de variables categóricas
        self._encode_categorical_data_fit()
        logging.info("Codificación de datos realizada exitosamente!")

        # Escalado o Encoder de variable objetivo
        if self.model_type == 'Regression':
            self._scale_target_variable_fit()
            logging.info("Escalado de variable objetivo realizada exitosamente!")
        else:
            self._encode_target_variable_fit()
            logging.info("Codificación de variable objetivo realizada existosamente!")

        logging.info("Finaliza exitosamente el entrenamiento y ajuste de transformadores!")

    def _impute_numerical_data_fit(self):
        """Imputa los valores faltantes en las columnas numéricas."""
        logging.info("Imputando datos numéricos...")
        numeric_data = self.X[self.numeric_columns]
        
        if numeric_data.isnull().any().any():
            if numeric_data.isnull().mean().mean() < self.missing_threshold:
                imputer = SimpleImputer(strategy=self.numeric_imputer) if self.numeric_imputer != "knn" else KNNImputer(n_neighbors=self.imputer_n_neighbors_n)
                imputer.fit(numeric_data)
                self.transformers['numeric_imputer'] = imputer
                logging.info("Imputación numérica realizada.")
            else:
                logging.warning(f"Al menos una columna numérica tiene más del {self.missing_threshold * 100}% de datos faltantes.")
        else:
            logging.info("No hay datos faltantes en las columnas numéricas.")

    def _impute_categorical_data_fit(self):
        """Imputa los valores faltantes en las columnas categóricas."""
        logging.info("Imputando datos categóricos...")
        categorical_data = self.X[self.categorical_columns]
        
        if categorical_data.isnull().any().any():
            if categorical_data.isnull().mean().mean() < self.missing_threshold:
                imputer = SimpleImputer(strategy=self.categorical_imputer) if self.categorical_imputer != "knn" else KNNImputer(n_neighbors=self.imputer_n_neighbors_c)
                imputer.fit(categorical_data)
                self.transformers['categorical_imputer'] = imputer
                logging.info("Imputación categórica realizada.")
            else:
                logging.warning(f"Al menos una columna categórica tiene más del {self.missing_threshold * 100}% de datos faltantes.")
        else:
            logging.info("No hay datos faltantes en las columnas categóricas.")

    def _scale_numerical_data_fit(self):
        """Escala los datos numéricos."""
        logging.info("Escalando datos numéricos...")
        self.scaler_X.fit(self.X[self.numeric_columns].to_numpy())
        self.transformers['scaler_X'] = self.scaler_X

    def _scale_target_variable_fit(self):
        """Escala la variable objetivo (solo para modelos de regresión)."""
        logging.info("Escalando variable objetivo...")
        self.y = np.array(self.y).reshape(-1, 1)
        if self.scaling_method_target == 'standard':
            self.scaler_y.fit(self.y)
            self.transformers['scaler_y'] = {'method': 'standard', 'scaler': self.scaler_y}
        elif self.scaling_method_target == 'minmax':
            self.scaler_y.fit(self.y)
            self.transformers['scaler_y'] = {'method': 'minmax', 'scaler': self.scaler_y}
        else:
            self.transformers['scaler_y'] = {'method': self.scaling_method_target}

    def _encode_categorical_data_fit(self):
        """Codifica las variables categóricas."""
        logging.info("Codificando datos categóricos...")
        if len(self.categorical_columns) > 0:
            categorical_data = self.X[self.categorical_columns]
            self.one_hot_encoder.fit(categorical_data)
            self.transformers['one_hot_encoder'] = self.one_hot_encoder
            logging.info("Codificación de variables categóricas realizada.")
        else:
            logging.info("No hay columnas categóricas para codificar.")
    
    def _encode_target_variable_fit(self):
        """Codifica la variable objetivo (solo para modelos de clasificacion)."""
        logging.info("Codificando variable objetivo...")
        self.label_encoder.fit(self.y)
        self.transformers['label_encoder_y']  = self.label_encoder
        
    def apply_transform(self):
        """
        Aplica las transformaciones (imputación, escalado y codificación) a los datos.
        """
        logging.info("Inicia la aplicacion de los transformadores...")
        # Imputación de datos nulos
        self._impute_missing_data()

        # Codificación de datos categóricos
        self._encode_categorical_data()

        # Codificación y balanceo de datos para clasificación
        if self.model_type == "Classification":
            self._encode_target_variable()
            self._balance_data()
        else :
            self._scale_target_variable()
        
        # Escalado de datos numéricos
        self._scale_numeric_data()
        
        logging.info("Finaliza exitosamente la aplicacion de transformadores!")
        return self.X

    def _impute_missing_data(self):
        """Imputa los datos nulos en variables numéricas y categóricas."""
        logging.info("Imputando datos nulos...")

        # Imputación de datos nulos en variables numéricas
        if 'numeric_imputer' in self.transformers:
            logging.info("Imputando datos nulos en variables numéricas.")
            self.X[self.numeric_columns] = self.transformers['numeric_imputer'].transform(self.X[self.numeric_columns])

        # Imputación de datos nulos en variables categóricas
        if 'categorical_imputer' in self.transformers:
            logging.info("Imputando datos nulos en variables categóricas.")
            if self.transformers['categorical_imputer'] != "knn":
                self.X[self.categorical_columns] = self.transformers['categorical_imputer'].transform(self.X[self.categorical_columns])
            else:
                self._impute_categorical_with_knn()

    def _impute_categorical_with_knn(self):
        """Imputa los datos categóricos con KNN."""
        logging.info("Imputando datos categóricos con KNN.")
        categorical_data = self.X[self.categorical_columns]
        ordinal_encoder = OrdinalEncoder()
        X_categorical_encoded = ordinal_encoder.fit_transform(categorical_data)

        # Ajustar el imputador y transformar los datos
        X_categorical_imputed = self.transformers['categorical_imputer'].transform(X_categorical_encoded)

        # Decodificar las variables imputadas
        X_categorical_imputed = ordinal_encoder.inverse_transform(X_categorical_imputed)

        # Actualizar los valores imputados en el DataFrame original
        self.X[self.categorical_columns] = X_categorical_imputed

    def _scale_numeric_data(self):
        """Escala las variables numéricas."""
        logging.info("Escalando datos numéricos.")
        self.X[self.numeric_columns] = self.transformers['scaler_X'].transform(self.X[self.numeric_columns].to_numpy())
        logging.info("Escalado de la datos numéricos completado.")

    def _scale_target_variable(self):
        """Escala la variable objetivo según el método especificado."""
        logging.info(f"Aplicando el método de escalado '{self.scaling_method_target}' a la variable objetivo.")
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

        logging.info("Escalado de la variable objetivo completado.")

    def _encode_categorical_data(self):
        """Codifica las variables categóricas con One Hot Encoding."""
        logging.info("Codificando datos categóricos.")
        encoded_features = self.transformers['one_hot_encoder'].transform(self.X[self.categorical_columns])
        encoded_feature_names = self.transformers['one_hot_encoder'].get_feature_names_out(input_features=self.categorical_columns)
        encoded_df = pd.DataFrame(encoded_features.toarray(), columns=encoded_feature_names)
        self.X.drop(columns=self.categorical_columns, inplace=True)
        self.X = pd.concat([self.X, encoded_df], axis=1)

    def _encode_target_variable(self):
        """Codifica la variable objetivo según el método especificado."""
        logging.info("Codificando la variable a predecir...")
        self.y = pd.Series(self.transformers['label_encoder_y'].transform(self.y))

        # Mostramos el mapeo de etiquetas originales a códigos numéricos
        logging.info("Mapeo de etiquetas originales a códigos numéricos:")
        for label, code in zip(self.transformers['label_encoder_y'].classes_, self.transformers['label_encoder_y'].transform(self.transformers['label_encoder_y'].classes_)):
           logging.info(f"{label}: {code}")
        
        logging.info("Codificacion de la variable objetivo completado.")

    def _balance_data(self):
        """Balancea los datos de la variable objetivo en clasificación."""
        logging.info("Balanceando datos...")
        logging.info(f"Datos balanceados usando {self.balance_method} con {self.sampler}")
        logging.info(f"Cantidad de clases antes del balanceo: {self.y.value_counts()}")

        X_resampled, y_resampled = self.sampler.fit_resample(self.X, self.y)
        self.X = X_resampled
        self.y = y_resampled

        logging.info(f"Cantidad de clases después del balanceo: {self.y.value_counts()}")

        logging.info("Balanceo de datos completado.")

    def select_features(self):
        """
        Selecciona las características más representativas utilizando el método seleccionado (RFE, RFECV, o SelectKBest).

        Se ajusta el objeto selector de características al conjunto de datos y se seleccionan las mejores características.
        Se actualiza el DataFrame `X` con las características seleccionadas.
        """

        logging.info("Iniciando la selección de características...")
        # Calcular la cantidad de características a seleccionar
        n_features = int(self.X.shape[1] * self.k)
        logging.info(f'Cantidad de características a seleccionar: {n_features}')
        logging.info(f'Cantidad de características iniciales: {self.X.shape[1]}')

        # Aplicar el selector de características según el método especificado
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

        # Actualizar el DataFrame `X` con las características seleccionadas
        self.X = selected_features
        logging.info(f'Características seleccionadas: {self.X.columns.tolist()}')

        # Guardar las características seleccionadas en los transformadores
        self.transformers['feature_selector'] = self.X.columns
        logging.info("Finaliza la selección de características exitosamente!")
        
        return self.X

    def get_processed_dataframe(self):
        """
        Obtiene el DataFrame procesado con las características seleccionadas (`X`) y la variable objetivo (`y`).

        Returns:
            X (DataFrame): Variables predictoras procesadas.
            y (Series): Variable objetivo procesada.
        """
        logging.info("Obteniendo el DataFrame procesado...")
        return self.X, self.y
    
    def save_transformers(self, filename):
        """
        Guarda los transformadores en un archivo usando `joblib`.

        Args:
            filename (str): Nombre del archivo donde se guardarán los transformadores.

        Raises:
            Exception: Si ocurre un error al guardar los transformadores.
        """
        logging.info("Guardando transformadores...")
        
        try:
            # Guarda el diccionario de transformadores en un archivo
            joblib.dump(self.transformers, filename)
            logging.info(f"Las transformaciones se guardaron en '{filename}'.")
        except Exception as e:
            logging.error(f"Error al guardar las transformaciones: {e}")
    
    def load_dataset_prediction(self, path_file):
        """
        Carga un dataset desde un archivo (CSV o Excel) y realiza un preprocesamiento básico.

        Args:
            path_file (str): Ruta del archivo a cargar.

        Returns:
            pd.DataFrame: DataFrame con los datos cargados y preprocesados.

        Raises:
            ValueError: Si el archivo tiene un formato no soportado o hay un error al cargarlo.
        """
        logging.info(f"Cargando archivo: {path_file}")
        
        try:
            # Obtener la extensión del archivo
            file_extension = os.path.splitext(path_file)[1].lower()
            if file_extension == '.csv':
                # Intentar con varios separadores comunes
                separators = [",", ";", "|"]
                for sep in separators:
                    try:
                        # Intentar leer el archivo con el separador
                        temp_df = pd.read_csv(path_file, sep=sep, nrows=5)
                        
                        # Validar que el archivo tiene más de una columna
                        if len(temp_df.columns) > 1:
                            df = pd.read_csv(path_file, sep=sep)
                            logging.info(f"Archivo CSV cargado correctamente con separador '{sep}'")
                            break
                        else:
                            logging.warning(f"Separador '{sep}' no parece ser el correcto. Intentando otro.")
                    except pd.errors.ParserError:
                        logging.warning(f"Separador '{sep}' no funcionó, probando otro.")
                else:
                    raise ValueError("Ninguno de los separadores funcionó para el archivo CSV.")
            
            elif file_extension in ['.xls', '.xlsx']:
                df = pd.read_excel(path_file)
                logging.info("Archivo Excel cargado correctamente.")
            else:
                raise ValueError(f"Tipo de archivo no soportado: {file_extension}")
            
        except Exception as e:
            logging.error(f"Error al cargar el archivo: {e}")
            raise
        
        # Estandarizar columnas categóricas a minúsculas
        logging.info("Estandarizando columnas categóricas a minúsculas")
        df = df.apply(lambda x: x.str.lower() if x.dtype == 'object' else x)

        return df
    
    def load_transformers(self, filename):
        """
        Carga los transformadores desde un archivo.

        Args:
            filename (str): Nombre del archivo que contiene los transformadores.

        Returns:
            dict: Diccionario con los transformadores cargados.

        Raises:
            FileNotFoundError: Si el archivo no existe.
            Exception: Si ocurre un error al cargar los transformadores.
        """
        logging.info(f"Cargando transformadores desde '{filename}'...")

        # Convertir a Path para facilitar el manejo de rutas
        filename = Path(filename)

        # Validar si el archivo existe y es un archivo
        if not filename.is_file():
            logging.error(f"El archivo '{filename}' no existe o no es un archivo válido.")
            raise FileNotFoundError(f"El archivo '{filename}' no existe o no es un archivo válido.")

        try:
            # Cargar los transformadores desde el archivo
            transformers = joblib.load(filename)
            logging.info(f"Las transformaciones se cargaron desde '{filename}'.")
            return transformers
        except Exception as e:
            logging.error(f"Error al cargar las transformaciones: {e}")
            raise
    
    def apply_transformers(self, transformers, X, y, trained_features):
        """
        Aplica los transformadores a los datos de entrada (`X`) asegurando que las columnas coincidan con las del entrenamiento.

        Args:
            transformers (dict): Diccionario de transformadores.
            X (pd.DataFrame): Datos de entrada a predecir.
            trained_features (list): Características que fueron usadas en el entrenamiento.

        Returns:
            pd.DataFrame: Datos de entrada transformados.

        Raises:
            ValueError: Si las columnas de entrada no coinciden con las características entrenadas.
        """
        logging.info("Aplicando transformadores a los datos de entrada...")

        # Asegurar que las columnas coincidan con las del entrenamiento
        missing_columns = [col for col in trained_features if col not in X.columns]
        extra_columns = [col for col in X.columns if col not in trained_features]
        
        # Eliminar columnas adicionales no usadas en el entrenamiento
        if extra_columns:
            logging.warning(f"Eliminando columnas no entrenadas: {extra_columns}")
            X = X.drop(columns=extra_columns)
        
        # Agregar columnas faltantes con valores nulos o cero
        if missing_columns:
            logging.warning(f"Agregando columnas faltantes con valores cero: {missing_columns}")
            for col in missing_columns:
                X[col] = 0  # Usa 0 como valor predeterminado; ajusta según las necesidades del modelo.
        
        # Obtener columnas numéricas y categóricas
        numeric_columns = X.select_dtypes(include=['number']).columns
        categorical_columns = X.select_dtypes(include=['object', 'category']).columns

        # Aplicar transformadores a los datos
        for name, transformer in transformers.items():
            if name == 'numeric_imputer':
                X[numeric_columns] = transformer.transform(X[numeric_columns])
            elif name == 'categorical_imputer':
                X[categorical_columns] = transformer.transform(X[categorical_columns])
            elif name == 'scaler_X':
                X[numeric_columns] = transformer.transform(X[numeric_columns])
            elif name == 'one_hot_encoder':
                encoded_feature_names = []
                for i, column in enumerate(categorical_columns):
                    encoded_feature_names.extend([f"{column}_{category}" for category in transformer.categories_[i]])

                encoded_features = transformer.transform(X[categorical_columns])
                encoded_df = pd.DataFrame(encoded_features.toarray(), columns=encoded_feature_names)
                
                # Reemplazar las columnas categóricas con las codificadas
                X.drop(columns=categorical_columns, inplace=True)
                X = pd.concat([X, encoded_df], axis=1)
            
            elif name == 'feature_selector':
                # Validar las columnas para el selector de características
                missing_selector_columns = [
                    col for col in transformers['feature_selector'] if col not in X.columns
                ]

                if missing_selector_columns:
                    logging.warning(f"Agregando columnas faltantes para el selector de características: {missing_selector_columns}")
                    for col in missing_selector_columns:
                        X[col] = 0  # Agrega las columnas faltantes con valores cero.

                logging.info(f"Columnas seleccionadas: {X[list(transformers['feature_selector'])].columns}")
                # Reordenar columnas para que coincidan con el orden del selector
                X = X[list(transformers['feature_selector'])]
            
            elif name == 'label_encoder_y' and y is not None:
                y = pd.Series(transformer.transform(y)) 
        
        return X, y