Este proyecto permite generar Modelos de Machine Learning supervisado : Regresión y Clasificación.

Instrucciones de uso :
    1. Crear un Entorno virtual: versión de python 3.9.6
        https://docs.python.org/es/3/tutorial/venv.html

    2. Instalar las dependencias del requirements.txt :
        Una vez activo el entorno virtual : 
        pip install -r requirements.txt

    3. Configuración del proyecto:
            "data" - Dirección del insumo a procesar
            "delete_columns"  - Columnas que el usuario identifique que se puedan eliminar.
            "split"  - Porcentaje de separación del dataset inicial para posteriormente hacer predicciones.
            "k_features" - Porcentaje de caractaristicas representativas a seleccionar para entrenar el modelo. 
            "target_column" - Variable objetivo.
            "project_name" - Nombre del proyecto para guardar e identificar las transformaciones / modelos / datos.
            "model_type" - Modelo a utilizar : Rregression o  Classification
            "function" - Función a utilizar : Entrenamiento para la generación del modelo y predicción para el consumo del modelo. "training","predict" 

            "missing_threshold" -- Porcentaje de datos que pueden ser nulos para realizar imputacion de datos. 
            "balance_method" -- Modo de balanceo de datos : over-sampling o under-sampling
            "threshold_outlier" -- Cantidad en desviació de estandar para identificar datos atipicos
            "lower_percentile"  -- Percentil mas bajo para datos atipicos
            "upper_percentile" -- Percentil mas alto para datos atipicos

            "n_jobs" -- Cantidad de procesamiento a utilizar de la maquina . -1 para tomar todos los recursos.
            "cv" -- Validacion cruzada. 
            "scoring_regression" -- Métrica a utilizar para seleccionar el mejor modelo en Modelos de Regresión 
            "scoring_classification" -- Métrica a utilizar para seleccionar el mejor modelo en Modelos de Clasificación.
            "random_state": -- Semilla para generar los modelos. 

            "models_regression" - Modelos de regresión que van a competir. Puede activar o desactivar con las banderas true o false
            "params_regression" - Hiperparametros de cada modelo. Puede ser modificados los rangos

            "models_classification" - Modelos de clasificacion que van a competir. Puede activar o desactivar con las banderas true o false.
            "params_classification" - Hiperparametros de cada modelo. Puede ser modificados los rangos.

    4. Ejecutar el archivo main.py.