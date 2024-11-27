from pydantic import BaseModel
from typing import List, Dict, Union

# Clase base para la configuración inicial del proyecto
class ConfigDataHome(BaseModel):
    project_name: str
    target_column: str
    dataset_path: str

# Clase base común para los parámetros de modelos
class ModelParams(BaseModel):
    pass

#----------------------------------------------------------------------------------------------
#----------------------- Parámetros para cada modelo de regresión -----------------------------
#----------------------------------------------------------------------------------------------
class LinearRegressionParams(ModelParams):
    fit_intercept: List[bool]

class RidgeParams(ModelParams):
    alpha: List[float]

class RandomForestParams(ModelParams):
    n_estimators: List[int]
    max_depth: List[int]
    max_features: List[str]
    criterion: List[str]

class AdaBoostParams(ModelParams):
    n_estimators: List[int]
    learning_rate: List[float]

class GradientBoostingParams(ModelParams):
    n_estimators: List[int]
    learning_rate: List[float]
    max_depth: List[int]

class LightGBMParams(ModelParams):
    n_estimators: List[int]
    max_depth: List[int]
    learning_rate: List[float]
    num_leaves: List[int]

#----------------------------------------------------------------------------------------------
#----------------------- Parámetros para cada modelo de clasificacióna ------------------------
#----------------------------------------------------------------------------------------------
class LogisticRegressionParams(ModelParams):
    multi_class: List[str]
    solver: List[str]
    class_weight: List[str]
    max_iter: List[int]

class RandomForestClassifierParams(ModelParams):
    n_estimators: List[int]
    max_features: List[int]
    max_depth: List[int]
    criterion: List[str]

class SVMParams(ModelParams):
    kernel: List[str]
    C: List[float]
    gamma: List[Union[str, float]]
    degree: List[int]
    coef0: List[float]

class KNNParams(ModelParams):
    n_neighbors: List[int]
    weights: List[str]
    metric: List[str]
    p: List[int]

# Clase de configuración general
class ConfigData(BaseModel):
    split: float
    missing_threshold: float
    numeric_imputer: str
    categorical_imputer: str
    imputer_n_neighbors_n: int
    imputer_n_neighbors_c: int
    scaling_method_features: str
    scaling_method_target: str
    threshold_outlier: float
    balance_method: str
    select_sampler: str
    balance_threshold: float
    k_features: float
    feature_selector_method: str
    pca_n_components: float
    delete_columns: List[str]
    model_type: str
    function: str
    n_jobs: int
    cv: int
    scoring_regression: str
    scoring_classification: str
    random_state: int
    model_competition: str
    models_regression: Dict[str, bool]
    models_classification: Dict[str, bool]
    params_regression: Dict[str, Union[LinearRegressionParams, RidgeParams, RandomForestParams, AdaBoostParams, GradientBoostingParams]]
    params_classification: Dict[str, Union[LogisticRegressionParams, RandomForestClassifierParams, SVMParams, KNNParams, ModelParams, ModelParams, ModelParams]]
    advanced_options: bool

    class Config:
        # para evitar warning con palabras reservadas model_
        protected_namespaces = ()

# Clase para selección de modelo para predicción
class ModelSelection(BaseModel):
    selectedModel: str

# Clase para inicial el modulo Predict
class PredictRequest(BaseModel):
    project: str
    file: str