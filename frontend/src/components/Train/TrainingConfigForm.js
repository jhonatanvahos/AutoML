import React, { useState } from 'react';
import { useNavigate , useLocation} from 'react-router-dom'; 
import { updateConfig, trainModels } from '../../services/api';
import LoadingScreen from './TrainingProgress';
import styles from  './TrainingConfigForm.module.css';

function ConfigForm() {
  const [config, setConfig] = useState({
    split: 0.1,
    missing_threshold: 0.1,
    numeric_imputer: "mean",
    categorical_imputer: "most_frequent",
    imputer_n_neighbors_n: 5,
    imputer_n_neighbors_c: 5,
    scaling_method_features: "standard",
    scaling_method_target: "standard",
    threshold_outlier: 4,
    balance_method: "over_sampling",
    select_sampler : "SMOTE",
    balance_threshold: 0.7,
    k_features: 0.5,
    feature_selector_method : "select_k_best",
    pca_n_components : 0.9,
    delete_columns: [],
    model_type: "Classification",
    function: "training",
    n_jobs: -1,
    cv: 5,
    scoring_regression: "neg_mean_squared_error",
    scoring_classification: "f1",
    random_state: 1234,
    model_competition : "Grid_Search",
    models_regression: {
      linearRegression: true,
      ridge: false,
      random_forest: false,
      ada_boost: false,
      gradient_boosting: false
    },
    models_classification: {
      logisticRegression: false,
      random_forest: false,
      SVM: false,
      KNN: false,
      GaussianNB: false,
      BernoulliNB: true
    },
    params_regression: {
      linearRegression: { fit_intercept: [true, false] },
      ridge: { alpha: [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0] },
      random_forest: { 
        n_estimators: [20, 50, 100, 200],
        max_depth: [5, 10, 20],
        max_features: ["None", "log2", "sqrt"],
        criterion: ["squared_error", "absolute_error", "friedman_mse", "poisson"]},
      ada_boost: {
        n_estimators : [10,30,50,70,100],
        learning_rate :[0.001,0.01,0.1]},
      gradient_boosting: {
        n_estimators: [10,30,50,70,100],
        learning_rate: [0.1, 0.01, 0.001],
        max_depth: [3, 5, 7]}
    },
    params_classification: {
      logisticRegression: { 
        multi_class: ["ovr", "multinomial"],
        solver: ["liblinear","lbfgs", "newton-cg", "newton-cholesky", "sag", "saga"],
        class_weight: ["balanced"],
        max_iter : [1000]},
      random_forest: {
        n_estimators : [20, 50, 100, 200, 300],
        max_features : [5, 7, 9],
        max_depth : [5, 10, 20, 30, 40, 50],
        criterion : ["gini", "entropy"]},
      SVM: {
        kernel : ["linear", "rbf", "poly"],
        C : [0.1, 1, 10], 
        gamma: ["scale", "auto", 1.0],
        degree: [3],
        coef0: [0.0]},
      KNN : {
        n_neighbors: [3, 5, 7, 9],
        weights: ["uniform", "distance"],
        metric: ["euclidean", "manhattan", "minkowski"],
        p: [1, 2]},
      GaussianNB:{},
      BernoulliNB:{}
    },
    advanced_options: false
  });
  const [loading, setLoading] = useState(false);
  const navigate = useNavigate();
  const location = useLocation();
  const { columns } = location.state;
  const [loadingStatus] = useState({
    message: 'Entrenamiento en proceso, por favor espere...',
    progress: 0,
  });
  const handleChange = (e) => {
    const { name, value, type, checked } = e.target;
  
    const parsedValue = type === "number" ? parseFloat(value) : value;
  
    console.log(`Campo: ${name}, Valor: ${parsedValue}`); // Debug: verifica el valor
  
    setConfig((prevConfig) => ({
      ...prevConfig,
      [name]: type === "checkbox" ? checked : parsedValue,
    }));
  };  
  const handleColumnDeleteChange = (column) => {
    setConfig((prevConfig) => {
      const delete_columns = prevConfig.delete_columns.includes(column)
        ? prevConfig.delete_columns.filter(col => col !== column) // Remover la columna si ya estaba seleccionada
        : [...prevConfig.delete_columns, column]; // Agregar la columna a la lista de columnas a eliminar
      return {
        ...prevConfig,
        delete_columns,
      };
    });
  };
  const handleModelRegressionChange = (e) => {
    const { name, checked } = e.target;
    setConfig({
      ...config,
      models_regression: {
        ...config.models_regression,
        [name]: checked,
      },
    });
  };
  const handleModelClassificationChange = (e) => {
    const { name, checked } = e.target;
    setConfig({
      ...config,
      models_classification: {
        ...config.models_classification,
        [name]: checked,
      },
    });
  };
  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);

    try {
      await updateConfig(config); 
      const result = await trainModels(); // Llamar al backend para entrenar los modelos
      setLoading(false); 
      // Mostrar el resultado en la consola
      console.log("Resultado recibido del backend:", result);
      // Navegar hacia AutoML con métricas
      navigate('/automl', { state: { result } });

    } catch (error) {
      console.error("Error updating config:", error);
      setLoading(false);
    }
  };
  const toggleAdvancedOptions = () => {
    setConfig((prevConfig) => ({
      ...prevConfig,
      advanced_options: !prevConfig.advanced_options,
    }));
  };
  const handleParameterChange = (model, param, value) => {
    setConfig(prevConfig => ({
      ...prevConfig,
      params_classification: {
        ...prevConfig.params_classification,
        [model]: {
          ...prevConfig.params_classification[model],
          [param]: value
        }
      }
    }));
  };
  if (loading) {
    return <LoadingScreen message={loadingStatus.message} progress={loadingStatus.progress} />;
  }

  return (
  <div className={styles.container}>
    <header className={styles.header}>
      <div className={styles.logoContainer}>
        <img src="/logo.png" alt="PredictLab Logo" className={styles.logo} />
        <h1>PredictLab</h1>
      </div>
    </header>
    
    <h2 className={styles.title}>Parámetros para el entrenamiento</h2>
    <form className={styles.form} onSubmit={handleSubmit}>
      {/* Sección para seleccionar columnas a eliminar */}
      <fieldset className={styles.fieldset}>
        <legend className={styles.legend}>Seleccionar columnas para eliminar:</legend>
        {columns.map((column) => (
          <label className={styles.label} key={column}>
            <input
              className={styles.checkbox}
              type="checkbox"
              checked={config.delete_columns.includes(column)}
              onChange={() => handleColumnDeleteChange(column)}
            />
            {column}
          </label>
        ))}
      </fieldset>

      {/* Porcentaje de datos para testeo */}
      <label className={styles.label}>
        Porc. datos Testeo:
        <input
          className={styles.input}
          type="number"
          step="0.01"
          name="split"
          value={config.split}
          onChange={handleChange}
        />
        <span className={styles.info_icon} title="Proporción de los datos para el conjunto de prediccion (testeo). Se recomienda un valor menor a 0.2 (20%) de los dato. Valores de (0.0 a 1.0)">ℹ️</span>
      </label>
      <br />

      {/* Umbral de imputación de datos nulos */}
      <label className={styles.label}>
        Umbral de imputación:
        <input
          className={styles.input}
          type="number"
          step="0.01"
          name="missing_threshold"
          value={config.missing_threshold}
          onChange={handleChange}
        />
        <span className={styles.info_icon} title="Umbral de decision de datos que pueden ser nulos para realizar imputacion,Se recomienda un valor inferior al 0.3(30%). Valores de (0.0 a 1.0)">ℹ️</span>
      </label>
      <br />

      {/* Selección de imputador de variables numericas */}
      <label className={styles.label}>
        Imputador numérico:
        <select
          className={styles.select}
          name="numeric_imputer"
          value={config.numeric_imputer}
          onChange={handleChange}
        >
          <option value="mean">Mean</option>
          <option value="median">Median</option>
          <option value="most_frequent">Most Frequent</option>
          <option value="knn">KNN</option>
        </select>
        <span className={styles.info_icon} title="Estrategia con la que se van a reemplazar los datos numéricos nulos: mean, median, most_frequent, knn">ℹ️</span>
      </label>
      <br />

      {/* Mostrar el campo de vecinos solo si KNN está seleccionado */}
      {(config.numeric_imputer === 'knn') && (
        <div className={styles.form_group}>
          <label className={styles.label}>
            Number of Neighbors (for KNN):
            <input
              className={styles.input}
              type="number"
              name="imputer_n_neighbors_n"
              step = "1"
              value={config.imputer_n_neighbors_n}
              onChange={handleChange}
              min={1}
            />
          </label>
        </div>
      )}

       {/* Selección de imputador de variables categóricas */}
      <label className={styles.label}>
        Imputador Categórico:
        <select
          className={styles.select}
          name="categorical_imputer"
          value={config.categorical_imputer}
          onChange={handleChange}
        >
          <option value="most_frequent">Most Frequent</option>
          <option value="knn">KNN</option>
        </select>
        <span className={styles.info_icon} title="Estrategia con la que se van a reemplazar los datos categóricos nulos: most_frequent, knn">ℹ️</span>
      </label>
      <br />

      {/* Mostrar el campo de vecinos solo si KNN está seleccionado */}
      {(config.categorical_imputer === 'knn') && (
        <div className={styles.form_group}>
          <label className={styles.label}>
            Number of Neighbors (for KNN):
            <input
              className={styles.input}
              type="number"
              step = "1"
              name="imputer_n_neighbors_c"
              value={config.imputer_n_neighbors_c}
              onChange={handleChange}
              min={1}
            />
          </label>
        </div>
      )}

      {/* Selección de método de escalado para las caracteristicas*/}
      <label className={styles.label}>
        Método de escalado(Características):
        <select
          className={styles.select}
          name="scaling_method_features"
          value={config.scaling_method_features}
          onChange={handleChange}
        >
          <option value="standard">Estandar</option>
          <option value="minmax">MinimoMaximo</option>
        </select>
        <span className={styles.info_icon} title="Estrategia con la que se van a escalar los datos de las caracteristicas: standard, minmax">ℹ️</span>
      </label>
      <br />
      
      {/* Umbral para los datos atípicos */}
      <label className={styles.label}>
        Umbral de datos atípicos:
        <input
          className={styles.input}
          type="number"
          step="0.25"
          name="threshold_outlier"
          value={config.threshold_outlier}
          onChange={handleChange}
          min={1}
          max={5}
        />
        <span className={styles.info_icon} title="Umbral de decision para eliminar o descartar los datos atípicos, Se recomienda un valor entre 3 y 4. Valores de (1 a 5)">ℹ️</span>
      </label>
      <br />

      {/* Porcentaje para selección de caracteristicas */}
      <label className={styles.label}>
        Porc. selcción caracteristicas:
        <input
          className={styles.input}
          type="number"
          step="0.05"
          name="k_features"
          value={config.k_features}
          onChange={handleChange}
          min={0.1}
          max={1.0}
        />
        <span className={styles.info_icon} title="Porcentaje de selección de caracteristicas, este dependerá de que porcentaje desee conservar ejem: 0.5 (50% de las caracteristicas). Valores de (0.1 a 1.0)">ℹ️</span>
      </label>
      <br />

      {/* Selección de Método de selección de caracteristicas */}
      <label className={styles.label}>
        Método de selección de caracteristicas:
        <select
          className={styles.select}
          name="feature_selector_method"
          value={config.feature_selector_method}
          onChange={handleChange}
        >
          <option value="select_k_best">Select KBest</option>
          <option value="rfe">RFE</option>
          <option value="rfecv">RFECV</option>
          <option value="mutual_info_classif">Mutual info Classif</option> 
          <option value="mutual_info_regression">Mutual info Regression</option> 
        </select>
        <span className={styles.info_icon} title="Método de balanceo de las clases: over_sampling, under_sampling, combine">ℹ️</span>
      </label>
      <br />

      {/* Porcentaje para selección de caracteristicas */}
      <label className={styles.label}>
        Reducción de dimensionalidad - PCA:
        <input
          className={styles.input}
          type="number"
          step="0.025"
          name="pca_n_components"
          value={config.pca_n_components}
          onChange={handleChange}
          min={0.5}
        />
        <span className={styles.info_icon} title="Si se selecciona un valor entre 0.1 a 1.0 se tomará como el Porcentaje de la varianza que se requiere obtener, se recomienda un valor superior al 0.8(80%).
                                           Si el valor es superior a 1.0  se consideran la cantidad de caracteristicas representativas que quiere obtener - valores de 1.0 hasta x">ℹ️</span>
      </label>
      <br />

      <label className={styles.label}>
        Tipo de Modelo:
        <select
          className={styles.select}
          name="model_type"
          value={config.model_type}
          onChange={handleChange}
        >
          <option value="Classification">Classification</option>
          <option value="Regression">Regression</option>
        </select>
        <span className={styles.info_icon} title="Tipo de problema que se desea resolver - Regresión o Clasificación">ℹ️</span>
      </label>
      <br />

      {/* Mostrar solo los modelos de regresión si el tipo de modelo es 'regression' */}
      {config.model_type === "Regression" && (
      <>
        <label className={styles.label}>
          Método de escalado(Target):
          <select 
            className={styles.select}
            name="scaling_method_target"
            value={config.scaling_method_target}
            onChange={handleChange}
          >
            <option value="standard">Estandar</option>
            <option value="minmax">MinimoMaximo</option>
          </select>
          <span className={styles.info_icon} title="Estrategia con la que se van a escalar los datos de la variable objetivo: standard, minmax">ℹ️</span>
        </label>
        <br />

        <fieldset className={styles.fieldset}>
          <legend className={styles.legend}>Modelos de Regresión:</legend>
          {Object.keys(config.models_regression).map((model) => (
            <div key={model}>
              <label className={styles.label}>
                <input
                  className={styles.checkbox}
                  type="checkbox"
                  name={model}
                  checked={config.models_regression[model]}
                  onChange={handleModelRegressionChange}
                />
                {model}
              </label>
            </div>
          ))}
        </fieldset>
      </>
      )}
      <br />

      {/* Mostrar solo los modelos de clasificación si el tipo de modelo es 'classification' */}
      {config.model_type === "Classification" && (
      <>
          {/* Umbral para realizar balanceo de clases */}
          <label className={styles.label}>
          Umbral de balanceo:
          <input
            className={styles.input}
            type="number"
            step="0.05"
            name="balance_threshold"
            value={config.balance_threshold}
            onChange={handleChange}
            min={0.5}
            max={0.9}
          />
          <span className={styles.info_icon} title="Umbral de decision para realizar el balanceo de clases, Se recomienda un valor entre 0.55(55%) y 0.75(75%) de desbalanceo entre clases. Valores de (0.5 a 0.9)">ℹ️</span>
        </label>
        <br />

        {/* Selección de Método de balanceo */}
        <label className={styles.label}>
          Método de balanceo:
          <select
            className={styles.select}
            name="balance_method"
            value={config.balance_method}
            onChange={handleChange}
          >
            <option value="over_sampling">Over Sampling</option>
            <option value="under_sampling">Under Sampling</option>
            <option value="combine">Combine</option>
          </select>
          <span className={styles.info_icon} title="Método de balanceo de las clases: over_sampling, under_sampling, combine">ℹ️</span>
        </label>
        <br />

        {/* Opciones para Over Sampling */}
        {config.balance_method === "over_sampling" && (
          <div>
            <label className={styles.label}>
              Balanceador over_sampling:
              <select
                className={styles.select}
                name="select_sampler"
                value={config.select_sampler}
                onChange={handleChange}
              >
                <option value="SMOTE">SMOTE</option>
                <option value="ADASYN">ADASYN</option>
                <option value="RandomOverSampler">Random</option>
              </select>
            </label>
            <br />
          </div>
        )}

        {/* Opciones para Under Sampling */}
        {config.balance_method === "under_sampling" && (
          <div>
            <label className={styles.label}>
              Balanceador under_sampling:
              <select
                className={styles.select}
                name="select_sampler"
                value={config.select_sampler}
                onChange={handleChange}
              >
                <option value="RandomUnderSampler">Random</option>
                <option value="ClusterCentroids">Cluster Centroids</option>
                <option value="TomekLinks">Tomek Links</option>
              </select>
            </label>
            <br />
          </div>
        )}

        {/* Opciones para Combine */}
        {config.balance_method === "combine" && (
          <div>
            <label className={styles.label}>
              Balanceador combine:
              <select
                className={styles.select}
                name="select_sampler"
                value={config.select_sampler}
                onChange={handleChange}
              >
                <option value="SMOTEENN">SMOTEENN</option>
                <option value="SMOTETomek">SMOTETomek</option>
              </select>
            </label>
            <br />
          </div>
        )}
  
        <fieldset className={styles.fieldset}>
          <legend className={styles.legend}>Modelos de Clasificación:</legend>
          {Object.keys(config.models_classification).map((model) => (
            <div key={model}>
              <label className={styles.label}>
                <input
                  className={styles.checkbox}
                  type="checkbox"
                  name={model}
                  checked={config.models_classification[model]}
                  onChange={handleModelClassificationChange}
                />
                {model}
              </label>
            </div>
          ))}
        </fieldset>
      </>
      )}
      <br />
      
      <button className={styles.buttonOption} type="button" onClick={toggleAdvancedOptions}>
        {config.advanced_options ? "Ocultar" : "Mostrar"} Opciones avanzadas
      </button>
      {config.advanced_options && (
        <div>
          <label className={styles.label}>
            Random State:
            <input
              className={styles.input}
              type="number"
              name="random_state"
              value={config.random_state}
              onChange={handleChange}
              min={0}
            />
          <span className={styles.info_icon} title="Semilla para poder replicar los entrenamientos">ℹ️</span>
          </label>
          <br />

        {/* Mostrar los Score si el tipo de modelo es 'regression' */}
        {config.model_type === "Regression" && (
        <>
          <label className={styles.label}>
            Métrica regresión:
            <select
              className={styles.select}
              name="scoring_regression"
              value={config.scoring_regression}
              onChange={handleChange}
            >
              <option value="neg_mean_squared_error">Error cuadratico medio</option>
            </select>
            <span className={styles.info_icon} title="Métrica con la cuál se va seleccionar el mejor modelo: neg_mean_squared_error">ℹ️</span>
          </label>

          {Object.keys(config.models_regression).map((model) => (
            config.models_regression[model] && (
              <div key={model}>
                <h4>{model}</h4>
                {Object.keys(config.params_regression[model]).map((param) => (
                  <label className={styles.label} key={param}>
                    {param}:
                    <input
                      className={styles.input}
                      type="text"
                      value={config.params_regression[model][param]}
                      onChange={(e) => handleParameterChange(model, param, e.target.value)}
                    />
                  </label>
                ))}
              </div>
            )
          ))}

          </>
        )}
        <br />

        {/* Mostrar los Score si el tipo de modelo es 'Clasificacion' */}
        {config.model_type === "Classification" && (
        <>
          <label className={styles.label}>
            Métrica clasificación:
            <select
              className={styles.select}
              name="scoring_classification"
              value={config.scoring_classification}
              onChange={handleChange}
            >
              <option value="f1">F1</option>
            </select>
            <span className={styles.info_icon} title="Métrica con la cuál se va seleccionar el mejor modelo: F1">ℹ️</span>
          </label>

          {Object.keys(config.models_classification).map((model) => (
            config.models_classification[model] && (
              <div key={model}>
                <h4>{model}</h4>
                {Object.keys(config.params_classification[model]).map((param) => (
                  <label className={styles.label} key={param}>
                    {param}:
                    <input
                      className={styles.input}
                      type="text"
                      value={config.params_classification[model][param]}
                      onChange={(e) => handleParameterChange(model, param, e.target.value)}
                    />
                  </label>
                ))}
              </div>
            )
          ))}

          </>
        )}
        <br />
        
        <label className={styles.label}>
            Número de jobs:
            <input
              className={styles.input}
              type="number"
              name="n_jobs"
              value={config.n_jobs}
              onChange={handleChange}
            />
          <span className={styles.info_icon} title="numero de jobs dedicados al procesamiento, si se selecciona -1 toma todos los recursos del equipo.
                                             Se puede asignar 1,2,3,4 cores del equipo donde se ejecuta . Tener en cuenta que entre menor sean mayor tiempo de ejecución.">ℹ️</span>
          </label>
          <br />

        <label className={styles.label}>
            Validación cruzada (CV):
            <input
              className={styles.input}
              type="number"
              name="cv"
              value={config.cv}
              onChange={handleChange}
              min={1}
            />
          <span className={styles.info_icon} title="Numero de validaciones cruzadas que se desean realizar. Entre mayor sea el número mas iteraciones se tendrán e incrementará el tiempo de ejecución">ℹ️</span>
          </label>
          <br />
        </div>
      )}

      <button className={styles.continueButtonForm} type="submit">Entrenar modelos</button>
    </form>

    <footer className={styles.footer}>
      <p>© 2024 PredictLab. Todos los derechos reservados.</p>
      <p>by: Jhonatan Stick Gomez Vahos</p>
      <p>Sebastian Saldarriaga Arias</p>
    </footer>
  </div>
  );
}

export default ConfigForm;
