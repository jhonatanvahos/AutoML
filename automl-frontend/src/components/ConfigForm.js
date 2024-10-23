import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom'; // Para redirigir a la página de resultados
import { updateConfig, trainModels } from '../services/api'; // trainModels es la llamada para entrenar
import './ConfigForm.css';

function ConfigForm() {
  const [config, setConfig] = useState({
    split: 0.1,
    missing_threshold: 0.1,
    numeric_imputer: "mean",
    categorical_imputer: "most_frequent",
    variable_imputer: "KNNImputer",
    imputer_n_neighbors: 5,
    scaling_method_features: "standard",
    scaling_method_target: "standard",
    threshold_outlier: 4,
    balance_method: "over_sampling",
    select_sampler : "SMOTE",
    balance_threshold: 0.7,
    k_features: 0.5,
    feature_selector_method : "select_k_best",
    pca_n_components : 0.95,
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
      gradient_boosting: false,
      lightGBM: false
    },
    models_classification: {
      logisticRegression: false,
      random_forest: false,
      SVM: false,
      KNN: false,
      GaussianNB: false,
      MultinomialNB: false,
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
        max_depth: [3, 5, 7]},
      lightGBM:{
        n_estimators: [10,50,200,500],
        max_depth: [3, 5, 9],
        learning_rate: [0.001, 0.01, 0.1],
        num_leaves: [5, 10, 15]}
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
      MultinomialNB:{},
      BernoulliNB:{}
    },
    advanced_options: false
  });

  const [loading, setLoading] = useState(false); // Estado para el loading
  const navigate = useNavigate(); // Para redireccionar

  const handleChange = (e) => {
    const { name, value, type, checked } = e.target;
    setConfig({
      ...config,
      [name]: type === "checkbox" ? checked : value,
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
    setLoading(true); // Mostrar el loading al empezar

    try {
      await updateConfig(config); // Actualizar la configuración
      const result = await trainModels(); // Llamar al backend para entrenar los modelos
      setLoading(false); // Dejar de mostrar el loading
      // Mostrar el resultado en la consola
      console.log("Resultado recibido del backend:", result);
      // Navegar hacia AutoML con métricas
      navigate('/automl', { state: { result } });

    } catch (error) {
      console.error("Error updating config:", error);
      setLoading(false); // Detener el loading si hay un error
    }
  };

  const toggleAdvancedOptions = () => {
    setConfig((prevConfig) => ({
      ...prevConfig,
      advancedOptions: !prevConfig.advancedOptions,
    }));
  };

  if (loading) {
    return <div className="loading">Training in progress, please wait...</div>; // Mostrar loading
  }

  return (
    <form onSubmit={handleSubmit}>
      <label>
        Split Ratio:
        <input
          type="number"
          step="0.01"
          name="split"
          value={config.split}
          onChange={handleChange}
        />
        <span className="info-icon" title="Proporción de los datos para el conjunto de entrenamiento.">ℹ️</span>
      </label>
      <br />

      <label>
        Missing Threshold:
        <input
          type="number"
          step="0.01"
          name="missing_threshold"
          value={config.missing_threshold}
          onChange={handleChange}
        />
      </label>
      <br />

      <label>
        Numeric Imputer:
        <select
          name="numeric_imputer"
          value={config.numeric_imputer}
          onChange={handleChange}
        >
          <option value="mean">Mean</option>
          <option value="median">Median</option>
          <option value="most_frequent">Most Frequent</option>
        </select>
      </label>
      <br />

      <label>
        Model Type:
        <select name="model_type" value={config.model_type} onChange={handleChange}>
          <option value="Classification">Classification</option>
          <option value="Regression">Regression</option>
        </select>
      </label>
      <br />

      {/* Mostrar solo los modelos de regresión si el tipo de modelo es 'regression' */}
      {config.model_type === "Regression" && (
        <fieldset>
          <legend>Models for Regression:</legend>
          {Object.keys(config.models_regression).map((model) => (
            <div key={model}>
              <label>
                <input
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
      )}
      <br />

      {/* Mostrar solo los modelos de clasificación si el tipo de modelo es 'classification' */}
      {config.model_type === "Classification" && (
        <fieldset>
          <legend>Models for Classification:</legend>
          {Object.keys(config.models_classification).map((model) => (
            <div key={model}>
              <label>
                <input
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
      )}
      <br />

      <button type="button" onClick={toggleAdvancedOptions}>
        {config.advanced_options ? "Hide" : "Show"} Advanced Options
      </button>
      {config.advanced_options && (
        <div>
          <label>
            Random State:
            <input
              type="number"
              name="random_state"
              value={config.random_state}
              onChange={handleChange}
            />
          </label>
          <br />

          <label>
            Scoring Classification:
            <input
              type="text"
              name="scoring_classification"
              value={config.scoring_classification}
              onChange={handleChange}
            />
          </label>
          <br />

          <label>
            Cross Validation Folds (CV):
            <input
              type="number"
              name="cv"
              value={config.cv}
              onChange={handleChange}
            />
          </label>
          <br />
        </div>
      )}

      <button type="submit">Save Config and Train Models</button>
    </form>
  );
}

export default ConfigForm;
