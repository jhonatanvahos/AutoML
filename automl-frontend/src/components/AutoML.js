import React, { useState } from 'react';
import { useLocation, Link } from 'react-router-dom';
import '../styles/AutoML.css';
import { saveModelSelection } from '../services/api';

function AutoML() {
  const location = useLocation();
  const metrics = location.state;
  const [selectedModel, setSelectedModel] = useState('');

  if (!metrics) {
    return <div>No metrics available. Something went wrong.</div>;
  }

  const handleModelSelection = (modelName) => {
    setSelectedModel(modelName);
  };

  const handleSaveSelection = async () => {
    try {
      await saveModelSelection(selectedModel);
      alert(`Modelo seleccionado guardado: ${selectedModel}`);
    } catch (error) {
      alert('Error al guardar la selección del modelo');
    }
  };
  
  return (
<div className="automl-container">
  <header className="header">
    <div className="logo-container">
      <img src="logo.png" alt="PredictLab Logo" className="logo" />
      <h1>PredictLab</h1>
    </div>
  </header>

  <h2 className="automl-title">Resultado del entrenamiento</h2>

  <h2>Seleccione el modelo que desea guardar:</h2>
  <div className="model-selection">
    {Object.keys(metrics.result).map((modelName) => (
      <div key={modelName} className="model-option">
        <label>
          <input
            type="radio"
            value={modelName}
            checked={selectedModel === modelName}
            onChange={() => handleModelSelection(modelName)}
          />
          <span>{modelName}</span> - Score: {metrics.result[modelName][`score`]}
        </label>
      </div>
    ))}
  </div>

  <button className="button-home" onClick={handleSaveSelection}>
    Guardar Modelo Seleccionado
  </button>

  <Link to="/" className="button-home link">
    Volver a la página principal
  </Link>

  <footer className="footer">
    <p>© 2024 PredictLab. Todos los derechos reservados.</p>
    <p>by: Jhonatan Stick Gomez Vahos</p>
    <p>Sebastian Saldarriaga Arias</p>
  </footer>
</div>

  );
}

export default AutoML;
