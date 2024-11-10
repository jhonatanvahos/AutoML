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
    <div className="container">
      <h1>Model Training Results</h1>
      <h2>Select a Model for Prediction:</h2>
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
        Save Selection
      </button>
      <Link to="/" className="button-home">Return to Home</Link>
    </div>
  );
}

export default AutoML;
