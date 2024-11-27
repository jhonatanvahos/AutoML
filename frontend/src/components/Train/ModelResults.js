import React, { useState } from 'react';
import { useLocation, Link } from 'react-router-dom';
import { saveModelSelection } from '../../services/api';
import styles from './ModelResults.module.css';

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
    <div className={styles.container}>
      {/* Header */}
      <header className={styles.header}>
          <div className={styles.logoContainer}>
          <img src="/logo.png" alt="PredictLab Logo" className={styles.logo} />
          <h1>PredictLab</h1>
          </div>
      </header>

      <h2 className={styles.title}>Resultado del entrenamiento</h2>

      <h2>Seleccione el modelo que desea guardar:</h2>
      <div className={styles.modelSelection}>
        {Object.keys(metrics.result).map((modelName) => (
          <div key={modelName} className={styles.modelOption}>
            <label>
              <input
                className={styles.input}
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

      <button className={styles.button} onClick={handleSaveSelection}>
        Guardar Modelo Seleccionado
      </button>

      <Link to="/" className={styles.button}>
        Volver a la página principal
      </Link>

      <footer className={styles.footer}>
        <p>© 2024 PredictLab. Todos los derechos reservados.</p>
        <p>by: Jhonatan Stick Gomez Vahos</p>
        <p>Sebastian Saldarriaga Arias</p>
      </footer>
    </div>

  );
}

export default AutoML;
