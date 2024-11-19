import React, { useState, useEffect } from 'react';
import UploadDatasetForm from './UploadDatasetForm';
import DatasetVisualization from './DatasetVisualization';
import { useNavigate } from 'react-router-dom';
import '../styles/Train.css';

function Train() {
  const navigate = useNavigate();
  const [projectName, setProjectName] = useState('');
  const [targetColumn, setTargetColumn] = useState('');
  const [isUploaded, setIsUploaded] = useState(false);
  const [datasetPath, setDatasetPath] = useState('');
  const [columns, setColumns] = useState([]);
  const [popupMessage, setPopupMessage] = useState(""); // Nuevo estado para manejar el popup
  const [showVisualization, setShowVisualization] = useState(false);

  const handleProjectNameChange = (e) => setProjectName(e.target.value);
  const handleTargetColumnChange = (e) => setTargetColumn(e.target.value);

  const handleFileUploadSuccess = (path, columns) => {
    setIsUploaded(true);
    setDatasetPath(path);
    setColumns(columns);
    setPopupMessage("Carga de datos exitosa!"); // Mostrar mensaje de éxito
  };

  const handleFileUploadError = (errorMessage) => {
    setPopupMessage(errorMessage); // Mostrar mensaje de error
  };

  // Este efecto limpia el mensaje después de un tiempo
  useEffect(() => {
    if (popupMessage) {
      const timer = setTimeout(() => setPopupMessage(""), 3000); // Desaparece en 3 segundos
      return () => clearTimeout(timer); // Limpia el temporizador cuando el mensaje cambia o el componente se desmonta
    }
  }, [popupMessage]);

  const saveConfig = async () => {
    try {
      const response = await fetch('http://localhost:8000/save-config', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          project_name: projectName,
          target_column: targetColumn,
          dataset_path: datasetPath,
        }),
      });

      if (!response.ok) {
        throw new Error('Failed to save config');
      }

      const data = await response.json();
      setDatasetPath(data.dataset_path);
    } catch (error) {
      console.error("Error saving config:", error);
      alert("There was an error saving the configuration.");
    }
  };

  const handleContinue = async () => {
    if (!projectName || !targetColumn || !datasetPath) {
      alert("Please fill in all the fields");
      return;
    }

    await saveConfig();

    setShowVisualization(true);

  };

  if (showVisualization) {
    return (
      <DatasetVisualization
        targetColumn={targetColumn}
        columns={columns}
        onContinue={() => setShowVisualization(false)}
      />
    );
  }

  return (
    <div className="train-container">
      <header className="header">
        <div className="logo-container">
          <img src="logo.png" alt="PredictLab Logo" className="logo" />
          <h1>PredictLab</h1>
        </div>
      </header>

      <main className="train-content">
        <h2 className="train-title">Configuración del Proyecto</h2>

        <div className="form-group-horizontal">
          <label>Nombre del Proyecto:</label>
          <input
            type="text"
            value={projectName}
            onChange={handleProjectNameChange}
            placeholder="Ingresa el nombre del proyecto"
            className="input-field"
          />
        </div>

        <div className="upload-button-container">
          <UploadDatasetForm
            onSuccess={handleFileUploadSuccess}
            onError={handleFileUploadError}
          />
        </div>

        <div className="form-group-horizontal">
          <label>Columna Objetivo:</label>
          {isUploaded ? (
            <select value={targetColumn} onChange={handleTargetColumnChange} className="input-select">
              <option value="">Selecciona la columna objetivo</option>
              {columns.map((column) => (
                <option key={column} value={column}>
                  {column}
                </option>
              ))}
            </select>
          ) : (
            <p className="upload-message">Sube un conjunto de datos para seleccionar la columna objetivo</p>
          )}
        </div>

        {isUploaded && (
          <button className="continue-button" onClick={handleContinue}>
            Continuar
          </button>
        )}

        {popupMessage && (
          <div className="popup">
            {popupMessage}
          </div>
        )}
      </main>

      <footer className="footer">
        <p>© 2024 PredictLab. Todos los derechos reservados.</p>
        <p>by: Jhonatan Stick Gomez Vahos</p>
        <p>Sebastian Saldarriaga Arias</p>
      </footer>
    </div>
  );
}

export default Train;
