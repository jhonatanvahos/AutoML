import React, { useState, useEffect } from 'react';
import UploadDatasetForm from '../common/DatasetUploader';
import DatasetVisualization from './DatasetPreview';
import { saveConfig } from '../../services/api'; 
import styles from './TrainProjectPage.module.css';

function Train() {
  const [projectName, setProjectName] = useState('');
  const [targetColumn, setTargetColumn] = useState('');
  const [isUploaded, setIsUploaded] = useState(false);
  const [datasetPath, setDatasetPath] = useState('');
  const [columns, setColumns] = useState([]);
  const [popupMessage, setPopupMessage] = useState(""); 
  const [showVisualization, setShowVisualization] = useState(false);

  const handleProjectNameChange = (e) => setProjectName(e.target.value);
  const handleTargetColumnChange = (e) => setTargetColumn(e.target.value);

  const handleFileUploadSuccess = (path, columns) => {
    setIsUploaded(true);
    setDatasetPath(path);
    setColumns(columns);
    setPopupMessage("Carga de datos exitosa!"); 
  };

  const handleFileUploadError = (errorMessage) => {
    setPopupMessage(errorMessage);
  };

  // Este efecto limpia el mensaje después de un tiempo
  useEffect(() => {
    if (popupMessage) {
      const timer = setTimeout(() => setPopupMessage(""), 3000); 
      return () => clearTimeout(timer); 
    }
  }, [popupMessage]);

  const handleContinue = async () => {
    if (!projectName || !targetColumn || !datasetPath) {
      alert("Please fill in all the fields");
      return;
    }

    try {
      // Llamar a la API para guardar la configuración utilizando la función saveConfig importada
      const response = await saveConfig({
        project_name: projectName,
        target_column: targetColumn,
        dataset_path: datasetPath,
      });

      setDatasetPath(response.dataset_path);
    } catch (error) {
      console.error("Error saving config:", error);
      alert("There was an error saving the configuration.");
    }

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
    <div className={styles.container}>
      <header className={styles.header}>
        <div className={styles.logoContainer}>
          <img src="/logo.png" alt="PredictLab Logo" className={styles.logo} />
          <h1>PredictLab</h1>
        </div>
      </header>

      <main className={styles.trainContent}>
        <h2 className={styles.title}>Configuración del Proyecto</h2>

        <div className={styles.formGroupHorizontal}>
          <label>Nombre del Proyecto:</label>
          <input
            type="text"
            value={projectName}
            onChange={handleProjectNameChange}
            placeholder="Ingresa el nombre del proyecto"
            className={styles.inputField}
          />
        </div>

        <div className={styles.uploadButtonContainer}>
          <UploadDatasetForm
            onSuccess={handleFileUploadSuccess}
            onError={handleFileUploadError}
          />
        </div>

        <div className={styles.formGroupHorizontal}>
          <label>Columna Objetivo:</label>
          {isUploaded ? (
            <select value={targetColumn} onChange={handleTargetColumnChange} className={styles.inputSelect}>
              <option value="">Selecciona la columna objetivo</option>
              {columns.map((column) => (
                <option key={column} value={column}>
                  {column}
                </option>
              ))}
            </select>
          ) : (
            <p className={styles.uploadMessage}>Sube un conjunto de datos para seleccionar la columna objetivo</p>
          )}
        </div>

        {isUploaded && (
          <button className={styles.continueButton} onClick={handleContinue}>
            Continuar
          </button>
        )}

        {popupMessage && (
          <div className={styles.popup}>
            {popupMessage}
          </div>
        )}
      </main>

      <footer className={styles.footer}>
        <p>© 2024 PredictLab. Todos los derechos reservados.</p>
        <p>by: Jhonatan Stick Gomez Vahos</p>
        <p>Sebastian Saldarriaga Arias</p>
      </footer>
    </div>
  );
}

export default Train;