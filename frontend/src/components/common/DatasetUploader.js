import React, { useState } from 'react';
import { uploadDataset } from '../../services/api';
import styles from './DatasetUploader.module.css'

function DatasetUploader({ onSuccess, onError }) {
  const [file, setFile] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState('');
  const [success, setSuccess] = useState('');

  const handleFileChange = (e) => {
    setFile(e.target.files[0]);
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError('');
    setSuccess('');

    if (!file) {
      setError('Por favor seleccione un archivo primero!');
      return;
    }

    const formData = new FormData();
    formData.append("file", file);

    setIsLoading(true);

    try {
      const response = await uploadDataset(formData);

      if (response.message === "File uploaded successfully") {
        setSuccess('Archivo cargado exitosamente!');
        onSuccess(response.file_path, response.columns);
      }
    } catch (error) {
      console.error("Error cargando el archivo:", error);
      setError('Error cargando el archivo');
      onError('Error cargando el archivo');
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className={styles.uploadFormContainer}>
      <form onSubmit={handleSubmit}>
        <label>
          Seleccione los datos en formato (CSV o XLSX):
          <input
            type="file"
            onChange={handleFileChange}
            accept=".csv,.xlsx"
          />
        </label>
        <button type="submit" disabled={isLoading}>
          {isLoading ? "Cargando..." : "Cargar"}
        </button>
      </form>

      {error && <div className={styles.errorMessage}>{error}</div>}
      {success && <div className={styles.successMessage}>{success}</div>}
    </div>
  );
}

export default DatasetUploader;
