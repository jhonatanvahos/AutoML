import React, { useState } from 'react';
import { uploadDataset } from '../services/api';

function UploadDatasetForm({ onSuccess, onError }) {
  const [file, setFile] = useState(null);
  const [isLoading, setIsLoading] = useState(false);

  const handleFileChange = (e) => {
    setFile(e.target.files[0]);
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!file) {
      alert("Please select a file first!");
      return;
    }

    const formData = new FormData();
    formData.append("file", file);

    setIsLoading(true); // Indicar que la carga está en progreso

    try {
      const response = await uploadDataset(formData);

      if (response.message === "File uploaded successfully") {
        // Pasar el file_path y las columnas a la función onSuccess
        onSuccess(response.file_path, response.columns);
      }
    } catch (error) {
      console.error("Error uploading file:", error);
      onError("Error uploading file"); // Pasar el error al componente padre
    } finally {
      setIsLoading(false); // Finaliza el estado de carga
    }
  };

  return (
    <div>
      <form onSubmit={handleSubmit}>
        <label>
          Select dataset (CSV or XLSX):
          <input type="file" onChange={handleFileChange} accept=".csv,.xlsx" />
        </label>
        <button type="submit" disabled={isLoading}>
          {isLoading ? "Uploading..." : "Upload"}
        </button>
      </form>
    </div>
  );
}

export default UploadDatasetForm;
