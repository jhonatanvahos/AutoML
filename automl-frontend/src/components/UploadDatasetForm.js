import React, { useState } from 'react';
import { uploadDataset } from '../services/api';

function UploadDatasetForm({ onSuccess }) {
  const [file, setFile] = useState(null);
  const [uploadMessage, setUploadMessage] = useState("");

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

    try {
      const response = await uploadDataset(formData);
      setUploadMessage(response.message);

      if (response.message === "File uploaded successfully") {
        // Pasamos el file_path al componente padre
        onSuccess(response.file_path);
      }
    } catch (error) {
      console.error("Error uploading file:", error);
      setUploadMessage("Error uploading file");
    }
  };

  return (
    <div>
      <form onSubmit={handleSubmit}>
        <label>
          Select dataset (CSV or XLSX):
          <input type="file" onChange={handleFileChange} accept=".csv,.xlsx" />
        </label>
        <button type="submit">Upload</button>
      </form>
      {uploadMessage && <p>{uploadMessage}</p>}
    </div>
  );
}

export default UploadDatasetForm;
