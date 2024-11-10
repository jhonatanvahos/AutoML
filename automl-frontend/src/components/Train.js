import React, { useState } from 'react';
import UploadDatasetForm from './UploadDatasetForm';
import { useNavigate } from 'react-router-dom';
import '../styles/Train.css';

function Train() {
  const navigate = useNavigate();
  const [projectName, setProjectName] = useState('');
  const [targetColumn, setTargetColumn] = useState('');
  const [isUploaded, setIsUploaded] = useState(false);
  const [datasetPath, setDatasetPath] = useState('');
  const [columns, setColumns] = useState([]); 

  const handleProjectNameChange = (e) => setProjectName(e.target.value);
  const handleTargetColumnChange = (e) => setTargetColumn(e.target.value);

  const handleFileUploadSuccess = (path, columns) => {
    setIsUploaded(true);
    setDatasetPath(path);
    setColumns(columns); 
  };

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
      console.log(data)
      setDatasetPath(data.dataset_path); // Actualiza el estado con la nueva ruta del dataset
      console.log(data.message);
    } catch (error) {
      console.error("Error saving config:", error);
      alert("There was an error saving the configuration.");
    }
  };

  const handleContinue = async () => {
    console.log(datasetPath)
    if (!projectName || !targetColumn || !datasetPath) {
      alert("Please fill in all the fields");
      return;
    }

    await saveConfig();
    // Crear una nueva lista de columnas excluyendo targetColumn
    const filteredColumns = columns.filter(column => column !== targetColumn);

    // Pasar projectName, filteredColumns y targetColumn a ConfigForm
    navigate('/ConfigForm', { state: {columns: filteredColumns } });
  };

  return (
    <div className="train-container">
      <h1>Project Setup</h1>
      
      <div className="form-group">
        <label>Project Name:</label>
        <input
          type="text"
          value={projectName}
          onChange={handleProjectNameChange}
        />
      </div>

      <UploadDatasetForm onSuccess={handleFileUploadSuccess} />

      <div className="form-group">
        <label>Target Column:</label>
        {isUploaded ? (
          <select value={targetColumn} onChange={handleTargetColumnChange}>
            <option value="">Select Target Column</option>
            {columns.map((column) => (
              <option key={column} value={column}>
                {column}
              </option>
            ))}
          </select>
        ) : (
          <p>Upload a dataset to select the target column</p>
        )}
      </div>

      {isUploaded && (
        <button onClick={handleContinue}>Continue to Configuration</button>
      )}
    </div>
  );
}

export default Train;