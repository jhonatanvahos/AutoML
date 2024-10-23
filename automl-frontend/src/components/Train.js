import React, { useState } from 'react';
import UploadDatasetForm from './UploadDatasetForm';
import { useNavigate } from 'react-router-dom';
import './Train.css'; // Importa el archivo de estilos

function Train() {
  const navigate = useNavigate();
  const [projectName, setProjectName] = useState('');
  const [targetColumn, setTargetColumn] = useState('');
  const [isUploaded, setIsUploaded] = useState(false);
  const [datasetPath, setDatasetPath] = useState('');

  const handleProjectNameChange = (e) => setProjectName(e.target.value);
  const handleTargetColumnChange = (e) => setTargetColumn(e.target.value);
  
  const handleFileUploadSuccess = (path) => {
    setIsUploaded(true);
    setDatasetPath(path);
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
      console.log(data.message);
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
    navigate('/ConfigForm');
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
        <input
          type="text"
          value={targetColumn}
          onChange={handleTargetColumnChange}
        />
      </div>

      {isUploaded && (
        <button onClick={handleContinue}>Continue to Configuration</button>
      )}
    </div>
  );
}

export default Train;
