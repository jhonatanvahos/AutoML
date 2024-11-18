import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { predictModels, fetchProjects } from '../services/api';
import UploadDatasetForm from './UploadDatasetForm'; // Importamos el componente de subida de archivos
import '../styles/Predict.css';

function Predict() {
  const [projects, setProjects] = useState([]); // Lista de proyectos
  const [selectedProject, setSelectedProject] = useState(''); // Proyecto seleccionado
  const [selectedFile, setSelectedFile] = useState('predict'); // Archivo seleccionado
  const [loading, setLoading] = useState(false); // Estado de carga
  const [showUploadForm, setShowUploadForm] = useState(false); // Mostrar el formulario de carga
  const [uploadedFilePath, setUploadedFilePath] = useState(null); // Ruta del archivo
  const navigate = useNavigate();

  useEffect(() => {
    const loadProjects = async () => {
      try {
        const result = await fetchProjects();
        setProjects(result.projects);
      } catch (error) {
        console.error("Error fetching projects:", error);
      }
    };

    loadProjects();
  }, []);

  const handleProjectChange = (e) => {
    setSelectedProject(e.target.value);
  };

  const handleFileChange = (e) => {
    setSelectedFile(e.target.value);
    setShowUploadForm(e.target.value === "upload");
  };

  const handleSubmit = async (e) => {
    e.preventDefault();

    if (!selectedProject) {
      alert("Por favor, selecciona un proyecto.");
      return;
    }

    setLoading(true);

    try {
      let result;
      if (selectedFile === "predict") {
        // Realizar predicción con el archivo de testeo
        result = await predictModels(selectedProject, "predict");
      } else if (selectedFile === "upload" && uploadedFilePath) {
        // Realizar predicción con el archivo subido
        result = await predictModels(selectedProject, uploadedFilePath);
      } else {
        alert("Por favor, sube un archivo antes de continuar.");
        return;
      }
      navigate('/resultpredict', { state: { result } });
      alert("Predicción realizada con éxito.");
    } catch (error) {
      console.error("Error al realizar la predicción:", error);
      alert("Hubo un error durante la predicción.");
    } finally {
      setLoading(false);
    }
  };

  const handleFileUploadSuccess = (filePath, columns) => {
    alert(`Archivo subido exitosamente: ${filePath}`);
    setUploadedFilePath(filePath);
  };

  const handleFileUploadError = (error) => {
    alert(`Error al subir el archivo: ${error}`);
  };

  return (
    <div className="predict-container">
      <header className="header">
        <div className="logo-container">
          <img src="logo.png" alt="PredictLab Logo" className="logo" />
          <h1>PredictLab</h1>
        </div>
      </header>

      <h2 className="predict-title">Página de Predicción</h2>

      <form onSubmit={handleSubmit} className="predict-form">
        <div className="form-group">
          <label>Seleccionar Proyecto:</label>
          <select value={selectedProject} onChange={handleProjectChange} required>
            <option value="">--Seleccionar Proyecto--</option>
            {projects.map((project, index) => (
              <option key={index} value={project}>
                {project}
              </option>
            ))}
          </select>
        </div>

        <div className="form-group">
          <label>Seleccionar archivo para predicción:</label>
          <select value={selectedFile} onChange={handleFileChange} required>
            <option value="predict">Archivo de Testeo</option>
            <option value="upload">Subir archivo</option>
          </select>
        </div>

        <div className="form-group">
          <button type="submit" className="predict-button" disabled={loading}>
            {loading ? "Realizando predicción..." : "Realizar Predicción"}
          </button>
        </div>
      </form>

      {showUploadForm && (
        <div className="upload-section">
          <UploadDatasetForm
            onSuccess={handleFileUploadSuccess}
            onError={handleFileUploadError}
          />
        </div>
      )}

      <footer className="footer">
        <p>© 2024 PredictLab. Todos los derechos reservados.</p>
        <p>by: Jhonatan Stick Gomez Vahos</p>
        <p>Sebastian Saldarriaga Arias</p>
      </footer>
    </div>
  );
}

export default Predict;