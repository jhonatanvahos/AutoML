import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom'; 
import { predictModels, fetchProjects } from '../services/api';

function Predict() {
  const [projects, setProjects] = useState([]); // Para almacenar la lista de proyectos
  const [selectedProject, setSelectedProject] = useState(''); // Proyecto seleccionado
  const [selectedFile, setSelectedFile] = useState('predict'); // Archivo de predicción seleccionado
  const [loading, setLoading] = useState(false); // Estado de carga
  const navigate = useNavigate();

  // Cargar proyectos al cargar la página
  useEffect(() => {
    const loadProjects = async () => {
      try {
        const result = await fetchProjects();  // Llamar al backend para obtener proyectos
        setProjects(result.projects);
      } catch (error) {
        console.error("Error fetching projects:", error);
      }
    };
    
    loadProjects();
  }, []);

  // Manejar el cambio de proyecto seleccionado
  const handleProjectChange = (e) => {
    setSelectedProject(e.target.value);
  };

  // Manejar el envío del formulario
  const handleSubmit = async (e) => {
    e.preventDefault();

    if (!selectedProject) {
      alert("Por favor, selecciona un proyecto.");
      return;
    }

    setLoading(true);  // Activar el estado de carga

    try {
      const result = await predictModels(selectedProject, selectedFile);  // Llamar al backend para realizar la predicción
      console.log("Resultado recibido del backend:", result);
      // Navegar hacia AutoML con métricas
      navigate('/resultpredict', { state: { result } });
      alert("Predicción realizada con éxito.");
    } catch (error) {
      console.error("Error al realizar la predicción:", error);
      alert("Hubo un error durante la predicción.");
    } finally {
      setLoading(false);  // Desactivar el estado de carga
    }
  };

  return (
    <div>
      <h1>Página de Predicción</h1>
      
      <form onSubmit={handleSubmit}>
        <div>
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

        <div>
          <label>Seleccionar archivo para predicción:</label>
          <select value={selectedFile} onChange={(e) => setSelectedFile(e.target.value)} required>
            <option value="predict">Archivo de Testeo</option>
            {/* Aquí podrías agregar más opciones si quieres permitir cargar archivos externos */}
          </select>
        </div>

        <div>
          <button type="submit" disabled={loading}>
            {loading ? "Realizando predicción..." : "Realizar Predicción"}
          </button>
        </div>
      </form>
    </div>
  );
}

export default Predict;
