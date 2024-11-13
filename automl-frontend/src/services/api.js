import axios from 'axios';

const API_URL = 'http://localhost:8000';  // Cambia esta URL al endpoint de tu backend

// Enviar los datos del config Home al backend
export const saveConfig = async (configData) => {
  const response = await axios.post('http://localhost:8000/save-config', configData);
  return response.data;
};

// Enviar los datos del config al backend
export const updateConfig = async (config) => {
  const response = await axios.post(`${API_URL}/update-config`, config);
  return response.data;
};

// Enviar el archivo al backend
export const uploadDataset = async (formData) => {
  const response = await axios.post(`${API_URL}/upload-dataset`, formData, {
    headers: {
      'Content-Type': 'multipart/form-data',
    },
  });
  return response.data;
};

// Entrenar modelos 
export const trainModels = async () => {
  try {
    const response = await axios.post(`${API_URL}/train`); // Llama al backend que inicia el entrenamiento
    return response.data; // Devuelve las métricas del modelo ganador
  } catch (error) {
    throw new Error('Error training model:', error);
  }
};

// Guardar la selección del modelo para predicción
export const saveModelSelection = async (modelName) => {
  try {
    const response = await axios.post(`${API_URL}/save-model-selection`, { selectedModel: modelName });
    return response.data; // Devuelve el mensaje de éxito del backend
  } catch (error) {
    throw new Error('Error saving model selection:', error);
  }
};

// Obtener lista de proyectos
export const fetchProjects = async () => {
  try {
    const response = await axios.get(`${API_URL}/api/projects`);
    return response.data; // Devuelve los proyectos
  } catch (error) {
    throw new Error('Error fetching projects:', error);
  }
};

// Realizar predicción
export const predictModels = async (project, file) => {
  try {
    const response = await axios.post(`${API_URL}/api/predict`, { project, file }); // Enviar proyecto y archivo al backend
    return response.data; // Devuelve el resultado de la predicción
  } catch (error) {
    throw new Error('Error predicting model:', error);
  }
};

// Función para obtener el estado de entrenamiento desde el backend
export const fetchTrainingStatus = async () => {
  try {
    const response = await axios.get(`${API_URL}/train/status`);
    return response.data;
  } catch (error) {
    console.error("Error fetching training status:", error);
    throw error;
  }
};
