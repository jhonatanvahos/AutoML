import axios from 'axios';

const API_URL = 'http://localhost:8000';  // Endpoint del backend

// Enviar los datos del config Home al backend
export const saveConfig = async (configData) => {
  const response = await axios.post(`${API_URL}/save-config`, configData);
  return response.data;
};

// Enviar los datos del config al backend
export const updateConfig = async (config) => {
  const response = await axios.post(`${API_URL}/update-config`, config);
  return response.data;
};

// Enviar el dataset al backend
export const uploadDataset = async (formData) => {
  const response = await axios.post(`${API_URL}/upload-dataset`, formData, {
    headers: {
      'Content-Type': 'multipart/form-data',
    },
  });
  return response.data;
};

// Función para obtener los datos desde el back y poderlos visualizar
export const fetchDatasetPreview = async () => {
  try {
    const response = await axios.get(`${API_URL}/preview-dataset`, {});
    return response.data;
  } catch (error) {
    console.error('Error fetching dataset preview:', error);
    throw error;
  }
};

// Entrenar modelos 
export const trainModels = async () => {
  try {
    const response = await axios.post(`${API_URL}/train`);
    return response.data;
  } catch (error) {
    throw new Error('Error training model:', error);
  }
};

// Guardar la selección del modelo para predicción
export const saveModelSelection = async (modelName) => {
  try {
    const response = await axios.post(`${API_URL}/save-model-selection`, { selectedModel: modelName });
    return response.data;
  } catch (error) {
    throw new Error('Error saving model selection:', error);
  }
};

// Obtener lista de proyectos entrenados
export const fetchProjects = async () => {
  try {
    const response = await axios.get(`${API_URL}/api/projects`);
    return response.data;
  } catch (error) {
    throw new Error('Error fetching projects:', error);
  }
};

// Realizar predicción
export const predictModels = async (project, file) => {
  try {
    const response = await axios.post(`${API_URL}/api/predict`, { project, file }); // Enviar proyecto y archivo al backend
    return response.data;
  } catch (error) {
    throw new Error('Error predicting model:', error);
  }
};