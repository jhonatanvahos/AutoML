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

// Predecir con modelo 
export const predictModels = async () => {
  try {
    const response = await axios.post(`${API_URL}/predict`); // Llama al backend que inicia el entrenamiento
    return response.data; // Devuelve las predicciones
  } catch (error) {
    throw new Error('Error predict model:', error);
  }
};