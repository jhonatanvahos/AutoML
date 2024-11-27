import axios from 'axios';
import MockAdapter from 'axios-mock-adapter';
import * as api from '../services/api'; 

// Crear un mock de axios
const mock = new MockAdapter(axios);

describe('API tests', () => {
  
  afterEach(() => {
    // Restablecer las respuestas de axios entre cada prueba
    mock.reset();
  });

  // Prueba para la función saveConfig
  it('should save config', async () => {
    const mockData = { message: 'Config saved' };
    mock.onPost('http://localhost:8000/save-config').reply(200, mockData);

    const response = await api.saveConfig({ someConfig: 'data' });
    expect(response).toEqual(mockData);
  });

  // Prueba para la función updateConfig
  it('should update config', async () => {
    const mockData = { message: 'Config updated' };
    mock.onPost('http://localhost:8000/update-config').reply(200, mockData);

    const response = await api.updateConfig({ someConfig: 'data' });
    expect(response).toEqual(mockData);
  });

  // Prueba para la función uploadDataset
  it('should upload dataset', async () => {
    const mockData = { message: 'Dataset uploaded' };
    mock.onPost('http://localhost:8000/upload-dataset').reply(200, mockData);

    const response = await api.uploadDataset(new FormData());
    expect(response).toEqual(mockData);
  });

  // Prueba para la función fetchDatasetPreview
  it('should fetch dataset preview', async () => {
    const mockData = { preview: 'data' };
    mock.onGet('http://localhost:8000/preview-dataset').reply(200, mockData);

    const response = await api.fetchDatasetPreview();
    expect(response).toEqual(mockData);
  });

  // Prueba para la función trainModels
  it('should train models', async () => {
    const mockData = { message: 'Models trained' };
    mock.onPost('http://localhost:8000/train').reply(200, mockData);

    const response = await api.trainModels();
    expect(response).toEqual(mockData);
  });

  // Prueba para la función saveModelSelection
  it('should save model selection', async () => {
    const mockData = { message: 'Model selection saved' };
    mock.onPost('http://localhost:8000/save-model-selection').reply(200, mockData);

    const response = await api.saveModelSelection('modelName');
    expect(response).toEqual(mockData);
  });

  // Prueba para la función fetchProjects
  it('should fetch projects', async () => {
    const mockData = [{ id: 1, name: 'Project 1' }];
    mock.onGet('http://localhost:8000/api/projects').reply(200, mockData);

    const response = await api.fetchProjects();
    expect(response).toEqual(mockData);
  });

  // Prueba para la función predictModels
  it('should predict models', async () => {
    const mockData = { prediction: 'result' };
    mock.onPost('http://localhost:8000/api/predict').reply(200, mockData);

    const response = await api.predictModels('project1', new File([], 'testfile'));
    expect(response).toEqual(mockData);
  });
});