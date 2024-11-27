import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_upload_dataset_csv():
    # Simula un archivo CSV para la prueba
    test_file_content = "col1,col2\n1,2\n3,4"
    test_file = {"file": ("test.csv", test_file_content, "text/csv")}
    
    response = client.post("/upload-dataset", files=test_file)
    assert response.status_code == 200
    data = response.json()
    assert "message" in data
    assert data["message"] == "File uploaded successfully"
    assert "columns" in data
    assert data["columns"] == ["col1", "col2"]

@pytest.mark.asyncio
async def test_preview_dataset():
    # Prueba la ruta de previsualización
    response = client.get("/preview-dataset")
    assert response.status_code == 200
    data = response.json()
    assert "dataPreview" in data
    assert "numericColumns" in data
    assert "categoricalColumns" in data
