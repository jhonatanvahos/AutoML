import React, { useState } from 'react';
import { useLocation, Link } from 'react-router-dom';
import ReactPaginate from 'react-paginate';
import '../styles/ResultPredict.css';

function ResultPredict() {
  const location = useLocation();
  const result = location.state?.result;

  // Verifica si es clasificación o regresión
  const isClassification = result.model_type === "classification";

  // Lógica de paginación
  const [currentPage, setCurrentPage] = useState(0);
  const resultsPerPage = 10;

  const handlePageClick = (event) => {
    setCurrentPage(event.selected);
  };

  // Calcular los resultados a mostrar para la página actual
  const indexOfLastResult = (currentPage + 1) * resultsPerPage;
  const indexOfFirstResult = indexOfLastResult - resultsPerPage;
  const currentResults = result.predictions.slice(indexOfFirstResult, indexOfLastResult);

  // Función para generar el CSV
  const downloadCSV = () => {
    const headers = Object.keys(result.predictions[0]).filter((key) => key !== 'match'); // Excluye 'match' si no es necesario

    const rows = result.predictions.map((prediction) => 
      headers.map((header) => {
        // Asegura que todos los valores, incluso los 0, sean incluidos
        let value = prediction[header] === undefined || prediction[header] === null ? '' : prediction[header];

        // Si es un valor numérico, elimina los separadores de miles (si es que hay)
        if (typeof value === 'number') {
          value = value.toFixed(2);  // Asegura que siempre tenga dos decimales, ajusta según lo necesites
        } else if (typeof value === 'string') {
          // Reemplaza comas por puntos si es necesario para evitar problemas con el CSV
          value = value.replace(/,/g, '');  // Elimina cualquier coma en los números
        }

        return value;
      })
    );

    // Genera el contenido del CSV
    const csvContent = [
      headers.join(','), // Encabezados
      ...rows.map((row) => row.join(',')) // Filas
    ].join('\n');

    // Crear el archivo blob y disparar la descarga
    const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
    const url = URL.createObjectURL(blob);

    const link = document.createElement('a');
    link.href = url;
    link.setAttribute('download', 'resultados_predicciones.csv');
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  return (
    <div className="resultpredict-container">
      <header className="header">
        <div className="logo-container">
          <img src="logo.png" alt="PredictLab Logo" className="logo" />
          <h1>PredictLab</h1>
        </div>
      </header>

      <h2 className="predict-title">Resultados de la predicción</h2>

      {/* Mostrar métricas dinámicamente */}
      <div>
        {isClassification ? (
          <>
            <p><strong>Accuracy:</strong> {result.accuracy}</p>
            <p><strong>Precision:</strong> {result.precision}</p>
            <p><strong>F1 Score:</strong> {result.f1_score}</p>
            <p><strong>Total Predictions:</strong> {result.total_predictions}</p>
            <p><strong>Correct Predictions:</strong> {result.correct_predictions}</p>
            <p><strong>Incorrect Predictions:</strong> {result.incorrect_predictions}</p>
            <p><strong>Prediction Accuracy:</strong> {(result.prediction_accuracy * 100).toFixed(2)}%</p>
          </>
        ) : (
          <>
            <p><strong>Mean Squared Error (MSE):</strong> {result["Error cuadrático medio"]}</p>
            <p><strong>Mean Absolute Error (MAE):</strong> {result["Error absoluto medio"]}</p>
            <p><strong>R-Squared (R²):</strong> {result["R2"]}</p>
          </>
        )}
      </div>

      {/* Tabla de resultados */}
      <div className="table-container">
        <table>
          <thead>
            <tr>
              <th>Index</th>
              {Object.keys(currentResults[0]).map((key) => {
                if (key !== 'match') {
                  return <th key={key}>{key.replace(/_/g, ' ').toUpperCase()}</th>;
                }
                return null;
              })}
            </tr>
          </thead>
          <tbody>
            {currentResults.map((prediction, index) => (
              <tr key={index}>
                <td>{indexOfFirstResult + index + 1}</td>
                {Object.keys(prediction).map((key) => {
                  if (key !== 'match') {
                    return <td key={key}>{prediction[key]}</td>;
                  }
                  return null;
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      {/* Botón de descarga */}
      <button onClick={downloadCSV} className="button-download">
        Descargar Resultados (CSV)
      </button>

      {/* Paginación */}
      <ReactPaginate
        previousLabel={"Previous"}
        nextLabel={"Next"}
        breakLabel={"..."}
        pageCount={Math.ceil(result.predictions.length / resultsPerPage)}
        marginPagesDisplayed={2}
        pageRangeDisplayed={5}
        onPageChange={handlePageClick}
        containerClassName={"pagination"}
        activeClassName={"active"}
      />

      <Link to="/" className="button-home">Return to Home</Link>
      
      <footer className="footer">
        <p>© 2024 PredictLab. Todos los derechos reservados.</p>
        <p>by: Jhonatan Stick Gomez Vahos</p>
        <p>Sebastian Saldarriaga Arias</p>
      </footer>
    </div>
  );
}

export default ResultPredict;
