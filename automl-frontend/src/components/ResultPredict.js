import React, { useState } from 'react';
import { useLocation , Link } from 'react-router-dom';
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
      <table>
        <thead>
          <tr>
            <th>Index</th>
            {/* Genera encabezados dinámicos para cada columna en la predicción */}
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
              {/* Genera celdas dinámicas para cada predicción */}
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
