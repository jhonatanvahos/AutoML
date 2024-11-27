import React, { useState } from 'react';
import { useLocation, Link } from 'react-router-dom';
import ReactPaginate from 'react-paginate';
import Plot from 'react-plotly.js';
import styles from './PredictionResults.module.css';

function ResultPredict() {
  const location = useLocation();
  const result = location.state?.result;

  // Verifica si es clasificación o regresión
  const isClassification = result.model_type === "classification";
  const isTest = result.data === "test";

  // Lógica de paginación
  const [currentPage, setCurrentPage] = useState(0);
  const resultsPerPage = 10;

  const handlePageClick = (event) => {
    setCurrentPage(event.selected);
  };

  const indexOfLastResult = (currentPage + 1) * resultsPerPage;
  const indexOfFirstResult = indexOfLastResult - resultsPerPage;
  const currentResults = result.predictions.slice(indexOfFirstResult, indexOfLastResult);

  const downloadCSV = () => {
    const headers = Object.keys(result.predictions[0]).filter((key) => key !== 'match');
    const rows = result.predictions.map((prediction) =>
      headers.map((header) => prediction[header] ?? '')
    );

    const csvContent = [headers.join(','), ...rows.map((row) => row.join(','))].join('\n');
    const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
    const url = URL.createObjectURL(blob);

    const link = document.createElement('a');
    link.href = url;
    link.setAttribute('download', 'resultados_predicciones.csv');
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  // Datos para el gráfico de regresión
  const regressionData = {
    labels: result.predictions.map((_, idx) => idx + 1),
    datasets: [
      {
        label: 'Valores reales',
        data: result.actual_values,
        borderColor: 'rgba(75,192,192,1)',
        borderWidth: 1,
        fill: false,
      },
      {
        label: 'Valores predichos',
        data: result.predicted_values,
        borderColor: 'rgba(255,99,132,1)',
        borderWidth: 1,
        fill: false,
      },
    ],
  };

  // Comprobar si la matriz de confusión está definida antes de intentar acceder a ella
  const confusionMatrixData = result.confusion_matrix || null;

  // Si la matriz de confusión está definida, procesar los datos para el gráfico
  let confusionMatrixContent = null;
  if (isClassification && confusionMatrixData) {
    const xLabels = ['Predicción: Negativo', 'Predicción: Positivo'];
    const yLabels = ['Real: Negativo', 'Real: Positivo'];

    // Calcular los porcentajes y crear etiquetas de texto
    const total = confusionMatrixData.flat().reduce((a, b) => a + b, 0);
    const confusionMatrixPercentages = confusionMatrixData.map(row =>
      row.map(value => (value / total) * 100)  // Convertir a porcentaje respecto al total
    );

    const textLabels = confusionMatrixData.map((row, i) =>
      row.map((value, j) => `${value} (${confusionMatrixPercentages[i][j].toFixed(2)}%)`)
    );

    confusionMatrixContent = (
      <div className="heatmap-container">
        <Plot
          data={[
            {
              z: confusionMatrixData,
              x: xLabels,
              y: yLabels,
              type: 'heatmap',
              colorscale: 'YlGnBu',
              colorbar: {
                title: 'Frecuencia',
              },
              text: textLabels, // Mostrar los valores y porcentajes como texto
              hoverinfo: 'text', // Mostrar texto al pasar el ratón
              textfont: {
                color: 'black',  // Cambia el color del texto a negro (o el color que prefieras)
                size: 16,        // Asegúrate de que el tamaño del texto sea adecuado
              },
            },
          ]}
          layout={{
            title: 'Matriz de Confusión',
            xaxis: { title: 'Predicción' },
            yaxis: { title: 'Real' },
            margin: { t: 50, b: 50, l: 50, r: 50 },
          }}
        />
      </div>
    );
  }

  return (
    <div className={styles.container}>
      <header className={styles.header}>
        <div className={styles.logoContainer}>
          <img src="/logo.png" alt="PredictLab Logo" className={styles.logo} />
          <h1>PredictLab</h1>
        </div>
      </header>

      <h2 className={styles.title}>Resultados de la predicción</h2>

      <div>
        {isClassification ? (
          <>
            <p><strong>Total predicciones: </strong>{result.total_predictions}</p>

            {/* Mostrar solo si isTest es true */}
            {isTest && (
              <>
                <p><strong>Exactitud:</strong> {result.accuracy}
                  <span className="info-icon" title="Fórmula: Accuracy = (TP + TN) / (TP + TN + FP + FN)">ℹ️</span>
                </p>
                <p><strong>Precisión:</strong> {result.precision}
                  <span className="info-icon" title="Fórmula: Precision = TP / (TP + FP)">ℹ️</span>
                </p>
                <p><strong>Puntuación F1:</strong> {result.f1_score}
                  <span className="info-icon" title="Fórmula: F1 = 2 * (Precision * Recall) / (Precision + Recall)">ℹ️</span>
                </p>

                {/* Mostrar la matriz de confusión solo si es clasificación */}
                {confusionMatrixContent}
              </>
            )}
          </>
        ) : (
          <>
            <p><strong>Total predicciones: </strong>{result.total_predictions}</p>

            {/* Mostrar solo si isTest es true */}
            {isTest && (
              <>
                <p><strong>(MSE) Error cuadrático medio:</strong> {result["Error cuadrático medio"]}
                  <span className="info-icon" title="Fórmula: MSE = (1/n) * Σ(yi - ŷi)², donde yi es el valor real y ŷi es el valor predicho">ℹ️</span>
                </p>
                <p><strong>(MAE) Error absoluto medio:</strong> {result["Error absoluto medio"]}
                  <span className="info-icon" title="Fórmula: MAE = (1/n) * Σ|yi - ŷi|, donde yi es el valor real y ŷi es el valor predicho">ℹ️</span>
                </p>
                <p><strong>(R²) Coeficiente de determinación:</strong> {result["R2"]}
                  <span className="info-icon" title="Fórmula: R² = 1 - (Σ(yi - ŷi)² / Σ(yi - ȳ)²), donde yi es el valor real y ȳ es la media de los valores reales">ℹ️</span>
                </p>
              </>
            )}

            {/* Mostrar la gráfica de regresión solo si isTest es true */}
            {isTest && (
              <div className="regression-container">
                <Plot
                  data={[
                    {
                      x: regressionData.labels,
                      y: result.actual_values,
                      type: 'scatter',
                      mode: 'lines+markers',
                      name: 'Valores reales',
                      line: { color: 'rgba(75,192,192,1)', width: 2 },
                    },
                    {
                      x: regressionData.labels,
                      y: result.predicted_values,
                      type: 'scatter',
                      mode: 'lines+markers',
                      name: 'Valores predichos',
                      line: { color: 'rgba(255,99,132,1)', width: 2 },
                    },
                  ]}
                  layout={{
                    title: 'Regresión: Valores reales vs. Predichos',
                    xaxis: { title: 'Índice' },
                    yaxis: { title: 'Valores' },
                    margin: { t: 50, b: 50, l: 50, r: 50 },
                  }}
                />
              </div>
            )}
          </>
        )}
      </div>


      <div className={styles.tableContainer}>
        <table className={styles.table}>
          <thead className={styles.thead}>
            <tr className={styles.tr}>
              <th>Indice</th>
              {Object.keys(currentResults[0]).map((key) => key !== 'match' && <th key={key}>{key.replace(/_/g, ' ').toUpperCase()}</th>)}
            </tr>
          </thead>
          <tbody className={styles.tbody}>
            {currentResults.map((prediction, index) => (
              <tr className={styles.tr} key={index}>
                <td>{indexOfFirstResult + index + 1}</td>
                {Object.keys(prediction).map((key) => key !== 'match' && <td key={key}>{prediction[key]}</td>)}
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <button onClick={downloadCSV} className={styles.buttonDownload}>
        Descargar Resultados (CSV)
      </button>

      <ReactPaginate
        previousLabel={"Anterior"}
        nextLabel={"Siguiente"}
        breakLabel={"..."}
        pageCount={Math.ceil(result.predictions.length / resultsPerPage)}
        marginPagesDisplayed={2}
        pageRangeDisplayed={5}
        onPageChange={handlePageClick}
        containerClassName={styles.pagination}
        activeClassName={"active"}
      />

      <Link to="/" className={styles.buttonHome}>Regresar al inicio</Link>

      <footer className={styles.footer}>
        <p>© 2024 PredictLab. Todos los derechos reservados.</p>
        <p>by: Jhonatan Stick Gomez Vahos</p>
        <p>Sebastian Saldarriaga Arias</p>
      </footer>
    </div>
  );
}

export default ResultPredict;
