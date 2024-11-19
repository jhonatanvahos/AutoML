import React, { useEffect, useState } from "react";
import Plot from "react-plotly.js";
import { useNavigate } from "react-router-dom";
import { fetchDatasetPreview } from "../services/api"; // Función de la API

const DatasetPreview = ({ targetColumn, onContinue }) => {
  const [dataPreview, setDataPreview] = useState([]);
  const [columns, setColumns] = useState([]);
  const [numericColumns, setNumericColumns] = useState([]);
  const [categoricalColumns, setCategoricalColumns] = useState([]);
  const [loading, setLoading] = useState(true);

  const navigate = useNavigate(); // Para la navegación entre páginas

  useEffect(() => {
    const fetchPreview = async () => {
      try {
        const data = await fetchDatasetPreview(); // Llamada al backend
        setDataPreview(data.dataPreview); // Aquí se guarda todo el dataframe
        setColumns(Object.keys(data.dataPreview[0] || {}));
        setNumericColumns(data.numericColumns);
        setCategoricalColumns(data.categoricalColumns);
        setLoading(false);
      } catch (error) {
        console.error("Error fetching dataset preview:", error);
        setLoading(false);
      }
    };

    if (targetColumn) {
      fetchPreview();
    }
  }, [targetColumn]);

  const handleContinue = () => {
    onContinue(); // Llamar función pasada como prop
    const filteredColumns = columns.filter(column => column !== targetColumn);
    navigate('/ConfigForm', { state: { columns: filteredColumns } });
  };

  const renderNumericVisualizations = () =>
    numericColumns.map((col) => (
      <div key={col} style={{ marginBottom: "20px" }}>
        <Plot
          data={[
            {
              x: dataPreview.map((row) => row[col]),
              type: "histogram",
              marker: { color: "blue" },
            },
          ]}
          layout={{
            title: `Distribución de ${col}`,
            xaxis: { title: col },
            yaxis: { title: "Frecuencia" },
            margin: { t: 50, l: 50, r: 50, b: 50 },
          }}
        />
      </div>
    ));

  const renderCategoricalVisualizations = () =>
    categoricalColumns.map((col) => {
      const counts = dataPreview.reduce((acc, row) => {
        const value = row[col] || "Desconocido";
        acc[value] = (acc[value] || 0) + 1;
        return acc;
      }, {});

      return (
        <div key={col} style={{ marginBottom: "20px" }}>
          <Plot
            data={[
              {
                x: Object.keys(counts),
                y: Object.values(counts),
                type: "bar",
                marker: { color: "orange" },
              },
            ]}
            layout={{
              title: `Distribución de ${col}`,
              xaxis: { title: col, tickangle: 45 },
              yaxis: { title: "Frecuencia" },
              margin: { t: 50, l: 50, r: 50, b: 100 },
            }}
          />
        </div>
      );
    });

  return (
    <div className="train-container">
      <header className="header">
        <div className="logo-container">
          <img src="logo.png" alt="PredictLab Logo" className="logo" />
          <h1>PredictLab</h1>
        </div>
      </header>

      {loading ? (
        <p>Cargando datos...</p>
      ) : (
        <>
          <h2 className="train-title">Previsualización del Dataset</h2>
          <table className="data-table">
            <thead>
              <tr>
                {columns.map((col) => (
                  <th key={col}>{col}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {dataPreview.slice(0, 10).map((row, index) => ( // Mostrar solo 10 registros
                <tr key={index}>
                  {columns.map((col) => (
                    <td key={`${index}-${col}`}>{row[col]}</td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>

          <h3>Visualización de Variables Numéricas</h3>
          {numericColumns.length > 0 ? (
            renderNumericVisualizations()
          ) : (
            <p>No hay variables numéricas para visualizar.</p>
          )}

          <h3>Visualización de Variables Categóricas</h3>
          {categoricalColumns.length > 0 ? (
            renderCategoricalVisualizations()
          ) : (
            <p>No hay variables categóricas para visualizar.</p>
          )}

          <button onClick={handleContinue} className="continue-button">
            Continuar
          </button>
        </>
      )}

      <footer className="footer">
        <p>© 2024 PredictLab. Todos los derechos reservados.</p>
        <p>by: Jhonatan Stick Gomez Vahos</p>
        <p>Sebastian Saldarriaga Arias</p>
      </footer>
    </div>
  );
};

export default DatasetPreview;
