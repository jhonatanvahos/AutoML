import React from 'react';
import { useNavigate } from 'react-router-dom';
import '../styles/Home.css'; // Archivo CSS para los estilos

function Home() {
  const navigate = useNavigate();

  const handleTrainClick = () => {
    navigate('/train'); // Redirige a la página de entrenamiento
  };

  const handlePredictClick = () => {
    navigate('/predict'); // Redirige a la página de predicción
  };

  return (
    <div className="home-container">
      <h1 className="home-title">AutoML - Modelos Supervisados</h1>
      
      <div className="cards-container">
        <div className="card" onClick={handleTrainClick}>
          <h2>Entrenar</h2>
          <p>Entrena un modelo con tus datos.</p>
        </div>
        <div className="card" onClick={handlePredictClick}>
          <h2>Predecir</h2>
          <p>Realiza predicciones usando un modelo entrenado.</p>
        </div>
      </div>
      
      <footer className="footer">
        <p>by: Jhonatan Stick Gomez Vahos</p>
        <p>Sebastian Saldarriaga Arias</p>
      </footer>
    </div>
  );
}

export default Home;
