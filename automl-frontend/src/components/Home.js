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
    <header className="header">
      <div className="logo-container">
        <img src="logo.png" alt="PredictLab Logo" className="logo" />
        <h1>PredictLab</h1>
      </div>
    </header>
    
    <h2 className="home-title">Modelos Supervisados</h2>

    <div className="cards-container">
      <div className="card" onClick={handleTrainClick} aria-label="Entrenar un modelo">
        <h2>Entrenar</h2>
        <p>Entrena un modelo con tus datos.</p>
      </div>
      <div className="card" onClick={handlePredictClick} aria-label="Predecir usando un modelo">
        <h2>Predecir</h2>
        <p>Realiza predicciones usando un modelo entrenado.</p>
      </div>
    </div>

    <footer className="footer">
      <p>© 2024 PredictLab. Todos los derechos reservados.</p>
      <p>by: Jhonatan Stick Gomez Vahos</p>
      <p>Sebastian Saldarriaga Arias</p>
    </footer>
  </div>
  );
}

export default Home;
