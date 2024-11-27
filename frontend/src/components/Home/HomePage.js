import React from 'react';
import { useNavigate } from 'react-router-dom';
import styles from './HomePage.module.css'; // Uso de módulos CSS

// Definición de rutas como constantes para facilitar futuros cambios
const ROUTES = {
  TRAIN: '/train',
  PREDICT: '/predict',
};

function HomePage() {
  const navigate = useNavigate();

  // Handlers para navegación
  const handleNavigation = (path) => {
    navigate(path);
  };

  return (
    <div className={styles.container} aria-label="Página de inicio de PredictLab">
      {/* Header */}
      <header className={styles.header}>
        <div className={styles.logoContainer}>
          <img src="/logo.png" alt="PredictLab Logo" className={styles.logo} />
          <h1>PredictLab</h1>
        </div>
      </header>

      {/* Título principal */}
      <main>
        <h2 className={styles.title}>Modelos Supervisados</h2>

        {/* Tarjetas de acciones */}
        <div className={styles.cardsContainer}>
          <div
            className={styles.card}
            onClick={() => handleNavigation(ROUTES.TRAIN)}
            onKeyPress={(e) => e.key === 'Enter' && handleNavigation(ROUTES.TRAIN)}
            role="button"
            tabIndex="0"
            aria-label="Entrenar un modelo"
          >
            <h2>Entrenar</h2>
            <p>Entrena un modelo con tus datos.</p>
          </div>
          <div
            className={styles.card}
            onClick={() => handleNavigation(ROUTES.PREDICT)}
            onKeyPress={(e) => e.key === 'Enter' && handleNavigation(ROUTES.PREDICT)}
            role="button"
            tabIndex="0"
            aria-label="Predecir usando un modelo"
          >
            <h2>Predecir</h2>
            <p>Realiza predicciones usando un modelo entrenado.</p>
          </div>
        </div>
      </main>

      {/* Footer */}
      <footer className={styles.footer}>
        <p>© 2024 PredictLab. Todos los derechos reservados.</p>
        <p>by: Jhonatan Stick Gomez Vahos</p>
        <p>Sebastian Saldarriaga Arias</p>
      </footer>
    </div>
  );
}

export default HomePage;
