import React, { useEffect, useState } from 'react';
import '../styles/LoadingScreen.css';

function LoadingScreen() {
    const [progress, setProgress] = useState(0);

    useEffect(() => {
        // Actualiza el progreso cada 20ms para que complete el ciclo en 2 segundos (100 * 20ms = 2000ms)
        const interval = setInterval(() => {
            setProgress((prev) => (prev >= 100 ? 0 : prev + 1));
        }, 20);

        return () => clearInterval(interval); // Limpia el intervalo al desmontar
    }, []);

    return (
        <div className="loading-screen">
            <header className="header">
                <div className="logo-container">
                    <img src="logo.png" alt="PredictLab Logo" className="logo" />
                    <h1>PredictLab</h1>
                </div>
            </header>

            <div className="loading-content">
                <h1>Entrenando Modelos...</h1>
                <div className="progress-bar">
                    <div className="progress" style={{ width: `${progress}%` }}></div>
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

export default LoadingScreen;
