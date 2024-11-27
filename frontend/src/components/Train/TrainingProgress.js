import React, { useEffect, useState } from 'react';
import styles from './TrainingProgress.module.css';

function LoadingScreen() {
    const [progress, setProgress] = useState(0);
    const [timer, setTimer] = useState(0);

    useEffect(() => {
        // Incrementa el progreso cada 20ms
        const progressInterval = setInterval(() => {
            setProgress((prev) => (prev >= 100 ? 0 : prev + 1));
        }, 20);

        // Incrementa el temporizador cada 1000ms (1 segundo)
        const timerInterval = setInterval(() => {
            setTimer((prev) => prev + 1);
        }, 1000);

        // Limpia ambos intervalos al desmontar el componente
        return () => {
            clearInterval(progressInterval);
            clearInterval(timerInterval);
        };
    }, []);

    return (
        <div className={styles.container}>
            {/* Header */}
            <header className={styles.header}>
                <div className={styles.logoContainer}>
                <img src="/logo.png" alt="PredictLab Logo" className={styles.logo} />
                <h1>PredictLab</h1>
                </div>
            </header>

            <div className={styles.loadingContent}>
                <h1>Entrenando Modelos...</h1>
                <div className={styles.progressBar}>
                    <div className={styles.progress} style={{ width: `${progress}%` }}></div>
                </div>
                {/* Temporizador debajo de la barra de carga */}
                <p className={styles.timer}>Tiempo transcurrido: {timer} segundos</p>
            </div>

            <footer className={styles.footer}>
                <p>© 2024 PredictLab. Todos los derechos reservados.</p>
                <p>by: Jhonatan Stick Gomez Vahos</p>
                <p>Sebastian Saldarriaga Arias</p>
            </footer>
        </div>
    );
}

export default LoadingScreen;
