import React, { useEffect, useState } from 'react';

function LoadingScreen() {
    const [currentModel, setCurrentModel] = useState("Cargando...");
    const [elapsedTime, setElapsedTime] = useState(null);
    const [trainingProgress, setTrainingProgress] = useState("");
    const [progress, setProgress] = useState(0);

    useEffect(() => {
        // Establecer la conexión SSE
        const eventSource = new EventSource("http://localhost:8000/train/status");

        eventSource.onmessage = function (event) {
            try {
                // Ahora manejamos el evento correctamente, sin 'data:'
                const data = JSON.parse(event.data);  // Aquí no se necesita 'data:' porque ya está en formato JSON.
                console.log("Datos recibidos del servidor:", data);

                // Actualizamos el estado con los datos recibidos
                setCurrentModel(data.current_model);
                setElapsedTime(data.elapsed_time_minutes);
                setTrainingProgress(data.progress);

                if (data.progress === "Entrenamiento completado") {
                    setProgress(100);  // Si el progreso es completado, establecemos el 100%.
                } else {
                    setProgress(data.elapsed_time_minutes);  // De lo contrario, actualizamos el progreso en función del tiempo transcurrido.
                }
            } catch (error) {
                console.error("Error al procesar los datos SSE:", error);
                console.log("Datos del evento SSE:", event.data);  // Imprime los datos para verificar lo que está recibiendo.
            }
        };

        // Manejo de error de conexión SSE
        eventSource.onerror = function (error) {
            console.error("Error en la conexión SSE:", error);
            eventSource.close(); // Cerrar la conexión en caso de error.
        };

        // Limpiar la conexión cuando el componente se desmonte
        return () => {
            eventSource.close();
        };
    }, []);

    return (
        <div className="loading-screen">
            <header className="header">
                <div className="logo-container">
                    <img src="logo.png" alt="PredictLab Logo" className="logo" />
                    <h1>PredictLab</h1>
                </div>
            </header>

            <h1>Entrenando Modelo: {currentModel}</h1>
            {trainingProgress === "Entrenamiento completado" ? (
                <p>Entrenamiento completado en {elapsedTime} minutos</p>
            ) : (
                <>
                    <p>Progreso: {trainingProgress}</p>
                    <div className="progress-bar">
                        <div className="progress" style={{ width: `${progress}%` }}></div>
                    </div>
                </>
            )}

            <footer className="footer">
                <p>© 2024 PredictLab. Todos los derechos reservados.</p>
                <p>by: Jhonatan Stick Gomez Vahos</p>
                <p>Sebastian Saldarriaga Arias</p>
            </footer>
        </div>
    );
}

export default LoadingScreen;
