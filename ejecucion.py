import subprocess
import os
import sys

# Ruta base del proyecto
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Ruta del ambiente virtual (un nivel por encima de AutoML)
VENV_PATH = os.path.join(BASE_DIR, "..", "venvautoml")

# Rutas del backend y frontend
BACKEND_DIR = os.path.join(BASE_DIR, "backend")
FRONTEND_DIR = os.path.join(BASE_DIR, "frontend")

def start_backend():
    """
    Activa el ambiente virtual y lanza el backend con Uvicorn.
    """
    if os.name == "nt":
        # Comando para Windows
        activate_script = os.path.join(VENV_PATH, "Scripts", "activate.bat")
        command = f"{activate_script} && uvicorn app.main:app --reload"
    else:
        # Comando para Linux/Mac
        activate_script = os.path.join(VENV_PATH, "bin", "activate")
        command = f"source {activate_script} && uvicorn app.main:app --reload"

    print(f"Inicializando backend desde {BACKEND_DIR}...")
    return subprocess.Popen(command, shell=True, cwd=BACKEND_DIR)

def start_frontend():
    """
    Lanza el frontend con npm start.
    """
    print(f"Inicializando frontend desde {FRONTEND_DIR}...")
    return subprocess.Popen("npm start", shell=True, cwd=FRONTEND_DIR)

if __name__ == "__main__":
    print("Inicializando el Proyecto AutoML...")

    # Inicia el backend
    backend_process = start_backend()
    
    # Inicia el frontend
    frontend_process = start_frontend()

    try:
        print("Backend y frontend está corriendo. Presione Ctrl+C para finalizar la ejecución.")
        backend_process.wait()
        frontend_process.wait()
    except KeyboardInterrupt:
        print("\nCerrando los servidores...")
        backend_process.terminate()
        frontend_process.terminate()
        sys.exit(0)
