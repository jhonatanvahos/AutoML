# Proyecto de Generación de Modelos de Machine Learning Supervisado

Este proyecto permite generar modelos de Machine Learning supervisado para problemas de regresión y clasificación. Cuenta con un backend desarrollado en Python utilizando FastAPI y un frontend desarrollado con React.

## Instrucciones de uso

1. **Crear un Entorno Virtual:**
   - Se recomienda utilizar una versión de Python 3.9 o superior. Puedes crear un entorno virtual siguiendo las instrucciones en [este enlace](https://docs.python.org/es/3/tutorial/venv.html).
   - El entorno virtual se debe crear un nivel antes del directorio anterior a la instalación o clonación del proyecto con el nombre **venvautoml**. Ejemplo: si el proyecto está en la ruta usuario/escritorio/AutoML , el entorno virtual deberá crearce en usuario/escritorio.

2. **Instalar Dependencias para el Back:**
   - Una vez activado el entorno virtual, instala las dependencias del backend del archivo `requirements.txt` utilizando el siguiente comando:
      ```bash
      cd AutoML/backend
      pip install -r requirements.txt
      ```
3. **Instalar Dependencias para el Front:**
   - Para el frontend se debe instalar Node.js en [este enlace](https://nodejs.org/en/download/package-manager)
   - Una vez instalado Node.js, instala las dependencias del frontend del archivo `package utilizando el siguiente comando:  
      ```bash
      cd AutoML/frontend
      npm install
      ```
4. **Ejecución:**
   - Una vez instaladas las dependencias del Back y del Front puedes hacer uno de la Aplicación ejecutando el archivo ejecución.py.
   - Para ejecutar la aplicación, ejecuta el siguiente comando en la carpeta raíz
      ```bash
      cd AutoML
      python ejecucion.py
      ```

## ACLARACIONES:
    1. Tratamiento de fechas se tomará como categorías. 
    2. Caracteristicas compuestas Ejemplo: Distancia = [15Km,20.000mts, 12Km] - deben indicarse la columna la unidad y los datos serán los valores . Distancia(KM) = [15,20,12]
    3. Textos como oraciones, fraces, articulos , etc. Solo se aceptaran categorías claras. 
    4. Los columnas con grandes desbalanceos entre caracteristicas o nulos . Se dejarán aparte en una lista para que el usuario defina que preprocesamiento aplicar para dichas columnas o si desea eliminarlas. 