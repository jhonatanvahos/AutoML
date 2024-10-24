// Predict.js
import React from 'react';
import { predictModels } from '../services/api';

function Predict() {

  const handleSubmit = async (e) => {
    e.preventDefault();
    try {
      const result = await predictModels(); // Llamar al backend para entrenar los modelos
      console.log("Resultado recibido del backend:", result);


    } catch (error) {
      console.error("Error updating config:", error);
    }
  };

  return (
    <form onSubmit={handleSubmit}>
       <label>Página de Predicción </label>
       <br />
      <button type="submit">Relizar predicción</button>
    </form>
  );

}

export default Predict;