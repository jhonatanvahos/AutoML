import React from 'react';
import { useLocation, Link } from 'react-router-dom';
import './AutoML.css';

function AutoML() {
  const location = useLocation();
  const metrics  = location.state

  if (!metrics) {
    return <div>No metrics available. Something went wrong.</div>;
  }

  return (
    <div className="container">
      <h1>Model Training Results</h1>
      <h2>Winner Model Metrics:</h2>
      <div className="model-metrics">
        <ul>
          <li><span>Model:</span> {metrics.result.model_name}</li>
          <li><span>Metricas:</span> {metrics.result.score}</li>
        </ul>
      </div>
      <Link to="/" className="button-home">Return to Home</Link>
    </div>
  );
}

export default AutoML;
