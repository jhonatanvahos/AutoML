import React from 'react';
import { BrowserRouter, Routes, Route } from 'react-router-dom';
import Home from './components/Home/HomePage';
import Train from './components/Train/TrainProjectPage';
import ConfigForm from './components/Train/TrainingConfigForm';
import AutoML from './components/Train/ModelResults';
import Predict from './components/Predict/PredictionPage';
import ResultPredict from './components/Predict/PredictionResults';

function App() {
    return (
        <BrowserRouter>
            <Routes>
                <Route path="/" element={<Home />} />
                <Route path="/train" element={<Train />} />
                <Route path="/predict" element={<Predict />} />
                <Route path="/ConfigForm" element={<ConfigForm />} />
                <Route path='/automl' element={<AutoML />} />
                <Route path='/resultpredict' element={<ResultPredict />} />
            </Routes>
        </BrowserRouter>
    );
}

export default App;