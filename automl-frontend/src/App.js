import React from 'react';
import { BrowserRouter, Routes, Route } from 'react-router-dom';
import Home from './components/Home';
import Train from './components/Train';
import Predict from './components/Predict';
import ConfigForm from './components/ConfigForm';
import AutoML from './components/AutoML';

function App() {
    return (
        <BrowserRouter>
            <Routes>
                <Route path="/" element={<Home />} />
                <Route path="/train" element={<Train />} />
                <Route path="/predict" element={<Predict />} />
                <Route path="/ConfigForm" element={<ConfigForm />} />
                <Route path='/automl' element={<AutoML />} />
            </Routes>
        </BrowserRouter>
    );
}

export default App;
