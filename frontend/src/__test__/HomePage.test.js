import React from "react";
import { render, screen, fireEvent } from "@testing-library/react";
import { BrowserRouter } from "react-router-dom";
import HomePage from "../components/Home/HomePage";
import '@testing-library/jest-dom';

// Mockear useNavigate
jest.mock("react-router-dom", () => {
  const actual = jest.requireActual("react-router-dom");
  return {
    ...actual,
    useNavigate: jest.fn(), // Mock de useNavigate
  };
});

import { useNavigate } from "react-router-dom";

describe("HomePage Component", () => {
  let mockNavigate;

  beforeEach(() => {
    // Reiniciar mock antes de cada prueba
    mockNavigate = jest.fn();
    useNavigate.mockReturnValue(mockNavigate);
  });

  const renderWithRouter = (ui) => {
    return render(<BrowserRouter>{ui}</BrowserRouter>);
  };

  test("renders the header, main content, and footer", () => {
    renderWithRouter(<HomePage />);
  
    // Verificar encabezado
    const logo = screen.getByAltText("PredictLab Logo");
    const headerTitle = screen.getByRole("heading", { name: "PredictLab" }); // Buscar específicamente el <h1>
    expect(logo).toBeInTheDocument();
    expect(headerTitle).toBeInTheDocument();
  
    // Verificar contenido principal
    const mainTitle = screen.getByRole("heading", { name: /Modelos Supervisados/i });
    expect(mainTitle).toBeInTheDocument();
  
    const trainCard = screen.getByRole("button", { name: /Entrenar un modelo/i });
    const predictCard = screen.getByRole("button", { name: /Predecir usando un modelo/i });
    expect(trainCard).toBeInTheDocument();
    expect(predictCard).toBeInTheDocument();
  
    // Verificar pie de página
    const footerText = screen.getByText(/© 2024 PredictLab. Todos los derechos reservados./i);
    expect(footerText).toBeInTheDocument();
  });

  test("navigates to /train when the train card is clicked", () => {
    renderWithRouter(<HomePage />);

    const trainCard = screen.getByRole("button", { name: /Entrenar un modelo/i });
    fireEvent.click(trainCard);

    expect(mockNavigate).toHaveBeenCalledWith("/train");
    expect(mockNavigate).toHaveBeenCalledTimes(1);
  });

  test("navigates to /predict when the predict card is clicked", () => {
    renderWithRouter(<HomePage />);

    const predictCard = screen.getByRole("button", { name: /Predecir usando un modelo/i });
    fireEvent.click(predictCard);

    expect(mockNavigate).toHaveBeenCalledWith("/predict");
    expect(mockNavigate).toHaveBeenCalledTimes(1);
  });
});
