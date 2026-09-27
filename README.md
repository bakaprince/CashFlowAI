# AI-Powered Cash Flow Forecasting and Decision Support System

This project is a simplified, student-friendly implementation of an AI-based cash flow forecasting and decision support system.

## Features

- Data upload and preprocessing
- Multiple Linear Regression forecasting
- SHAP explainability for model predictions
- What-if scenario analysis
- Monte Carlo risk simulation
- Rule-based recommendation engine
- Streamlit dashboard

## Project flow

Business data -> Data cleaning -> Forecast -> SHAP -> What-if -> Monte Carlo -> Recommendation -> Dashboard

## Tech stack

- Python
- Pandas
- NumPy
- Scikit-learn
- SHAP
- Plotly
- Streamlit

## Run locally

1. Create and activate a virtual environment.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Start the app:
   ```bash
   streamlit run app.py
   ```

## Default data

A sample CSV file is included in the `data/` folder for quick testing.

## Notes

This is intentionally a simple academic prototype based on the project specification and avoids deep learning or LLM-based recommendations.
