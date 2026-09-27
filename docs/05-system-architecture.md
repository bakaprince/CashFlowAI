# System Architecture

## High-Level Architecture

The system follows a simple pipeline from raw business data to decision support output.

```text
Business Dataset
      ↓
Data Cleaning & Feature Creation
      ↓
Multiple Linear Regression Model
      ↓
Forecasted Cash Flow
      ↓
 ┌───────────────┬────────────────┬─────────────────┐
 ↓               ↓                ↓
SHAP           What-If         Monte Carlo
 ↓               ↓                ↓
Explain        Scenario        Risk Range
Prediction     Analysis        Estimation
 └───────────────┴────────────────┘
                    ↓
          Rule-Based Recommendation Engine
                    ↓
                 Streamlit Dashboard
```

## Components

### 1. Data Preparation Module
- Uploads CSV or Excel data.
- Checks basic column validity.
- Converts date fields.
- Converts numeric fields.
- Handles missing values.
- Sorts data chronologically.
- Selects key financial features.

### 2. Forecasting Module
- Uses Multiple Linear Regression.
- Learns patterns from historical business data.
- Predicts future cash flow values.
- Evaluates model using MAE, RMSE, and R².

### 3. Explainability Module
- Uses SHAP values to explain predictions.
- Identifies which variables contribute positively or negatively.

### 4. What-If Analysis Module
- Changes a business variable such as sales or expenses.
- Passes the modified input through the trained model.
- Compares baseline and scenario predictions.

### 5. Monte Carlo Risk Simulation Module
- Introduces controlled random variation around the forecast.
- Generates a distribution of possible outcomes.
- Displays central estimate and range of likely results.

### 6. Recommendation Module
- Uses predefined rules based on prediction, scenario difference, and threshold.
- Produces simple financial guidance.

### 7. Streamlit Dashboard
- Displays dataset preview and summary.
- Shows historical chart and forecasting results.
- Visualizes SHAP contributions.
- Supports scenario simulation and risk assessment.
- Displays recommendation outcome.

## Design Principle
The design intentionally prioritizes simplicity, interpretability, and demonstration of core AI concepts over complex advanced architectures.
