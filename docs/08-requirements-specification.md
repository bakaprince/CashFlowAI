# Requirements Specification

## Functional Requirements
1. The system shall allow the user to upload a CSV or Excel dataset.
2. The system shall validate and clean the dataset.
3. The system shall select relevant financial features.
4. The system shall train a Multiple Linear Regression model.
5. The system shall predict future cash flow.
6. The system shall display evaluation metrics such as MAE, RMSE, and R².
7. The system shall explain model output using SHAP values.
8. The system shall support What-If analysis for business variable changes.
9. The system shall simulate risk using Monte Carlo analysis.
10. The system shall produce a recommendation based on thresholds and scenario results.
11. The system shall present results through an interactive dashboard.

## Non-Functional Requirements
1. The system should be easy to understand by students and evaluators.
2. The solution should be implemented in Python.
3. The system should be explainable and transparent.
4. The dashboard should be user friendly and visually clear.
5. The project should avoid unnecessary complexity.
6. The system should use an academic and feasible design.

## User Interface Requirements
- File upload area
- Dataset preview
- Forecasting summary
- SHAP contribution view
- What-If controls
- Risk simulation panel
- Recommendation output

## Assumptions
- The input data includes relevant financial metrics.
- The dataset is historical and time ordered.
- The project is designed as a prototype rather than a production-grade financial system.
