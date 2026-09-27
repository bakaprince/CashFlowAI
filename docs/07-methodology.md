# Methodology

## Research Approach
This project follows a practical software-development and AI-modeling approach. The methodology is designed to be understandable, reproducible, and suitable for an exhibition project.

## Step 1: Data Collection
The system takes financial dataset files in CSV or Excel format. Relevant features may include sales, expenses, receivables, payables, cash inflow, cash outflow, and cash balance.

## Step 2: Data Cleaning and Preparation
The preprocessing stage checks for missing values, invalid values, date conversion issues, and inconsistent column names. The data is cleaned and sorted chronologically before use.

## Step 3: Feature Selection
The most relevant business variables are selected as model inputs. For this project, features such as sales, expenses, receivables, and payables are used to predict cash flow.

## Step 4: Model Training
A Multiple Linear Regression model is trained on historical data. The model learns the relationship between selected input variables and the target variable, which is cash flow.

## Step 5: Prediction and Evaluation
The trained model predicts future cash flow and is evaluated with MAE, RMSE, and R². These metrics help measure the accuracy and reliability of the forecast.

## Step 6: Explainability with SHAP
SHAP is applied after model training to explain how each feature contributes to the prediction. It helps show why the model predicts a certain cash-flow value.

## Step 7: What-If Analysis
The scenario module changes one dataset feature, such as sales or expenses, and passes the modified input through the trained model to observe the impact on future cash flow.

## Step 8: Monte Carlo Simulation
To represent uncertainty, the system introduces controlled random variation around the predicted cash flow value using Monte Carlo simulation. This produces a range of likely outcomes.

## Step 9: Recommendation Logic
The recommendation engine applies predefined rules based on predicted cash flow, scenario outcome, and threshold values to generate business recommendations.

## Step 10: Dashboard Presentation
The final system is presented through a Streamlit dashboard so users can upload data, view results, test scenarios, and understand recommendations in a user-friendly interface.
