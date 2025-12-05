🔑 Concept Dictionary

AI (Artificial Intelligence) – Computers doing tasks that normally need human intelligence (e.g., recognizing patterns, making predictions).

ML (Machine Learning) – A part of AI where computers learn from examples instead of being given explicit rules.

Project (Big Picture) –
“We use data from the past, teach a computer program (a model) to spot patterns, and then use that model to make predictions about the future.”

Wildfire Use Case – Using weather + land data to predict where fires are likely to happen and how bad they might be.

Mental Health Use Case – Using phone + behavior data (steps, sleep, time at home, calls, screen time) to estimate a person’s depression risk.

Dataset – A big table of examples. Each row = one example (one week, one location, etc.).

Feature – An input column used to make a prediction.

Wildfires: temperature, humidity, wind speed, rain, vegetation, population, month, year.

Mental health: steps, time at home, sleep hours, screen time, etc.

Label / Target – What we want the model to predict.

Wildfires: fire_occurred (yes/no), burned_area.

Mental health: depression_score, high_risk (yes/no).

Model – A mathematical formula/program that uses features to predict the label.

Training a Model – Feeding the model many labeled examples from the past so it can adjust itself and learn the relationship between features and labels.

Testing a Model – Giving the trained model new examples it has never seen to check how accurate its predictions are.

Regression – Type of ML task where we predict a number.

Example: predict depression_score (0–27).

In this project: mental health regression.

Classification – Type of ML task where we predict a category.

Examples: fire_occurred = 0/1, high_risk = yes/no.

In this project: wildfire classification, mental health high-risk classification.

Neural Network – A more complex ML model inspired by the brain; great for images and sequences.

Example: CNNs (Convolutional Neural Networks) for detecting smoke/flames in wildfire photos (notebooks mention this idea, even if they use simpler models).

Simple Models (in notebooks) – Logistic Regression, Random Forest, Gradient Boosting: easier to train, explain, and good for tabular data.

ROC Curve – A graph showing how well a classification model separates positive vs negative cases across different thresholds.

AUC (Area Under the ROC Curve) – A single number summarizing how good the ROC curve is; closer to 1.0 = better.

SHAP – A tool to explain model predictions by showing which features push the prediction up or down.

Git – A tool that tracks changes to code over time (version control).

GitHub – A website where Git projects are stored and shared; used here to host and share notebooks and code.

🛠 Tool & Notebook Dictionary
Languages & Libraries

Python – The main programming language used for all AI/ML code in this project.

Jupyter Notebook – A “digital workbook” for Python with cells where you:

Write code

Run it

See results (tables, charts) immediately below

Pandas – Python library for working with tabular data (like Excel in code).

Reads CSV files (e.g., wildfire_synthetic.csv, mental_health_mobile_sensing_synthetic.csv).

Lets you filter rows, select columns, compute averages, etc.

NumPy – Python library for numbers and arrays.

Used for fast math operations that ML models rely on.

Scikit-Learn (sklearn) – Python library for machine learning.

Split data into training/test sets.

Train models (logistic regression, random forest, gradient boosting).

Measure performance (accuracy, precision, recall, F1, ROC-AUC, MAE, RMSE, R²).

Notebooks (Step-by-Step Story)

00_environment_setup.ipynb –
“Get tools ready.”

Imports Python libraries (NumPy, Pandas, Matplotlib, Scikit-Learn).

Ensures the environment and project structure (data/raw, notebooks, models) are set up.

10_wildfire_data_and_eda.ipynb –
“Understand wildfire data.”

Uses wildfire_synthetic.csv.

Loads data with Pandas.

Shows first rows and summary stats.

Checks class balance for fire_occurred (how many 1s vs 0s).

Plots simple charts (e.g., temperature vs fire occurrence).

Purpose: understand the data before training any model.

20_wildfire_ml_models.ipynb –
“Build wildfire prediction models.”

Features: temp_c, humidity, wind_speed, rain_mm_last_7d, vegetation_index, population_density, month, year.

Uses train_test_split (Scikit-Learn) → training set + test set.

Trains:

Logistic Regression (simple, interpretable).

Random Forest (more flexible).

Evaluates with: Accuracy, Precision, Recall, F1, ROC-AUC.

Plots feature importances for Random Forest.

Story: “We feed weather + land features into Python and learn to predict fire risk.”

30_mental_health_data_and_eda.ipynb –
“Understand mental health data.”

Uses mental_health_mobile_sensing_synthetic.csv.

Loads behavior data (steps, distance, time at home, calls/texts, sleep, screen time) with Pandas.

Shows distributions of depression_score and high_risk.

Plots relationships (e.g., steps vs depression score).

Story: “We convert weekly phone/behavior logs into numbers and see how they relate to mental health scores.”

40_mental_health_ml_models.ipynb –
“Build mental health prediction models.”

Features: steps, distance, time at home, calls/texts, sleep, screen time, etc.

Splits data into training/test sets.

Trains:

Gradient Boosting Regressor → predicts depression_score (regression).

Random Forest Classifier → predicts high_risk (classification).

Evaluates:

Regression: MAE, RMSE, R².

Classification: Accuracy, Precision, Recall, F1, ROC-AUC.

Plots feature importances to see which behaviors are most linked to risk.

Story: “Given weekly behavior, can we guess mental health scores and high-risk weeks?”

50_model_evaluation_and_explainability.ipynb –
“Compare and explain models.”

Trains quick Random Forest models for:

Wildfire (fire_occurred).

Mental health (high_risk).

Plots ROC curves for both; computes AUC.

Optionally uses SHAP for feature-level explanations.

Story: “We check how strong the models are and explain why they predict what they do.”

60_serving_and_automation.ipynb –
“Use models in the real world.”

Trains a wildfire model again.

Does batch scoring: adds a risk_score to all rows.

Defines predict_fire_risk(...) function in Python that returns a probability of fire given new conditions.

Can be called from web apps, dashboards, or mobile apps.

Story: “The model leaves the lab and becomes a tool other systems can call.”

Learning Roadmap (as a mini-dictionary entry)

Suggested Notebook Order –

00_environment_setup.ipynb – tools ready

10_wildfire_data_and_eda.ipynb – understand wildfire data

20_wildfire_ml_models.ipynb – build wildfire models

30_mental_health_data_and_eda.ipynb – understand mental health data

40_mental_health_ml_models.ipynb – build mental health models

50_model_evaluation_and_explainability.ipynb – compare & explain

60_serving_and_automation.ipynb – use models as a service

You can literally print this and give it as a keyword dictionary for the session.