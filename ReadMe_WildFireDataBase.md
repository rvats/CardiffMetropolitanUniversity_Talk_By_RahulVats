# Predicting California Wildfire Likelihood Using Historical Weather and Fire Data

## Overview

This project explores whether we can **predict the likelihood of wildfires** in different locations across California using **historical wildfire records and weather data**.

The main goal is to support better **resource planning and risk awareness** by providing an estimate of how likely a fire is under given conditions.

---

## Problem Statement

Given historical data about **weather** and **wildfire activity** in California, can we build a machine learning model that predicts whether a wildfire will occur in a particular county and month?

---

## Data Summary

The dataset covers records from **2008 to 2020** and includes:

- **10,988 total entries** across all months between 2008–2020  
- Multiple entries for months where more than one fire occurred in the same county  
- **4,279 entries with a recorded fire**  
- **6,709 entries without a fire**

Each row in the dataset describes a single (county, month, year) combination along with associated weather and fire information.

### Feature Descriptions

| Feature        | Type   | Description                                                                 |
|----------------|--------|-----------------------------------------------------------------------------|
| `date`         | object | Month and year of the record                                               |
| `county`       | object | County in which the record is located                                      |
| `maxtempF`     | float  | Average maximum temperature (°F) for the month                             |
| `mintempF`     | float  | Average minimum temperature (°F) for the month                             |
| `avgtempF`     | float  | Average of daily average temperatures (°F) for the month                   |
| `totalSnow`    | float  | Total snow for the month                                                   |
| `humid`        | float  | Average humidity for the month                                             |
| `wind`         | float  | Average wind speed for the month                                           |
| `precip`       | float  | Total precipitation for the month                                          |
| `q_avgtempF`   | float  | Quarterly average temperature (°F)                                         |
| `q_avghumid`   | float  | Quarterly average humidity                                                 |
| `q_sumprecip`  | float  | Quarterly total precipitation                                              |
| `sunHour`      | float  | Average hours of sun per day in the month                                  |
| `FIRE_NAME`    | object | Name of the fire (if any)                                                  |
| `CAUSE`        | float  | Encoded cause of the fire                                                  |
| `lat`          | float  | Latitude coordinate of the county center                                   |
| `long`         | float  | Longitude coordinate of the county center                                  |
| `GIS_ACRES`    | float  | Total number of acres burned (if a fire occurred)                          |

### Target Variable

The primary target is a **binary classification label** indicating whether a fire occurred for that (county, month, year) combination.

---

## Modeling Approach

We treat the problem as a **supervised classification task**:

- **Inputs (features):** weather metrics, quarterly aggregates, and location information  
- **Output (target):** fire occurred (yes/no)

Several machine learning models are evaluated, including:

- **Logistic Regression** – a linear model that outputs probabilities
- **K-Nearest Neighbors (KNN) Classifier**
- **Random Forest Classifier** – an ensemble of decision trees
- **Voting Classifier** – combines predictions from multiple base models

The models are trained and evaluated on the historical dataset to see how well they can distinguish between months with fires and months without fires.

---

## Model Performance (High-Level)

A range of models and hyperparameters were tested. Two key metrics are highlighted for each main model:

| Model Type              | Metric      | Score |
|-------------------------|------------|-------|
| Logistic Regression     | Accuracy   | 76%   |
| Logistic Regression     | Precision  | 67%   |
| Random Forest Classifier| Accuracy   | 88%   |
| Random Forest Classifier| Precision  | 84%   |
| Voting Classifier       | Accuracy   | 87%   |
| Voting Classifier       | Recall     | 86%   |

- **Accuracy** measures the overall fraction of correct predictions.  
- **Precision** tells us, out of all predicted fires, how many were actual fires.  
- **Recall** tells us, out of all actual fires, how many were correctly predicted as fires.

The **Random Forest Classifier** achieved the highest accuracy and precision among the evaluated models, while the **Voting Classifier** provided a strong balance with high recall.

---

## Interpretation & Trade-Offs

Because wildfires are high-impact events, we often care more about **catching as many fires as possible** (high recall), even if this means occasionally predicting fires that do not actually occur (lower precision).

- A model with **higher recall** is useful when missing a fire is very costly.  
- A model with **higher precision** limits false alarms but may miss some real fire events.

The best choice of model depends on how stakeholders balance:

- **Available resources** (personnel, equipment, budget)  
- **Tolerance for false positives** vs **false negatives**  

In many risk-management scenarios, a model that slightly over-predicts fires can still be valuable if it helps allocate resources to high-risk areas in advance.

---

## How to Use This Project

1. **Explore the data** – understand distributions, correlations, and patterns in weather and fire history.  
2. **Run the modeling notebooks** – train and evaluate the different classification models.  
3. **Compare performance metrics** – decide which model best fits your desired trade-off between precision and recall.  
4. **Extend the work** –
   - Add new features (e.g., drought indices, vegetation type)  
   - Try additional models or tuning strategies  
   - Explore calibration of model probabilities for decision support

---

## Future Directions

Potential next steps for enhancing this work include:

- Incorporating **finer-grained spatial data** (e.g., higher-resolution coordinates or land cover types)  
- Using **time-series models** to better capture temporal dependencies  
- Exploring **explainable AI** techniques (e.g., SHAP values) to understand which features drive risk predictions  
- Integrating the model into a **dashboard or alerting system** for operational use

---

This README provides a high-level description of the project, dataset, and modeling approach, without referencing any external collaborators or third-party work. For full details, please refer to the associated notebooks and code in this repository.
