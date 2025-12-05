# Cardiff AI Talk – Wildfires & Mental Health Runbook

This repo contains **Jupyter notebooks**, **synthetic datasets**, and a **concept cheat sheet** used in the talk:

> **“AI in Action: Bridging Theory and Practice – Wildfires & Mental Health”**

The goal is to help you see how **AI** and **Machine Learning (ML)** can tackle **real-world problems** using data.

---

## 1. Project Overview

We explore **two real-world style use cases**:

### 🔥 Wildfire Risk

- Use **weather** and **land** data (temperature, humidity, wind, rain, vegetation, population, month, year).
- Train models to **predict where wildfires are likely to happen** and how severe they might be.

### 🧠 Mental Health via Mobile Sensing

- Use **phone and behavior data** (steps, sleep, time at home, calls, texts, screen time).
- Train models to **estimate depression risk** and flag **high-risk weeks**.

Across both use cases, we follow the same pattern:

> **Use data from the past → train a model to spot patterns → use that model to predict the future.**

---

## 2. Quick Start

1. Make sure you can run **Python** and **Jupyter Notebooks** (e.g., Anaconda, VS Code, or Google Colab).
2. Clone or download this repo.
3. Open the notebooks in this order:
   1. `00_environment_setup.ipynb` – tools ready  
   2. `10_wildfire_data_and_eda.ipynb` – understand wildfire data  
   3. `20_wildfire_ml_models.ipynb` – build wildfire models  
   4. `30_mental_health_data_and_eda.ipynb` – understand mental health data  
   5. `40_mental_health_ml_models.ipynb` – build mental health models  
   6. `50_model_evaluation_and_explainability.ipynb` – compare & explain models  
   7. `60_serving_and_automation.ipynb` – use models as a service  

---

## 3. Concept Glossary (Plain English)

**AI (Artificial Intelligence)**  
Computers doing tasks that normally need human intelligence (recognizing patterns, making predictions).

**ML (Machine Learning)**  
A part of AI where computers **learn from examples** instead of being given explicit rules.

**Dataset**  
A big table of examples.  
Each row = one example (one week, one location, one patient, etc.).

**Feature**  
An input column used to make a prediction.  
- Wildfires: temperature, humidity, wind speed, rain, vegetation, population, month, year.  
- Mental health: steps, time at home, sleep hours, screen time, etc.

**Label / Target**  
What we want the model to predict.  
- Wildfires: `fire_occurred` (yes/no), `burned_area`.  
- Mental health: `depression_score`, `high_risk` (yes/no).

**Model**  
A mathematical formula/program that uses **features** to predict the **label**.

**Training a Model**  
Feed the model many labeled examples from the past so it can **learn the relationship** between features and labels.

**Testing a Model**  
Give the trained model new examples it has never seen to check **how accurate** its predictions are.

**Regression**  
ML task where we **predict a number**.  
Example: predict `depression_score` (0–27).

**Classification**  
ML task where we **predict a category**.  
Examples:  
- `fire_occurred` = 0/1  
- `high_risk` = yes/no  

In this project we use **both**:
- Wildfires: classification (fire vs no fire).  
- Mental health: regression (score) + classification (high risk vs low risk).

**Neural Network**  
A more complex ML model inspired by the brain, great for images and sequences.  
Example: **CNNs (Convolutional Neural Networks)** for detecting smoke/flames in wildfire photos.  
In these notebooks we use **simpler models** (Logistic Regression, Random Forest, Gradient Boosting), but the idea—learning from examples—is the same.

**ROC Curve**  
A graph showing how well a classification model separates positive vs negative cases across different thresholds.

**AUC (Area Under the ROC Curve)**  
A single number summarizing how good the ROC curve is.  
Closer to 1.0 = better.

**SHAP**  
A tool to **explain model predictions** by showing which features push the prediction up or down.

**Git**  
A tool that tracks changes to code over time (**version control**).

**GitHub**  
A website where Git projects are stored and shared (used to host and share these notebooks and code).

---

## 4. Tools & Libraries

### Python

Main programming language used for all AI/ML code in this project.

### Jupyter Notebook

A “digital notebook” for Python where you can:

- **Write** code  
- **Run** it  
- **See** results (tables, charts) immediately below the code  

You will open and run files like:

- `00_environment_setup.ipynb`  
- `10_wildfire_data_and_eda.ipynb`  
- etc.

### Pandas

Python library for working with **tabular data** (like Excel, but in code).

Used to:

- Read CSV files (e.g., `wildfire_synthetic.csv`, `mental_health_mobile_sensing_synthetic.csv`).  
- Filter rows, select columns, compute averages, counts, etc.

### NumPy

Python library for **numbers and arrays**.

- Used for fast math operations that ML models rely on.

### Scikit-Learn (sklearn)

Python library for **machine learning**.

- Split data into training and test sets.  
- Train models (logistic regression, random forest, gradient boosting).  
- Measure performance:  
  - Classification → Accuracy, Precision, Recall, F1, ROC-AUC  
  - Regression → MAE, RMSE, R²  

---

## 5. Notebook Guide (Step-by-Step Story)

### 00_environment_setup.ipynb – “Get tools ready”

- Imports Python libraries (NumPy, Pandas, Matplotlib, Scikit-Learn).
- Ensures the environment and project structure (`data/raw`, `notebooks`, `models`) are set up.

---

### 10_wildfire_data_and_eda.ipynb – “Understand wildfire data”

- Uses `wildfire_synthetic.csv`.  
- Loads data with Pandas.  
- Shows first rows and summary statistics.  
- Checks **class balance** for `fire_occurred` (how many 1s vs 0s).  
- Plots simple charts (e.g., temperature vs fire occurrence).

**Purpose:**  
Understand the data before training any model.

---

### 20_wildfire_ml_models.ipynb – “Build wildfire prediction models”

- Features: `temp_c`, `humidity`, `wind_speed`, `rain_mm_last_7d`, `vegetation_index`, `population_density`, `month`, `year`.  
- Uses `train_test_split` (Scikit-Learn) → training set + test set.  
- Trains:
  - **Logistic Regression** (simple, interpretable).  
  - **Random Forest** (more flexible).  
- Evaluates with: Accuracy, Precision, Recall, F1, ROC-AUC.  
- Plots **feature importances** for Random Forest.

**Story:**  
“We feed weather + land features into Python and learn to predict fire risk.”

---

### 30_mental_health_data_and_eda.ipynb – “Understand mental health data”

- Uses `mental_health_mobile_sensing_synthetic.csv`.  
- Loads behavior data (steps, distance, time at home, calls/texts, sleep, screen time) with Pandas.  
- Shows distributions of `depression_score` and `high_risk`.  
- Plots relationships (e.g., steps vs depression score).

**Story:**  
“We convert weekly phone/behavior logs into numbers and see how they relate to mental health scores.”

---

### 40_mental_health_ml_models.ipynb – “Build mental health prediction models”

- Features: steps, distance, time at home, calls/texts, sleep, screen time, etc.  
- Splits data into training/test sets.  
- Trains:
  - **Gradient Boosting Regressor** → predicts `depression_score` (regression).  
  - **Random Forest Classifier** → predicts `high_risk` (classification).  
- Evaluates:
  - Regression → MAE, RMSE, R².  
  - Classification → Accuracy, Precision, Recall, F1, ROC-AUC.  
- Plots feature importances to see which behaviors are most linked to risk.

**Story:**  
“Given weekly behavior, can we guess mental health scores and high-risk weeks?”

---

### 50_model_evaluation_and_explainability.ipynb – “Compare and explain models”

- Trains quick Random Forest models for:
  - Wildfire (`fire_occurred`)  
  - Mental health (`high_risk`)  
- Plots **ROC curves** for both; computes **AUC**.  
- Optionally uses **SHAP** for feature-level explanations.

**Story:**  
“We check how strong the models are and explain why they predict what they do.”

---

### 60_serving_and_automation.ipynb – “Use models in the real world”

- Trains a wildfire model again.  
- Does **batch scoring**: adds a `risk_score` column to all rows.  
- Defines `predict_fire_risk(...)` function in Python that returns a **probability of fire** given new conditions.  
- Can be called from web apps, dashboards, or mobile apps.

**Story:**  
“The model leaves the lab and becomes a tool other systems can call.”

---

## 6. Learning Roadmap

If you are new to AI/ML (e.g., high school or early undergrad), follow this path:

1. **Understand the story first**
   - Wildfire risk prediction.
   - Mental health risk prediction from mobile data.

2. **Then follow the notebook order**
   1. `00_environment_setup.ipynb` – tools ready  
   2. `10_wildfire_data_and_eda.ipynb` – understand wildfire data  
   3. `20_wildfire_ml_models.ipynb` – build wildfire models  
   4. `30_mental_health_data_and_eda.ipynb` – understand mental health data  
   5. `40_mental_health_ml_models.ipynb` – build mental health models  
   6. `50_model_evaluation_and_explainability.ipynb` – compare & explain  
   7. `60_serving_and_automation.ipynb` – deploy models as a service  

3. **Keep this README handy**  
   - Use the glossary as your **keyword dictionary** during the talk and while exploring the code.

# Atrificial Intelligence

AI is defined as ability of a machine to perform cognitive functions we associate with human minds, such as perceiving, reasoning, learning, interacting with the environment, problem solving, and even exercising creativity. Examples of technologies that enable AI to solve business problems are robotics and autonomous vehicles, computer vision, language, virtual agents, and machine learning.

![simple definition of AI](https://github.com/Jean-njoroge/Machine-Learning-Resources/blob/master/Images/ai-machine-learning-deep-learning-1.jpg)

Source [Kdnuggets](https://www.kdnuggets.com/2017/07/rapidminer-ai-machine-learning-deep-learning.html)



## Machine-Learning
Most recent advances in AI have been achieved by applying machine learning to very large data sets. Machine-learning algorithms detect patterns and learn how to make predictions and recommendations by processing data and experiences, rather than by receiving explicit programming instruction. The algorithms also adapt in response to new data and experiences to improve efficacy over time.

This repository is a knowledge and learning hub that contains all resources relating to machine learning


### [Types of Machine Learnin](https://machinelearningmastery.com/a-tour-of-machine-learning-algorithms/)

| Ml-Approach| Description | Notebooks |
| --- | --- | --- |
| [Supervised Learning Machine Learning ](https://github.com/Jean-njoroge/Machine-Learning-Resources/tree/master/supervised_learning) | information on Supervised learning approaches |
| Un-Supervised Learning | Show file differences that haven't been staged |
| Semi-Supervised Learning | Show file differences that haven't been staged |
 
 ## Machine Learning Tools
 
 #### General Purpose Machine Learning

* [scikit-learn](https://scikit-learn.org/stable/)- Machine learning in Python
* [cuML](https://github.com/rapidsai/cuml)-RAPIDS Machine Learning Library.
* [Dask](https://dask.org/)-Flexible library for parallel computing in Python.


## Deep Learning Frameworks
* [Tensorflow](https://www.tensorflow.org/)TensorFlow is an end-to-end open source platform for machine learning
* [PyTorch](https://pytorch.org/)-Open source ML and DL framework
* [Chainer](https://chainer.org/) - A Powerful, Flexible, and Intuitive Framework for Neural Networks. Supports CUDA computation & requires a few lines of code of GPU


* [Regenerative Models](https://blog.prolego.io/the-ai-canvas-7a8717cddbe9)

_____
# Data-Driven Use Cases
* [DataStories](https://dataprophet.com/) - A.I. tool that within 30 minutes explains how you can understand, predict, and drive your business KPIs 
* [DataProphet](https://datastories.com/) -AI firm that enables manufacturers
* [Delta Analytics](http://www.deltanalytics.org/) - providing free data consulting and data services & build technical capacity in communities around the world
* [datascope systems](https://datascopesystems.com/) -supplying a range of innovative products for the construction industry
* [Datascope analytics](https://datascopeanalytics.com/) -Combines data, design, and machine learning to build intelligent products and services that improve people's lives
* [Ravel law](https://home.ravellaw.com/) - legal analytics and research company founded in 2012
* [https://datasaur.ai/](intuitive interface for all your Natural Language Processing related tasks)
* [Top VC Virms](https://growthlist.co/blog/ai-vc)
* [The Rise of AI-powered company in post crisis world](https://www.bcg.com/en-us/publications/2020/business-applications-artificial-intelligence-post-covid)-Article
* [ML in Cyber-Security](https://github.com/jivoi/awesome-ml-for-cybersecurity#-datasets)
* [AI In Security]( https://pelotoninnovations.com/process/)

# ML IN Production

* https://elvissaravia.substack.com/p/my-recommendations-to-learn-machine
