---
title: Deposit Subscriptions Predictions
emoji: 📈
sdk: streamlit
app_file: Code/Model_Deployment/prototype.py
pinned: false
tags:
  - streamlit
  - banking
  - ml
license: apache-2.0
---
# Deposit Subscriptions Predictions Project

This is a graduate course-level research project completed by Emily Au, Alex Mak, and Zheng En Than in MATH 509 (Data Structures and Platforms) at the University of Alberta. This project strives to predict whether bank clients will subscribe to term deposit subscriptions through tree-based machine-learning classifier models (Decision Tree, Random Forest, and XGBoost).

## 1. Project Task

- We utilize tree-based machine-learning models to predict whether a client will subscribe to a term deposit through direct marketing campaigns.

## 2. Project Objective

1. Identifying the significant factors influencing a potential client's decision to subscribe to a term deposit
2. Determine the predictive accuracy of our classifier models in forecasting subscription outcomes
3. Observe the predictive performance impact of utilizing bagging and boosting techniques on tree-based machine-learning models

## 3. Project Structure

### Code

- Entire codebase of the project (including data preprocessing, feature engineering, predictive modeling, model evaluation, and data visualization).
- The previous versions of the codebase are also stored.

### Data:

- The dataset used in this project, both the raw and processed dataset.
- Bank Marketing dataset from UCI (UC Irvine) machine learning repository (https://archive.ics.uci.edu/dataset/222/bank+marketing).

### Model:

- The fitted Model and their corresponding parameters after being trained in this project.

### Report

- The finalized report of our project.
- The legacy version of the report is also stored.

### Visualisations

- The visualizations generated from Python (matplotlib and seaborn), and Tableau.
- An influential presetnation to convey our findings and insights

## 4. Project Overview

We have conducted the following steps in our project:

1. Data Preprocessing
   <br> (data cleaning and transformation, anomaly detection analysis, exploratory data analysis)
2. Feature Engineering
   <br> (feature importance, feature selection)
3. Statistical Machine learning Model Development
   <br>(model training and fitting, model evaluation, model optimization, model prediction)
4. Data Visualization
   <br> (within and between models)

## 5. Project Key Insights

- The most important features are: last contact duration, outcome of the previous marketing campaign, and day of year.
- Bagging and boosting bring performance improvement from the Decision Tree for this specific problem and dataset.
- Numerical Results:
  <br>

| Model         | Training Accuracy | Testing Accuracy | Tuning Combinations | Compuation Time |
| ------------- | ----------------- | ---------------- | ------------------- | --------------- |
| Decision Tree | 86.76%            | 89.04%           | 2592                | ~ 10 Minutes    |
| Random Forest | 91.49%            | 90.22%           | 1024                | ~ 20 Minutes    |
| XGBoost       | 92.38%            | 91.00%           | 576                 | ~ 40 Minutes    |

## 6. Model Deployment

- The optimized models implemented in this project are deployed in a streamlit web application!
- Please clone this repo, then go to Code --> Model_Deployment, and enter the folloiwng command:

```bash
streamlit run Deployment_Codebase.py
```

The following screenshots are what the app looks like when it's deployed.

- Initialization

![Screenshot 1](Code/Model_Deployment/Visualizations/Screenshot.png)

- Successful Prediction

![Screenshot 2](Code/Model_Deployment/Visualizations/success_prediction.png)

- Failed Prediction

![Screenshot 3](Code/Model_Deployment/Visualizations/failed_prediction.png)

## 7. Project Critique

- Ensemble methods (in Random forest and XGBoost) can be more complex than Decision Tree, making it challenging to interpret the reasoning behind each prediction.
- Limited generalizability as the dataset consists of data from a Portuguese bank and its specific marketing approach.

## 8. Further Improvements & Investigation

- We would like to re-examine this project with a different dataset, where it may come from another bank in the world with a different telemarketing campaign.
- We are interested in further optimizing our tree-based machine learning models, but that also comes with the drawback of consuming additional computational resources.
- We are looking forward to implementing gradient-boosted random forest (GBRF), which incorporates both bagging and boosting in a tree-based model. We can analyze the impact of using both bagging and boosting compared to just one of them at a time in Random Forest and Decision Tree.
- We would conduct more in-depth analysis, such as exploring any temporal patterns or clustering the data based on client demographics to provide deeper insights into customer behavior, ultimately helping banks devise more effective targeted marketing strategies.

## If you are interested to know more about our project, please feel free to visit our report to see our work in detail! 
