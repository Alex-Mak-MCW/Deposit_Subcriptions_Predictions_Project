# Deposit Subscriptions Predictions Project

This is a graduate course-level research project completed by Emily Au, Alex Mak, and Zheng En Than in MATH 509 (Data Structures and Platforms) at the University of Alberta. This project strives to predict whether bank clients will subscribe to term deposit subscriptions through tree-based machine-learning classifier models (Decision Tree, Random Forest, and XGBoost).

## 1. Project Task
* Utilize tree-based machine-learning models to predict whether a client will subscribe to a term deposit through direct marketing campaigns.


## 2. Project Objective
1. The significant factors influencing a potential client's decision to subscribe to a term deposit
2. The predictive accuracy of our classifier models in forecasting subscription outcomes
3. The predictive performance impact of utilizing bagging and boosting techniques on tree-based machine-learning models


## 3. Project Structure

### Backup_Report:
* Unmodified version of the report.

### Codebase: 
* Entire codebase of the project (data preprocessing, feature engineering, predictive modeling, model evaluation, and data visualization).

### Data:
* The dataset used in this project, both the raw and processed dataset.
* Bank Marketing dataset from UCI (UC Irvine) machine learning repository (https://archive.ics.uci.edu/dataset/222/bank+marketing).

### Legacy Codebase:
* The previous versions of the codebase used in this project.

### Model:
* The fitted Model and their corresponding parameters after being trained in this project.

### Tableau Visualziations
* Visualizations conducted within Tableau.

### Visualisations
* Visualizations conducted within Python and its corresponding library (matplotlib and seaborn).

### Final Project Report
* The finalized report of our project.


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
* The most important features are: last contact duration, outcome of the previous marketing campaign, and day of year.
* Bagging and boosting bring performance improvement from the Decision Tree for this specific problem and dataset.
* Numerical Results:
<br>

| Model         | Training Accuracy | Testing Accuracy | Tuning Combinations | Compuation Time |
| ------------- | ----------------- | ---------------- | ------------------- | --------------- |
| Decision Tree | 89.95%            | 89.98%           | 2592                | ~ 10 Minutes    |
| Random Forest | 91.88%            | 90.44%           | 1024                | ~ 20 Minutes    |
| XGBoost       | 92.47%            | 91.17%           | 576                 | ~ 40 Minutes    |


## 6. Project Critique
*  Ensemble methods (in Random forest and XGBoost) can be more complex than Decision Tree, making it challenging to interpret the reasoning behind each prediction.
*  Limited generalizability as the dataset consists of data from a Portuguese bank and its specific marketing approach. 


## 7.  Further Improvements & Investigation
* We would like to re-examine this project with a different dataset, where it may come from another bank in the world with a different telemarketing campaign.
* We are interested in further optimizing our tree-based machine learning models, but that also comes with the drawback of consuming additional computational resources.
* We are looking forward to implementing gradient-boosted random forest (GBRF), which incorporates both bagging and boosting in a tree-based model. We can analyze the impact of using both bagging and boosting compared to just one of them at a time in Random Forest and Decision Tree.
* We would conduct more in-depth analysis, such as exploring any temporal patterns or clustering the data based on client demographics to provide deeper insights into customer behavior, ultimately helping banks devise more effective targeted marketing strategies.


## If you are interested to know more about our project, please feel free to visit our report to see our work in detail!
