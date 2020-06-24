# STAT-642-Final Project
# Prediction-of-Customer-Churn-in-Banking-Industry
#### Drexel University, Department of Decision Science

# Abstract
> _With the growing competition in banking industry, banks are required to follow customer retention strategies while they are trying to increase their market share by acquiring new customers. Current descript was developed using R to compare the performance of six supervised classification techniques to suggest an efficient model to predict customer churn in banking industry, given 10 demographic and personal attributes from 10000 customers of European banks. The effect of feature selection, class imbalance, and outliers were discussed for ANN and random forest as the two competing models. As shown, unlike random forest, ANN doesn’t reveal any serious concern regarding overfitting and is also robust to noise. Therefore, ANN structure with five nodes in a single hidden layer is recognized as the best performing classifier._

# Data
Dataset selected for this study is publically available in the following link: https://www.kaggle.com/barelydedicated/bank-customer-churn-modeling.
Out of 13 variables, CustomerId and Surname need to be removed as they don’t have any contribution to the classification purpose. We also replace binary values of the outcome variable (Exited) with “Stayed” and “Left” labels to have a better representation of outputs when visualizing results and discussing the performance. We will use data entirely in the analysis and don’t follow any sampling procedure because we need the training sample to be sufficiently large.

Current data doesn’t have any missing value in none of its 10000 observations and thus, there won’t be any concern in this regard. However, customers who stayed with banks (7963 customers) are around four times the number of those who left (2037 customers). Therefore, data is imbalance with respect to the outcome variable and this concern needs to be addressed in the modeling section.

# Outline 
- Data Pre-Processing
- Exploratory Data Analysis
- Analysis
    - Model Selection
    - Feature Selection
    - Class Imbalance
    - Effect of Outliers
 - Discussion
 
# Required Libraries
