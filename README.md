# credit-risk-classification
Module 20

![image](https://github.com/user-attachments/assets/61558e6f-faad-4fe0-9d3e-b6b9cf4f8da2)


## Overview

This historical dataset of lending activity from a peer-to-peer lending services company was used to build a model that can identify the creditworthiness of borrowers. This dataset was imported into a pandas data frame and separated into two variables (x and y). These variables were further split into training data and testing data to allow the models to learn using the training data and then test accuracy and precision with the testing data. Logistic regression models were used with the original data to predict healthy loan and high-risk loan labels. 

## Analysis

The purpose of this analysis was to create a supervised machine learning model that would predict if a loan is healthy or high-risk. The data used was a 77,500 line csv file that contained columns of various data about the loan (amount and interest rate) and the borrower's income, debt and accounts, as well as a column classifying the loan as healthy or high-risk.

The dataset contains lending activity records and serves as the basis for predicting the creditworthiness of borrowers. Two main variables, loan status (y) and all other features (x), were derived from the dataset. The focus of the analysis was on predicting healthy loans and high-risk loans so that decisions can be made around risk management associated with these classifications.

 - To preform the machine learning process the data was:
   1. Split into labels and features, with the loan status (healthy or high-risk) being the label and the remaining seven columns as features.
   2. The data was then split into training and testing data sets.
   3. A Logistic Regression model from sklearn was created with the Limited-memory BFGS ('lbfgs') solver.
      - Logistic Regression was chosen as we are classifying loans as healthy OR high-risk and only those two options.
   4. The model was then fit with the training data.
   5. Predictions were made with the test data
   6. The model was evaluated by comparing the predictions with the test labels.
  
# Results  
The follwing details provide balanced accuracy scores and the precision and recall scores of the machine learning models we used.
 - Machine Learning Model 1:
     Logistic Regression Confusion Matrix

      <img width="215" alt="image" src="https://github.com/user-attachments/assets/9577f14e-e5b6-4652-889b-fd5823edd4c7">

      Logistic Regression Classification Report

      <img width="395" alt="image" src="https://github.com/user-attachments/assets/df69fc8f-c29f-40a1-b61f-0f3c53b73d8a">

  Description of Model 1 Accuracy, Precision, and Recall scores.

Healthy Loans (0) Prediction:
  - Precision: 1.00: The model predicts healthy loans perfectly, indicating that all instances it identifies as healthy loans are correct;
   - Recall: 0.99: The model correctly identifies 99% of the actual healthy loans, missing only a small fraction.

High-Risk Loans (1) Prediction:
  
  - Precision: 0.84: The model correctly predicts 84% of the loans it labels as high-risk. This means there are some false positives, where healthy loans are misclassified as high-risk.
    
  - Recall: 0.94: The model captures 94% of the actual high-risk loans, meaning it is missing 6% of them. This is a strong performance, but there is room for improvement.

Accuracy score:
  
   - Overall this model provides 99% accuracy but due to the data available, that rating is heavily weighted to "healthy" loans.
   

# Summary   

The Logistic Regression model does a very good job predicting healthy loans, For trying to classify loans, trying to prevent high-risk loans is more important than giving out a loan as more money can be lost from a single loan that defaults than the interest earned from a loan.

I recommend using the scikit-learn Logistic Regression model with lbfgs solver as a filter for how to spend time evaluating loan applications. If a future loan is predicted to be "healthy", little time should be spent digging into further detail prior to approving the loan because the likelihood that it will be healthy is 99+%. However, if the model predicts the loan to be "high-risk" and the potential benefit of that loan to the lender is relatively high, more time should be spent evaluating that loan against additional features because 15% of the test loans predicted to be "high-risk" were actually "healthy".

Another way to say it, is to use this model as a first pass to streamline accepting loans but do not depend on it for rejecting loans. For trying to classify loans, trying to prevent "high-risk" loans is more important than giving out a loan as more money can be lost from a single loan that defaults than the interest earned from a loan, spend more time looking into the details based on the value of that loan to the lender prior to rejecting it.

# Repo Notes

# File Notes
  - credit_risk_classification.ipynb contains the code used for this analysis
  - Resources/lending_data.csv contains the data used in this analysis
  - The images folder contains images used in the analysis write-up

# Getting Started

  # Prerequisites
  
To run the jupyter notebook with the solution, you must have Python, jupyter notebook, pathlib, scikit-learn installed in your environment

# Cloning Repo

$ git clone 

$ cd credit-risk-classification

$ jupyter lab
