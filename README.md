# Credit-Card-Fraud-Detection

Our task is to solve the Problem given in Kaggle as [Credit Card Fraud Detection](https://www.kaggle.com/mlg-ulb/creditcardfraud)

## DataSet Overview

The datasets contain credit card transactions of European cardholders over a two day collection period in September 2013.
This dataset contains 30 features which include details of time and amount of the transaction and other 28 features from V1, V2,.. to V28 are the result of PCA transformation (this is done because of confidentiality). And another feature 'Class' is an indication that transaction is fraudulent or not.

**Feature Description:**
- Time: the seconds elapsed between each transaction and the first transaction in the dataset
- Amount: the Transaction Amount
- V1, V2,..., V28: numerical input variables which are the output of PCA transformation
- Class: the response variable, it takes two values as 1 - for fraud and 0 - otherwise

>**Objective:** Here our task is to find out given transaction is it fraudulent or not? 

This is binary classification problem. 
This dataset has 492 frauds out of 284,807 transactions means we have uneven distribution as fraud class accounts for only 0.172% of all transactions. Thus this *dataset is highly **unbalanced***.

![Image](readmeImage/dataset.png)

## Evaluation

Due to class imbalance ratio, accuracy is measured using the Area Under the **Precision-Recall Curve(AUPRC)**.
> Confusion matrix accuracy is not meaningful for unbalanced classification.

## Approaches

### 1. Logistic Regression

**code:** `Logistic_Regression.py`

### 2. SVM (Support Vector Machine)

**code:** `SVM.py`

### 3. Neural Network

**code:** `NeuralNet.py`

### 4. Random Forest

**code:** `Random_Forest.py`


## Dependencies

	python3.5
	numpy
	sklearn
	pandas
	imblearn
