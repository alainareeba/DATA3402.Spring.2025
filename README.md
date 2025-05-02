
![UTA-DataScience-Logo](https://github.com/user-attachments/assets/5721ef99-b9ce-4729-84d5-ad9f4afbda5f)

# DATA 3402 - Python for Data Science 2 - Spring 2025 - UTA

This repository is forked from [https://github.com/UTA-DataScience/DATA3402.Spring.2025] and contains my labs assigned by our professor. Below is a quick overview of what each lab covers. Below, you'll find a report on our end-of-semester Tabular Project. 

Lab 1: Downloading and using Unix
Lab 2: Create a Tic Tac Toe Game
Lab 3: Monte Carlo Simulation
Lab 4: Object Oriented Programming (Shapes)
Lab 5: Create a Simple Linear Algebra System (Matrix Representation)
Lab 6: Create a Black Jack Game
Lab 7: Data Analysis on the SUSY Dataset
Lab 8: Data Analysis on the SUSY Dataset Part 2


# Diabetes Risk Classification Using Health Indicators

* **One Sentence Summary** This repository has an attempt to apply machine learning methodology to the CDC's Diabetes Health Indicators to indicate diabetes risk classification (dataset provided by UCI Machine Learning Repository: [https://archive.ics.uci.edu/dataset/891/cdc+diabetes+health+indicators]). 

## Overview

The task was to determine if a patient with any given indicators such as BMI, cardiac history, high cholesterol, etc., is at risk for diabetes or not. The approach to this problem and using this dataset was to set it up as classification task. I used Random Forest as the main model with all of the features as input. I compared the performance of 3 models: Random Forest, XGBoost, and KNN. I also compared the baseline model a class imbalance issue and the other three models with the class imbalance solved. Our best model had accuracy score of 79% and weighted recall of 79%. 

## Summary of Workdone

### Data

* Data:
    * Input: The type of input is a directly forked tabular dataset from the UCI Machine Learning Repository, replicating a CSV file of features, output: tabular dataset with all feature names and description, roles, data type, demographic, and any missing values. There were 22 columns and 253,680 rows. I used the raw dataset with all instances. I used 177,576 patients for training, 38, 052 for testing, and 38,052 for validation. 

#### Preprocessing / Clean up

* For the preprocessing, I handled the outliers of the numerical columns by removing any data points that had a Z-score less than 3. After removing the outliers, I standardized the numerical columns using a standard scaler so that our columns would have a mean close to 0 and standard deviation to 1 for any models that might need a standardized dataset. After that, I ensured that all of the numerical entries were integers instead of floats in order to prevent errors in our models.  

#### Data Visualization
<img width="905" alt="Screenshot 2025-05-02 at 9 16 40 AM" src="https://github.com/user-attachments/assets/1a522b9c-7e55-44a0-80f2-2413ff0c55ba" />

<img width="912" alt="Screenshot 2025-05-02 at 9 18 05 AM" src="https://github.com/user-attachments/assets/fadecbea-6f25-4650-ae43-fd4522d45208" />

<img width="880" alt="Screenshot 2025-05-02 at 9 20 47 AM" src="https://github.com/user-attachments/assets/c9ddb144-88fe-4132-a611-7837fcaeac44" />

The distribution in these histograms is slightly skewed in these examples. These histograms also show a clear imbalance as there are more non-diabetic instances than diabetic, which is something to keep in mind when running our models. 

### Problem Formulation
To classify diabetes risk, certain indicators have to be high enough or present (such as a history) to consider the risk. This can be challenging with mulitple factors playing upon each other. I had to use models that would consider all features and underlying patterns, which is why I wanted to use ensemble methods to detect these patterns. 

### Training 

For model training, several algorithms were used, including Random Forest, XGBoost, and K-Nearest Neighbors (KNN). Our attempt for XGBoost raised convergence warnings suggesting more iterations. This was related to inconsistent input types handling rather than the algorithm itself, which was handled early on, but still raised an error. Specifically, KNN required more attention as it classifies samples based on the majority class of the k closest samples in the feature space. This method is sensitive to feature scaling, so the data was standardized using StandardScaler.To keep training efficient, a testing set of the data could be used to quickly iterate on model performance. In this case, the full dataset was used but stratified training/test splitting to make sure the target class was not missed. SMOTE was applied to balance the testing set due to class imbalance in the Diabetes_binary target aftering seeing the support on our baseline model. Models were then trained on the resampled (balanced) training data.


### Performance Comparison

<img width="537" alt="Screenshot 2025-05-02 at 9 45 57 AM" src="https://github.com/user-attachments/assets/a54ce6be-1ff8-4f8d-a371-06a30b259e6e" />


Baseline Model Evalution Metrics 

<img width="793" alt="Screenshot 2025-05-02 at 9 46 40 AM" src="https://github.com/user-attachments/assets/0a6e71de-a374-4132-8d95-30cc489c2909" />


Baseline Model ROC Curve 

<img width="1003" alt="Screenshot 2025-05-02 at 9 47 43 AM" src="https://github.com/user-attachments/assets/8bf9d8a6-ee69-4f4f-8635-6c7190219bd4" />


Decision Function Histogram 

<img width="512" alt="Screenshot 2025-05-02 at 9 54 11 AM" src="https://github.com/user-attachments/assets/56cd3baa-a82d-4887-9679-43ac555faf61" />


SMOTE Random Forest Evaluation Metrics 

<img width="851" alt="Screenshot 2025-05-02 at 9 54 59 AM" src="https://github.com/user-attachments/assets/d94d3338-4e43-40d4-ae11-5b008802a406" />


SMOTE Random Forest ROC Curve (slightly different)

<img width="461" alt="Screenshot 2025-05-02 at 9 52 44 AM" src="https://github.com/user-attachments/assets/0aa2005c-7dc3-42f7-973f-78ec263cbc6f" />


Model Comparison Table (Only focused on recall as false negatives are a priority)

Our key metric for our model was recall and accuracy. Recall prioritizes identify true positives, minimizing false negatives (as those are the most detrimental in the medical field). Accuracy shows the total proportion of correct predictions. Our ROC Curves 

### Conclusions

 Random Forest with SMOTE has the highest scores relatively for both classes, considering the class imbalance solution. But our baseline had the highest without addressing the class imbalance, meaning that it was really only classifying non-diabetic patients really well. SMOTE is a good way to oversample, but there is nuance to this technique as it might not always apply to the dataset properly. Other models surprisingly performed poorly considering the class imbalance solution and their nature in classification problem contexts. 

### Future Work

The next thing I would try is a different synthetic sampling technique to help the support. I would also try to reduce the features based on feature significance to see if that's why the other models. I would also try to add a new class: pre-diabetes and define the boundaries between non-diabetic, pre-diabetic, and diabetic to make the model more efficient in healthcare settings.  Some other studies that could be applied to this dataset is what is the threshold between non-diabetes and diabetes, considering A1Cs or having a study of diabetes risk specifically for the less common indicators such as income and education. 

## How to reproduce results

### Overview of variables used

* df - our raw dataset
* df_no_outliers - dataset with outliers removed from numerical columns
* df_scaled - dataset with scaled numerical columns 
  
### Software Setup
The UCI Machine Learning Repository must be installed in your notebook in order to fetch the dataset directly from their repository. 
1. pip install ucimlrepo
2. from ucimlrepo import fetch_ucirepo
 cdc_diabetes_health_indicators = fetch_ucirepo(id=891)

X = cdc_diabetes_health_indicators.data.features
y = cdc_diabetes_health_indicators.data.targets

print(cdc_diabetes_health_indicators.metadata)

print(cdc_diabetes_health_indicators.variables)

These directions are also directly on the website when clicking Import in Python: https://archive.ics.uci.edu/dataset/891/cdc+diabetes+health+indicators

### Training

To perform training, make sure no instances are continuous and that the variables are consistent with their data type listed at the beginning. If there are continuous variables that are supposed to be integers or floats that are supposed to be integers, convert all numerical columns to integers. Remove the ID column as well so it doesn't interfere.  Set X equal to your feature list (ENSURE YOU ARE USING THE PREPROCESSED VARIABLES) and y equal to your risk (yes or no) list and then split the training and the testing set using the 70-30 method (70% is used for training and 30% is used for testing). Use X_temp as a temporary version of the dataset as to not change anything with the regular dataset. If using SMOTE, make sure to apply SMOTE to your X and Y training and test sets in order to have it apply properly to the class distribution

#### Performance Evaluation

To evaluate performance, you can use a simple confusion matrix or a decision function histogram to see how well your model is identifying the amount true positive cases vs false negatives and how well it is making decisions. 


## Citations

* Centers for Disease Control and Prevention. (2022, April 29). CDC - 2014 BRFSS survey data and Documentation. Centers for Disease Control and Prevention. https://www.cdc.gov/brfss/annual_data/annual_2014.html
* Centers for Disease Control and Prevention. (2024, May 15). Diabetes Risk Factors. Diabetes. https://www.cdc.gov/diabetes/risk-factors/index.html







