
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

The task was to determine if a patient with any given indicators such as BMI, cardiac history, high cholesterol, etc., is at risk for diabetes or not. The approach to this problem and using this dataset was to set it up as classification task. I used Random Forest as the main model with all of the features as input. I compared the performance of 2 Random Forest models: one with a class imbalance issue and one with the class imbalance solved. Our best model had accuracy score of 79% and weighted recall of 79%. According to Kaggle, the

## Summary of Workdone

### Data

* Data:
    * Input: The type of input is a directly forked tabular dataset from the UCI Machine Learning Repository, replicating a CSV file of features, output: tabular dataset with all feature names and description, roles, data type, demographic, and any missing values. There were 22 columns and 253,680 rows. I used the raw dataset with all instances. I used 177,576 patients for training, 38, 052 for testing, and 38,052 for validation. 

#### Preprocessing / Clean up

* For the preprocessing, I handled the outliers of the numerical columns by removing any data points that had a Z-score less than 3. After removing the outliers, I standardized the numerical columns using a standard scaler so that our columns would have a mean close to 0 and standard deviation to 1 for any models that might need a standardized dataset. After that, I ensured that all of the numerical entries were integers instead of floats in order to prevent errors in our models.  

#### Data Visualization
<img width="905" alt="Screenshot 2025-05-02 at 9 16 40 AM" src="https://github.com/user-attachments/assets/1a522b9c-7e55-44a0-80f2-2413ff0c55ba" />

<img width="883" alt="Screenshot 2025-05-02 at 9 17 41 AM" src="https://github.com/user-attachments/assets/40fbcc88-1a0d-429e-89b3-6575e0312239" />

<img width="912" alt="Screenshot 2025-05-02 at 9 18 05 AM" src="https://github.com/user-attachments/assets/fadecbea-6f25-4650-ae43-fd4522d45208" />



Show a few visualization of the data and say a few words about what you see.

### Problem Formulation

* Define:
  * Input / Output
  * Models
    * Describe the different models you tried and why.
  * Loss, Optimizer, other Hyperparameters.

### Training

* Describe the training:
  * How you trained: software and hardware.
  * How did training take.
  * Training curves (loss vs epoch for test/train).
  * How did you decide to stop training.
  * Any difficulties? How did you resolve them?

### Performance Comparison

* Clearly define the key performance metric(s).
* Show/compare results in one table.
* Show one (or few) visualization(s) of results, for example ROC curves.

### Conclusions

* State any conclusions you can infer from your work. Example: LSTM work better than GRU.

### Future Work

* What would be the next thing that you would try.
* What are some other studies that can be done starting from here.

## How to reproduce results

* In this section, provide instructions at least one of the following:
   * Reproduce your results fully, including training.
   * Apply this package to other data. For example, how to use the model you trained.
   * Use this package to perform their own study.
* Also describe what resources to use for this package, if appropirate. For example, point them to Collab and TPUs.

### Overview of files in repository

* Describe the directory structure, if any.
* List all relavent files and describe their role in the package.
* An example:
  * utils.py: various functions that are used in cleaning and visualizing data.
  * preprocess.ipynb: Takes input data in CSV and writes out data frame after cleanup.
  * visualization.ipynb: Creates various visualizations of the data.
  * models.py: Contains functions that build the various models.
  * training-model-1.ipynb: Trains the first model and saves model during training.
  * training-model-2.ipynb: Trains the second model and saves model during training.
  * training-model-3.ipynb: Trains the third model and saves model during training.
  * performance.ipynb: loads multiple trained models and compares results.
  * inference.ipynb: loads a trained model and applies it to test data to create kaggle submission.

* Note that all of these notebooks should contain enough text for someone to understand what is happening.

### Software Setup
* List all of the required packages.
* If not standard, provide or point to instruction for installing the packages.
* Describe how to install your package.

### Data

* Point to where they can download the data.
* Lead them through preprocessing steps, if necessary.

### Training

* Describe how to train the model

#### Performance Evaluation

* Describe how to run the performance evaluation.


## Citations

* Provide any references.







