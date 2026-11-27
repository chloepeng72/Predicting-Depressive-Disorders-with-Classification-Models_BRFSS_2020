# Predicting Depressive Disorders with Classification Models (BRFSS 2020)

Final project for CS 699 Data Mining, Boston University (2024)

## Objective

The goal for this project was to use different classification models on the 2020 Behavioral Risk Factor Surveillance System (BRFSS) Survey Dataset. Our aim was to identify the most effective model for predicting the occurrence of depressive disorders in individuals based on a wide array of variables.

## Data

2020 BRFSS survey data from the CDC: https://www.cdc.gov/brfss/annual_data/annual_2020.html

## Process

1. **Feature selection:** selected the top variables using Information Gain, Gain Ratio and the Gini index.
2. **Balancing the dataset:** the attribute we are predicting is not balanced, with class 0 having over four times as many occurrences as class 1. As a result, it was necessary to use techniques such as bootstrapping and SMOTE, or the Boruta method, to assist in creating a more balanced dataset.
3. **Modeling:** trained and compared several classification models, including gradient boosting machine (GBM) and XGBoost.

## Results

In conclusion, out of all the models we applied to our dataset, the XGBoost algorithm was the best at accurately predicting depressive disorder in the dataset. Despite having the same metrics as the gradient boosting machine, this algorithm took less time to run, indicating it is less computationally expensive, which gives it an advantage in terms of being the best model for this dataset.

## Tools

- R
- XGBoost

## Reproducibility

We used `set.seed()` so that the random steps (sampling, SMOTE, train/test split and model training) give the same results each time the code is run.


## Files

- `BRFSS.project.R` main analysis code  
- `CS699_Term_Project.pdf` final presentation
