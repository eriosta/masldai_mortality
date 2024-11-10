import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
import pickle

def train_logistic_regression_model(data_path):

    """
    Loads data, converts date column to datetime, trains a logistic regression model, and evaluates its performance.

    Parameters:
    data_path (str): The path to the CSV file containing the data.

    Returns:
    None
    """
    data = pd.read_csv(data_path).dropna(subset='isF3')
    data = data[['isF3', 
                 'Age_in_years_at_screening', 
                  'Glycohemoglobin (%)', 
                  'Alanine Aminotransferase (ALT) (U/L)', 
                  'Aspartate Aminotransferase (AST) (U/L)', 
                  'Platelet count (1000 cells/uL)',
                  'Body Mass Index (kg/m**2)'
                  ]]

    X = data.drop(columns='isF3')
    y = data['isF3']

    # Set fixed parameters for the Logistic Regression model
    model = LogisticRegression(class_weight='balanced', C=0.01, max_iter=1000, 
                               solver='newton-cg', verbose=1)
    model.fit(X, y)  # Train on all data

    with open('logistic_regression_model.pkl', 'wb') as file:
        pickle.dump(model, file)

train_logistic_regression_model('masld_f3_n_1373.csv')
