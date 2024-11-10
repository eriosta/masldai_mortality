import numpy as np
import pandas as pd
import xgboost as xgb
import pickle
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score, confusion_matrix

def train_xgboost_model(data_path, n_splits=5):

    """
    Loads data, converts date column to datetime, trains an XGBoost model using k-fold cross-validation, and evaluates its performance.

    Parameters:
    data_path (str): The path to the CSV file containing the data.
    n_splits (int): The number of folds for cross-validation.

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
                  'Body Mass Index (kg/m**2)',
                  'GFR'
                  ]]

    X = data.drop(columns='isF3')
    y = data['isF3']

    # Set fixed parameters for the XGBoost model
    params = {
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'scale_pos_weight': len(y[y == 0]) / len(y[y == 1]),  # Adjust for class imbalance
        'max_depth': 5,
        'eta': 0.1
    }

    # Initialize KFold
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    models = []
    performance_metrics = []

    for fold, (train_index, val_index) in enumerate(kf.split(X), 1):
        X_train, X_val = X.iloc[train_index], X.iloc[val_index]
        y_train, y_val = y.iloc[train_index], y.iloc[val_index]

        # Convert the dataset into DMatrix format for XGBoost
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dval = xgb.DMatrix(X_val, label=y_val)

        # Train the model
        model = xgb.train(params, dtrain, num_boost_round=100)

        # Evaluate the model on the validation set
        val_pred = model.predict(dval)
        val_pred_binary = (val_pred >= 0.5).astype(int)

        # Calculate performance metrics
        auroc = roc_auc_score(y_val, val_pred)
        tn, fp, fn, tp = confusion_matrix(y_val, val_pred_binary).ravel()
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
        npv = tn / (tn + fn) if (tn + fn) > 0 else 0

        performance_metrics.append({
            'Fold': fold,
            'AUROC': auroc,
            'Sensitivity': sensitivity,
            'Specificity': specificity,
            'PPV': ppv,
            'NPV': npv
        })

        models.append(model)

    # Create a DataFrame for performance metrics
    metrics_df = pd.DataFrame(performance_metrics)
    metrics_df.loc[len(metrics_df)] = metrics_df.mean().astype(float)  # Add average metrics at the end
    metrics_df.at[len(metrics_df) - 1, 'Fold'] = 'Average'  # Set the last row's Fold label to 'Average'

    # # Save the last trained model (or you could save all models)
    # with open('xgboost_model.pkl', 'wb') as file:
    #     pickle.dump(models[-1], file)

    # # Optionally save the metrics DataFrame
    # metrics_df.to_csv('k_fold_performance_metrics.csv', index=False)

train_xgboost_model('masld_f3_n_1373.csv')
