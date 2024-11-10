import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import shap
import xgboost as xgb

# Load data and model (assuming you already have the data and model loaded as shown in the previous code)
data_path = 'masld_f3_n_1373.csv'
model_path = 'xgboost_model_11_10_2024.pkl'

# Load the data and model
data = pd.read_csv(data_path).dropna(subset=[
                'Age (years)',
                'Glycohemoglobin (%)',
                'Alanine aminotransferase (U/L)',
                'Aspartate aminotransferase (U/L)',
                'Platelet count (1000 cells/µL)',
                'Body-mass index (kg/m**2)',
                'GFR',
                'isF3'])

model = pickle.load(open(model_path, 'rb'))

# Select features and outcome
model_features = [
    'Age (years)',
    'Glycohemoglobin (%)',
    'Alanine aminotransferase (U/L)',
    'Aspartate aminotransferase (U/L)',
    'Platelet count (1000 cells/µL)',
    'Body-mass index (kg/m**2)',
    'GFR',
]

data_df = data[model_features]
true_outcome = data['isF3']  # Replace with the actual column name

# Predict probabilities
y_pred_proba = model.predict(xgb.DMatrix(data_df))


def plot_diagnostic_metrics(y_pred_proba, true_outcome, save_path='cutoff_11_10_2024.png'):
    # Sort prediction probabilities for thresholding
    sorted_pred_proba = np.sort(y_pred_proba)

    # Calculate metrics at each unique prediction probability
    sensitivity = []
    specificity = []
    ppv = []
    npv = []
    screen_failure_rate = []
    missed_cases_rate = []
    proportion_identified = []

    for thresh in sorted_pred_proba:
        # Binarize predictions based on the prediction probability threshold
        y_pred_thresh = (y_pred_proba >= thresh).astype(int)
        
        # Calculate confusion matrix components
        tp = np.sum((y_pred_thresh == 1) & (true_outcome == 1))
        tn = np.sum((y_pred_thresh == 0) & (true_outcome == 0))
        fp = np.sum((y_pred_thresh == 1) & (true_outcome == 0))
        fn = np.sum((y_pred_thresh == 0) & (true_outcome == 1))
        
        # Sensitivity (Recall, TPR)
        sensitivity.append(tp / (tp + fn) if (tp + fn) > 0 else 0)
        # Specificity (TNR)
        specificity.append(tn / (tn + fp) if (tn + fp) > 0 else 0)
        # Positive Predictive Value (PPV)
        ppv.append(tp / (tp + fp) if (tp + fp) > 0 else 0)
        # Negative Predictive Value (NPV)
        npv.append(tn / (tn + fn) if (tn + fn) > 0 else 0)
        
        # Screen Failure Rate (1 - PPV)
        screen_failure_rate.append(1 - ppv[-1] if ppv[-1] > 0 else 1)
        # Missed Cases Rate (1 - Sensitivity)
        missed_cases_rate.append(1 - sensitivity[-1])
        # Proportion of Patients Identified
        proportion_identified.append((tp + fp) / len(true_outcome))

    # Find thresholds for 99%, 95%, 90%, 80%, 70%, 60%, and 50% sensitivity and specificity
    specificity_threshold_99 = sorted_pred_proba[np.argmax(np.array(specificity) >= 0.99)]
    specificity_threshold_95 = sorted_pred_proba[np.argmax(np.array(specificity) >= 0.95)]
    specificity_threshold_90 = sorted_pred_proba[np.argmax(np.array(specificity) >= 0.90)]
    specificity_threshold_85 = sorted_pred_proba[np.argmax(np.array(specificity) >= 0.85)]

    # Print the cutoffs
    print(f'99% Specificity Threshold: {specificity_threshold_99:.5f}')
    print(f'95% Specificity Threshold: {specificity_threshold_95:.5f}')
    print(f'90% Specificity Threshold: {specificity_threshold_90:.5f}')


    # Plot 1: Sensitivity, Specificity, PPV, and NPV by Prediction Probability
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(sorted_pred_proba, sensitivity, color='red', label='Sensitivity')
    plt.plot(sorted_pred_proba, specificity, color='blue', label='Specificity')
    plt.plot(sorted_pred_proba, ppv, color='green', linestyle='--', label='Positive Predictive Value')
    plt.plot(sorted_pred_proba, npv, color='purple', linestyle='--', label='Negative Predictive Value')
    plt.axvline(x=specificity_threshold_90, color='black', linestyle=':', label=f'90% Specificity: {specificity_threshold_90:.5f}')
    plt.axvline(x=specificity_threshold_85, color='orange', linestyle=':', label=f'85% Specificity: {specificity_threshold_85:.5f}')
    plt.axvline(x=specificity_threshold_95, color='grey', linestyle=':', label=f'95% Specificity: {specificity_threshold_95:.5f}')
    plt.axvline(x=specificity_threshold_99, color='yellow', linestyle=':', label=f'99% Specificity: {specificity_threshold_99:.5f}')

    plt.xlabel('Prediction Probability')
    plt.ylabel('Value')
    plt.legend(loc='lower right')
    plt.xlim([0,1])
    plt.ylim([0, 1])

    # Plot 2: Screen Failure Rate, Missed Cases Rate, and Proportion Identified by Prediction Probability
    plt.subplot(1, 2, 2)
    plt.plot(sorted_pred_proba, screen_failure_rate, color='green', label='Screen Failure Rate (1 - PPV)')
    plt.plot(sorted_pred_proba, missed_cases_rate, color='purple', label='Missed Cases Rate (1 - Sensitivity)')
    plt.plot(sorted_pred_proba, proportion_identified, color='blue', linestyle='--', label='Proportion of Patients Identified')
    plt.xlabel('Prediction Probability')
    plt.ylabel('Value')
    plt.legend(loc='upper right')
    plt.xlim([0, 1])
    plt.ylim([0, 1])

    # Save and show the plot
    plt.tight_layout()
    plt.savefig(save_path, format='png', dpi=300)
    plt.show()

plot_diagnostic_metrics(y_pred_proba=y_pred_proba,
                        true_outcome=true_outcome)