import pandas as pd
import pickle
import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
import matplotlib.pyplot as plt
import xgboost as xgb

# Load data and model (assuming you already have the data and model loaded as shown in the previous code)
data_path = 'masld_f3_n_1373.csv'
model_path = 'xgboost_model_f4.pkl'

# Load the data and model
data = pd.read_csv(data_path).dropna(subset=['Age_in_years_at_screening', 'Glycohemoglobin (%)', 
                                             'Alanine Aminotransferase (ALT) (U/L)', 
                                             'Aspartate Aminotransferase (AST) (U/L)', 
                                             'Gamma Glutamyl Transferase (GGT) (IU/L)', 
                                             'Platelet count (1000 cells/uL)', 
                                             'Body Mass Index (kg/m**2)','isF3'])
model = pickle.load(open(model_path, 'rb'))

# Select features and outcome
model_features = ['Age_in_years_at_screening', 'Glycohemoglobin (%)', 
                  'Alanine Aminotransferase (ALT) (U/L)', 
                  'Aspartate Aminotransferase (AST) (U/L)', 
                  'Gamma Glutamyl Transferase (GGT) (IU/L)', 
                  'Platelet count (1000 cells/uL)',
                  'Body Mass Index (kg/m**2)']
data_df = data[model_features]

true_outcome = data['isF3']

prediction_probabilities = model.predict(xgb.DMatrix(data_df))

def plot_calibration_curve(true_outcome, prediction_probabilities, n_bins=10, n_bootstraps=2000, save_path='calibration_curve.png'):
    # Calculate calibration curve with prediction probabilities
    prob_true, prob_pred = calibration_curve(true_outcome, prediction_probabilities, n_bins=n_bins)

    # Calculate intercept and slope of a linear regression on the calibration data
    linear_model = LinearRegression().fit(prob_pred.reshape(-1, 1), prob_true)
    intercept = linear_model.intercept_
    slope = linear_model.coef_[0]

    # Bootstrap method to calculate confidence intervals manually
    intercepts = []
    slopes = []

    for _ in range(n_bootstraps):
        # Sample with replacement
        indices = np.random.choice(range(len(prob_pred)), size=len(prob_pred), replace=True)
        sample_pred = prob_pred[indices]
        sample_true = prob_true[indices]
        
        # Fit linear model on resampled data
        resample_model = LinearRegression().fit(sample_pred.reshape(-1, 1), sample_true)
        intercepts.append(resample_model.intercept_)
        slopes.append(resample_model.coef_[0])

    # Calculate 95% CI for intercept and slope
    intercept_ci = np.percentile(intercepts, [2.5, 97.5])
    slope_ci = np.percentile(slopes, [2.5, 97.5])

    # Apply Loess smoothing to the calibration curve for a flexible fit
    lowess = sm.nonparametric.lowess(prob_true, prob_pred, frac=0.7)

    # Plot calibration curve
    plt.figure(figsize=(6, 5))
    plt.plot([0, 1], [0, 1], 'k-', label='Perfect Calibration')  # Perfect calibration line
    plt.plot(prob_pred, prob_true, 'o', label='Deciles', color='blue', marker='^')  # Deciles as triangles
    plt.plot(lowess[:, 0], lowess[:, 1], 'r--', label='Loess Smoothed Calibration Curve')  # Loess line

    # Shaded area for 95% CI around the Loess line
    plt.fill_between(lowess[:, 0], lowess[:, 1] - np.std(slopes), 
                     lowess[:, 1] + np.std(slopes), color='gray', alpha=0.3, label='95% CI')

    # Display intercept and slope with their CIs in the top left corner
    plt.text(0.05, 0.95, f'Intercept: {intercept:.2f} (95% CI: {intercept_ci[0]:.2f}, {intercept_ci[1]:.2f})\n'
                         f'Slope: {slope:.2f} (95% CI: {slope_ci[0]:.2f}, {slope_ci[1]:.2f})',
             transform=plt.gca().transAxes, fontsize=10, verticalalignment='top', bbox=dict(facecolor='white', alpha=0.5))

    plt.legend(loc='lower right', bbox_to_anchor=(1, 0), 
               labels=['Deciles (Observed Proportion)', 'Loess Smoothed Calibration Curve'])
    plt.xlabel('Prediction Probability')
    plt.ylabel('Observed Proportion (True Outcome)')
    plt.title('Calibration Curve with Prediction Probabilities')

    plt.tight_layout()  # Make layout tight

    plt.savefig(save_path, format='png', dpi=300)  # Save the figure as a PNG file
    # Display the plot
    plt.show()

plot_calibration_curve(true_outcome=true_outcome,prediction_probabilities=prediction_probabilities)
