import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb
import shap
from sklearn.metrics import roc_auc_score, confusion_matrix, roc_curve
from MLstatkit.stats import Delong_test

df = pd.read_csv("nhanes3/nhanes3_masld_mortality.csv").dropna(subset=['mortstat', 'FIB4'])
df_subset = df[['SEQN', 'ATPSI', 'PLP', 'HSAGEIR', 'ASPSI', 'GHP','GFR','BMPBMI', 'mortstat', 'ucod_leading', 'permth_exm', 'FIB4']]

df_subset = df_subset[df_subset['ucod_leading'] != 4].rename(columns={
    
    'HSAGEIR': 'Age (years)',
    'GHP': 'Glycohemoglobin (%)', 
    'ATPSI': 'Alanine aminotransferase (U/L)',
    'ASPSI': 'Aspartate aminotransferase (U/L)',
    'PLP': 'Platelet count (1000 cells/µL)',
    'BMPBMI': 'Body-mass index (kg/m**2)',
    
}).set_index(['SEQN', 'mortstat', 'ucod_leading', 'permth_exm', 'FIB4']).dropna()

with open("xgboost_model_11_10_2024.pkl", 'rb') as file:
    model = pickle.load(file)

features = ['Age (years)',
            'Glycohemoglobin (%)',
            'Alanine aminotransferase (U/L)',
            'Aspartate aminotransferase (U/L)',
            'Platelet count (1000 cells/µL)',
            'Body-mass index (kg/m**2)',
            'GFR']

data_dmatrix = xgb.DMatrix(df_subset[features])
y_pred_proba = model.predict(data_dmatrix)


df_subset_reset = df_subset.reset_index()
df_subset_reset['is_cardiac_mortality'] = ((df_subset_reset['ucod_leading'] == 1) | 
                                            (df_subset_reset['ucod_leading'] == 5)).astype(int)
df_subset_reset['is_renal_mortality'] = (df_subset_reset['ucod_leading'] == 9).astype(int)


# Update the plot_auroc function to accept predictions from both the model and actual FIB4 scores
def plot_auroc(df, model_predictions, fib4_scores, save_path='auroc_probabilities.png'):
    """Plot AUROC for mortality labels and for FIB4 scores."""
    mortality_labels = ['mortstat', 'is_cardiac_mortality', 'is_renal_mortality']
    plt.figure(figsize=(15, 5))

    # Plot AUROC for model predictions
    for i, label in enumerate(mortality_labels, start=1):
        # Calculate ROC curve and AUROC score for model predictions
        fpr, tpr, _ = roc_curve(df[label], model_predictions)
        auc = roc_auc_score(df[label], model_predictions)

        # Plot the ROC curve
        plt.subplot(1, len(mortality_labels), i)
        plt.plot(fpr, tpr, label=f'AUROC (Model) = {auc:.2f}', color='blue')
        plt.plot([0, 1], [0, 1], 'k--', label='Random')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'AUROC for {label} (Model)')
        plt.legend(loc='lower right')

    # Plot AUROC for FIB4 scores
    for i, label in enumerate(mortality_labels, start=1):
        # Calculate ROC curve and AUROC score for FIB4 scores
        fpr_fib4, tpr_fib4, _ = roc_curve(df[label], fib4_scores)
        auc_fib4 = roc_auc_score(df[label], fib4_scores)

        plt.subplot(1, len(mortality_labels), i)
        plt.plot(fpr_fib4, tpr_fib4, label=f'AUROC (FIB4) = {auc_fib4:.2f}', color='orange')
        plt.plot([0, 1], [0, 1], 'k--', label='Random')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'AUROC for {label} (FIB4)')
        plt.legend(loc='lower right')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.show()


# Call the updated plot_auroc function with your model predictions and actual FIB4 scores
plot_auroc(df_subset_reset, y_pred_proba, df_subset_reset['FIB4'], save_path='auroc_probabilities_11_10_2024.png')

# Before 11/10/2024
threshold_90_spec = 0.32834
threshold_95_spec = 0.42690
threshold_99_spec: 0.64435

# After 11/10/2024
threshold_90_spec = 0.29537
threshold_95_spec = 0.39327
threshold_99_spec = 0.76443



threshold = threshold_95_spec


df_subset['Prediction'] = (y_pred_proba >= threshold).astype(int)
df_subset['Probability'] = y_pred_proba

df_subset_reset = df_subset.reset_index()
df_subset_reset['is_cardiac_mortality'] = ((df_subset_reset['ucod_leading'] == 1) | 
                                            (df_subset_reset['ucod_leading'] == 5)).astype(int)
df_subset_reset['is_renal_mortality'] = (df_subset_reset['ucod_leading'] == 9).astype(int)


# Calculate and export metrics function
def calculate_and_export_metrics(df_subset_reset, comparisons):
    def calculate_metrics(true_labels, predictions, probabilities, label_name):
        # AUROC based on continuous probabilities (for model predictions only)
        auroc = roc_auc_score(true_labels, probabilities)
        
        # Threshold-based AUROC using binary predictions at the threshold
        threshold_auroc = roc_auc_score(true_labels, predictions)

        # Confusion matrix metrics based on thresholded predictions
        tn, fp, fn, tp = confusion_matrix(true_labels, predictions).ravel()
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
        npv = tn / (tn + fn) if (tn + fn) > 0 else 0
        
        return {
            'Label': label_name,
            'AUROC': auroc,
            'Threshold_AUROC': threshold_auroc,
            'Sensitivity': sensitivity,
            'Specificity': specificity,
            'PPV': ppv,
            'NPV': npv
        }

    results = []
    # Create FIB4 binary variable based on the 1.3 cutoff
    df_subset_reset['FIB4_high_risk'] = (df_subset_reset['FIB4'] > 2.67).astype(int)
    
    # Calculate global AUROC for model predictions
    true_labels = ['mortstat', 'is_cardiac_mortality', 'is_renal_mortality']
    for true_label in true_labels:
        # Metrics for model predictions
        metrics = calculate_metrics(
            df_subset_reset[true_label],
            df_subset_reset['Prediction'],
            df_subset_reset['Probability'],
            'Prediction'
        )
        metrics['True_Label'] = true_label
        results.append(metrics)

    # Calculate metrics for FIB4_high_risk
    for true_label in true_labels:
        metrics = calculate_metrics(
            df_subset_reset[true_label],
            df_subset_reset['FIB4_high_risk'],
            df_subset_reset['FIB4'],  # Use FIB4 scores for AUROC calculation
            'FIB4'
        )
        metrics['True_Label'] = true_label
        results.append(metrics)

    metrics_df = pd.DataFrame(results)

    # DeLong Test for AUROC comparisons
    def perform_delong_test(true_labels, probabilities, comparisons):
        results = []
        for model_a, model_b in comparisons:
            z_score, p_value = Delong_test(true_labels, probabilities[model_a], probabilities[model_b])
            results.append({
                'Model_A': model_a,
                'Model_B': model_b,
                'Z-Score': z_score,
                'P-Value': p_value
            })
        return pd.DataFrame(results)

    # Use binary FIB4 predictions and model probabilities for AUROC comparison in DeLong Test
    true_labels = df_subset_reset[['mortstat', 'is_cardiac_mortality', 'is_renal_mortality']].values
    probabilities = {
        'FIB4': df_subset_reset['FIB4_high_risk'].values,
        'XGB': df_subset_reset['Probability'].values
    }

    all_results = []
    for i, true_label in enumerate(true_labels.T):
        results = perform_delong_test(true_label, probabilities, comparisons)
        all_results.append(results)

    delong_df = pd.concat(all_results, keys=['mortstat', 'is_cardiac_mortality', 'is_renal_mortality'])

    return metrics_df, delong_df

    # # Export to Excel
    # with pd.ExcelWriter('metrics_and_delong_results.xlsx') as writer:
    #     metrics_df.to_excel(writer, sheet_name='Metrics', index=False)
    #     delong_df.to_excel(writer, sheet_name='DeLong Test', index=True)

comparisons = [
    ('FIB4', 'XGB')
]

calculate_and_export_metrics(df_subset_reset, comparisons)






def generate_shap_plots(model, df_subset, features, y_pred_proba, threshold=0.5, output_folder='shap_plots'):
    
    # SHAP Explainer
    np.random.seed(42)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(df_subset[features].to_numpy())
    base_values = explainer.expected_value
    
    # Convert SHAP values to a DataFrame and add additional information
    shap_df = pd.DataFrame(shap_values, columns=features)
    shap_df['Total_SHAP'] = shap_df.abs().sum(axis=1)
    shap_df['Prediction'] = (y_pred_proba >= threshold).astype(int)
    shap_df['Probability'] = y_pred_proba
    shap_df['Base_Value'] = base_values
    
    # Concatenate SHAP values with the original data
    final_df = pd.concat([df_subset.reset_index(drop=True)[['mortstat', 'ucod_leading'] + features], shap_df], axis=1)
    os.makedirs(output_folder, exist_ok=True)
    final_df.to_csv(os.path.join(output_folder, 'final_df_with_shap.csv'), index=False)
    
    # SHAP Summary Plot with Dots
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, df_subset[features], plot_type="dot", show=False)
    plt.title("SHAP Summary Plot with Dots")
    plt.savefig(os.path.join(output_folder, "shap_summary_plot.png"))
    plt.close()
    
    # # SHAP Interaction Plot for all feature pairs
    # for i, feature1 in enumerate(features_cleaned):
    #     for feature2 in features_cleaned[i+1:]:  # Avoid duplicating pairs
    #         sanitized_feature1 = sanitize_filename(feature1)
    #         sanitized_feature2 = sanitize_filename(feature2)
    #         plt.figure(figsize=(10, 8))
    #         shap.dependence_plot(feature1, shap_values, df_subset[features_cleaned], interaction_index=feature2, show=False)
    #         plt.title(f"Interaction Plot: {feature1} and {feature2}")
    #         plt.savefig(os.path.join(output_folder, f"shap_interaction_{sanitized_feature1}_{sanitized_feature2}.png"))
    #         plt.close()


generate_shap_plots(model, df_subset_reset, features, y_pred_proba, threshold=threshold, output_folder='shap_plots')




