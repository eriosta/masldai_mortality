import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb
import shap
from sklearn.metrics import roc_auc_score, confusion_matrix, roc_curve
from MLstatkit.stats import Delong_test
from lifelines import CoxPHFitter

class NHANES3Predictor:
    """
    NHANES3Predictor is a class for predicting mortality using NHANES III dataset and a pre-trained XGBoost model.

    Methods:
    - __init__(self, model_path, data_path, output_folder='FibroX'): Initializes the predictor with model and data paths, and sets up the output folder.
    - load_model(self): Loads the pre-trained model from the specified path.
    - load_data(self): Loads and preprocesses the data from the specified path.
    - prepare_output_folder(self): Creates the output folder if it does not exist.
    - predict(self): Generates predictions using the loaded model and preprocessed data.
    - plot_auroc(self, df, model_predictions, fib4_scores, save_path='auroc_probabilities.png'): Plots the AUROC for mortality labels and FIB4 scores.

    Usage:
    predictor = NHANES3Predictor(model_path='path/to/model.pkl', data_path='path/to/data.csv')
    predictions = predictor.predict()
    predictor.plot_auroc(df=predictor.df, model_predictions=predictions, fib4_scores=predictor.df['FIB4'])
    """
    def __init__(self, model_path, data_path, output_folder='FibroX'):
        self.model_path = model_path
        self.data_path = data_path
        self.output_folder = output_folder
        self.model = self.load_model()
        self.df = self.load_data()
        self.features = [
            'Age (years)',
            'Glycohemoglobin (%)',
            'Alanine aminotransferase (U/L)',
            'Aspartate aminotransferase (U/L)',
            'Platelet count (1000 cells/µL)',
            'Body-mass index (kg/m**2)',
            'GFR'
        ]
        self.thresholds = {
            '90': 0.29537,
            '95': 0.39327,
            '99': 0.76443
        }
        self.prepare_output_folder()

    def load_model(self):
        with open(self.model_path, 'rb') as file:
            return pickle.load(file)

    def load_data(self):
        df = pd.read_csv(self.data_path).dropna(subset=['mortstat', 'FIB4'])
        df_subset = df[['SEQN', 'HSAGEIR', 'GHP', 'ATPSI', 'ASPSI', 'PLP', 'BMPBMI', 'GFR', 'mortstat', 'ucod_leading', 'permth_exm', 'FIB4','HSSEX']]
        df_subset = df_subset[df_subset['ucod_leading'] != 4].rename(columns={
            'HSAGEIR': 'Age (years)',
            'GHP': 'Glycohemoglobin (%)',
            'ATPSI': 'Alanine aminotransferase (U/L)',
            'ASPSI': 'Aspartate aminotransferase (U/L)',
            'PLP': 'Platelet count (1000 cells/µL)',
            'BMPBMI': 'Body-mass index (kg/m**2)',
            'HSSEX': 'Sex'
        }).set_index(['SEQN', 'mortstat', 'ucod_leading', 'permth_exm', 'FIB4', 'Sex']).dropna()
        return df_subset

    def prepare_output_folder(self):
        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)

    def predict(self):
        data_dmatrix = xgb.DMatrix(self.df[self.features])
        return self.model.predict(data_dmatrix)

    def plot_auroc(self, df, model_predictions, fib4_scores, save_path=None):
        if save_path is None:
            save_path = os.path.join(self.output_folder, 'auroc_probabilities.png')
        """Plot AUROC for mortality labels and for FIB4 scores."""
        mortality_labels = ['mortstat', 'is_cardiac_mortality']
        label_titles = {
            'mortstat': 'All-cause mortality',
            'is_cardiac_mortality': 'Cardiovascular-related mortality'
        }
        plt.figure(figsize=(10, 5))

        # Plot AUROC for model predictions
        for i, label in enumerate(mortality_labels, start=1):
            # Calculate ROC curve and AUROC score for model predictions
            fpr, tpr, _ = roc_curve(df[label], model_predictions)
            auc = roc_auc_score(df[label], model_predictions)

            # Plot the ROC curve
            plt.subplot(1, len(mortality_labels), i)
            plt.plot(fpr, tpr, label=f'FibroX = {auc:.2f}', color='blue')
            plt.plot([0, 1], [0, 1], 'k--')
            plt.xlabel('1-Specificity')
            plt.ylabel('Sensitivity')
            plt.title(label_titles[label])
            plt.legend(loc='lower right')

        # Plot AUROC for FIB4 scores
        for i, label in enumerate(mortality_labels, start=1):
            # Calculate ROC curve and AUROC score for FIB4 scores
            fpr_fib4, tpr_fib4, _ = roc_curve(df[label], fib4_scores)
            auc_fib4 = roc_auc_score(df[label], fib4_scores)

            plt.subplot(1, len(mortality_labels), i)
            plt.plot(fpr_fib4, tpr_fib4, label=f'FIB-4 = {auc_fib4:.2f}', color='orange')
            plt.plot([0, 1], [0, 1], 'k--')
            plt.xlabel('Sensitivity')
            plt.ylabel('1-Specificity')
            plt.title(label_titles[label])
            plt.legend(loc='lower right')

        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.show()

    def calculate_metrics(self, true_labels, predictions, probabilities, label_name):
        auroc = roc_auc_score(true_labels, probabilities)
        threshold_auroc = roc_auc_score(true_labels, predictions)

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

    def perform_delong_test(self, true_labels, probabilities, comparisons):
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

    def generate_shap_plots(self, model, df_subset, features, y_pred_proba, threshold, output_folder):
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
        final_df.to_csv(os.path.join(output_folder, 'final_df_with_shap.csv'), index=False)

        # SHAP Summary Plot with Dots
        plt.figure(figsize=(10, 8))
        shap.summary_plot(shap_values, df_subset[features], plot_type="dot", show=False)
        plt.title(f"FibroX-{threshold} (NHANES 3 External Cohort)")
        plt.savefig(os.path.join(output_folder, "shap_summary_plot.png"))
        plt.close()

    def run(self):
        y_pred_proba = self.predict()
        df_subset_reset = self.df.reset_index()
        df_subset_reset['is_cardiac_mortality'] = ((df_subset_reset['ucod_leading'] == 1) | 
                                                   (df_subset_reset['ucod_leading'] == 5)).astype(int)
        df_subset_reset['is_renal_mortality'] = (df_subset_reset['ucod_leading'] == 9).astype(int)

        self.plot_auroc(df_subset_reset, y_pred_proba, df_subset_reset['FIB4'], save_path=os.path.join(self.output_folder, 'auroc_probabilities.png'))

        for threshold_key, threshold_value in self.thresholds.items():
            threshold = threshold_value

            self.df['Prediction'] = (y_pred_proba >= threshold).astype(int)
            self.df['Probability'] = y_pred_proba

            df_subset_reset = self.df.reset_index()
            df_subset_reset['is_cardiac_mortality'] = ((df_subset_reset['ucod_leading'] == 1) | 
                                                       (df_subset_reset['ucod_leading'] == 5)).astype(int)

            results = []
            df_subset_reset['FIB4_high_risk_1.3'] = (df_subset_reset['FIB4'] >= 1.3).astype(int)
            df_subset_reset['FIB4_high_risk_2.67'] = (df_subset_reset['FIB4'] >= 2.67).astype(int)

            true_labels = ['mortstat', 'is_cardiac_mortality']
            for true_label in true_labels:
                metrics = self.calculate_metrics(
                    df_subset_reset[true_label],
                    df_subset_reset['Prediction'],
                    df_subset_reset['Probability'],
                    'Prediction'
                )
                metrics['True_Label'] = true_label
                results.append(metrics)

            for true_label in true_labels:
                metrics = self.calculate_metrics(
                    df_subset_reset[true_label],
                    df_subset_reset['FIB4_high_risk_1.3'],
                    df_subset_reset['FIB4'],  # Use FIB4 scores for AUROC calculation
                    'FIB4 >= 1.3'
                )
                metrics['True_Label'] = true_label
                results.append(metrics)

            for true_label in true_labels:
                metrics = self.calculate_metrics(
                    df_subset_reset[true_label],
                    df_subset_reset['FIB4_high_risk_2.67'],
                    df_subset_reset['FIB4'],  # Use FIB4 scores for AUROC calculation
                    'FIB4 >= 2.67'
                )
                metrics['True_Label'] = true_label
                results.append(metrics)

            metrics_df = pd.DataFrame(results)

            true_labels = df_subset_reset[['mortstat', 'is_cardiac_mortality']].values
            probabilities = {
                'FIB4 >= 1.3': df_subset_reset['FIB4_high_risk_1.3'].values,
                'FIB4 >= 2.67': df_subset_reset['FIB4_high_risk_2.67'].values,
                'XGB': df_subset_reset['Probability'].values
            }

            comparisons = [
                ('FIB4 >= 1.3', 'XGB'),
                ('FIB4 >= 2.67', 'XGB')
            ]

            all_results = []
            for i, true_label in enumerate(true_labels.T):
                results = self.perform_delong_test(true_label, probabilities, comparisons)
                all_results.append(results)

            delong_df = pd.concat(all_results, keys=['mortstat', 'is_cardiac_mortality'])

            # Under Label, replace Prediction with 'FibroX-95'
            # Under True_Label, replace Mortstat with 'All-cause mortality' and Is_cardiac_mortality with 'Cardiovascular-related mortality'    
            metrics_df.loc[metrics_df['Label'] == 'Prediction', 'Label'] = f'FibroX-{threshold_key}'
            metrics_df.loc[metrics_df['True_Label'] == 'mortstat', 'True_Label'] = 'All-cause mortality'
            metrics_df.loc[metrics_df['True_Label'] == 'is_cardiac_mortality', 'True_Label'] = 'Cardiovascular-related mortality'

            # Under Model_B, replace XGB with 'FibroX-95'
            # Reset the index and replace mortstat with All-cause mortality and is_cardiac_mortality with Cardiovascular-related mortality
            delong_df.loc[delong_df['Model_B'] == 'XGB', 'Model_B'] = f'FibroX-{threshold_key}'
            delong_df = delong_df.reset_index()
            delong_df.loc[delong_df['level_0'] == 'mortstat', 'level_0'] = 'All-cause mortality'
            delong_df.loc[delong_df['level_0'] == 'is_cardiac_mortality', 'level_0'] = 'Cardiovascular-related mortality'

            output_folder = os.path.join(self.output_folder, f'FibroX_{threshold_key}')
            os.makedirs(output_folder, exist_ok=True)

            metrics_df.to_csv(os.path.join(output_folder, 'metrics.csv'), index=False)
            delong_df.to_csv(os.path.join(output_folder, 'delong.csv'), index=False)

            self.generate_shap_plots(self.model, df_subset_reset, self.features, y_pred_proba, threshold=threshold_value, output_folder=output_folder)

    def perform_hr_analysis(self, follow_up_years, adjusted=True):
        df = self.df.reset_index()
        df['time'] = df['permth_exm'].clip(upper=12*follow_up_years) / 12  # Convert months to years

        # Create binary indicators for FIB4 categories
        df['isfib4mod'] = (df['FIB4'] >= 1.3).astype(int)
        df['isfib4high'] = (df['FIB4'] >= 2.67).astype(int)

        # Create binary indicators for mortality types
        df['is_cardiac_mortality'] = df['ucod_leading'].isin([1, 5]).astype(int)
        df['is_malignancy_mortality'] = df['ucod_leading'].isin([2]).astype(int)

        # Create binary columns for FibroX thresholds
        for threshold_key in self.thresholds.keys():
            df[f'Prediction_{threshold_key}_spec'] = (df['Probability'] >= self.thresholds[threshold_key]).astype(int)

        def fit_cox_model(df, event_column, covariates):
            df['event'] = df[event_column]
            cph = CoxPHFitter()
            try:
                cph.fit(df[covariates + ['time', 'event']], duration_col='time', event_col='event')
                summary_df = cph.summary
                summary_df['mortality_type'] = event_column
                summary_df['covariates'] = summary_df.index
                return summary_df
            except Exception as e:
                print(f"Skipping {event_column} due to error: {e}")
                return None

        mortality_columns = ['mortstat', 'is_cardiac_mortality']
        summary_list = []

        if adjusted:
            covariate_sets = [
                ['Prediction_90_spec', 'Age (years)', 'Sex'],
                ['Prediction_95_spec', 'Age (years)', 'Sex'],
                ['Prediction_99_spec', 'Age (years)', 'Sex'],
                ['isfib4mod', 'Age (years)', 'Sex'],
                ['isfib4high', 'Age (years)', 'Sex']
            ]
        else:
            covariate_sets = [
                ['Prediction_90_spec'],
                ['Prediction_95_spec'],
                ['Prediction_99_spec'],
                ['isfib4mod'],
                ['isfib4high']
            ]

        for covariates in covariate_sets:
            for mortality in mortality_columns:
                summary = fit_cox_model(df, mortality, covariates)
                if summary is not None:
                    summary_list.append(summary)

        final_summary_df = pd.concat(summary_list, ignore_index=True) if summary_list else pd.DataFrame()

        final_summary_df['mortality_type'] = final_summary_df['mortality_type'].replace({
            'mortstat': 'All-cause mortality',
            'is_cardiac_mortality': 'Cardiovascular-related mortality'})

        final_summary_df['covariates'] = final_summary_df['covariates'].replace({
            'isfib4mod': 'FIB-4 ≥1.3',
            'isfib4high': 'FIB-4 ≥2.67'
        })

        for threshold_key in self.thresholds.keys():
            final_summary_df['covariates'] = final_summary_df['covariates'].replace({
                f'Prediction_{threshold_key}_spec': f'FibroX-{threshold_key}'
            })

        final_summary_df['p<0.05'] = final_summary_df['p'].apply(lambda x: '*' if x < 0.05 else '')

        final_summary_df = final_summary_df.rename(columns={
            'covariates': 'Model',
            'mortality_type': 'Mortality Type',
            'exp(coef)': 'HR',
            'exp(coef) lower 95%': 'Lower CI',
            'exp(coef) upper 95%': 'Upper CI',
            'p': 'p-value'
        })

        final_summary_df = final_summary_df[['Model', 'Mortality Type', 'HR', 'Lower CI', 'Upper CI', 'p-value', 'p<0.05']]

        return final_summary_df

    def hr_analysis_10_year(self, adjusted=True):
        return self.perform_hr_analysis(10, adjusted)

    def hr_analysis_20_year(self, adjusted=True):
        return self.perform_hr_analysis(20, adjusted)

    def hr_analysis_30_year(self, adjusted=True):
        return self.perform_hr_analysis(30, adjusted)

if __name__ == "__main__":
    predictor = NHANES3Predictor(
        model_path="xgboost_model_11_10_2024.pkl",
        data_path="nhanes3/nhanes3_masld_mortality.csv",
        output_folder="FibroX"
    )
    predictor.run()
    # Print the HR analysis results for 10, 20, and 30 years
    print("HR analysis results for 10 years:")
    hr_10_year = predictor.hr_analysis_10_year(adjusted=True)
    print(hr_10_year)

    print("\nHR analysis results for 20 years:")
    hr_20_year = predictor.hr_analysis_20_year(adjusted=True)
    print(hr_20_year)

    print("\nHR analysis results for 30 years:")
    hr_30_year = predictor.hr_analysis_30_year(adjusted=True)
    print(hr_30_year)
