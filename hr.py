import pandas as pd
import pickle
import xgboost as xgb
from lifelines import CoxPHFitter

# Load and preprocess the data
df = pd.read_csv("nhanes3/nhanes3_masld_mortality.csv").dropna(subset=['mortstat', 'FIB4','HSSEX']) #, 'is_hypertension', 'is_dyslipidemia','HAR1'])

df_subset = df[[
    # Features
    'SEQN', 'ATPSI', 'PLP', 'HSAGEIR', 'ASPSI', 'GHP', 
    'BMPBMI', 'mortstat', 'ucod_leading', 'permth_exm', 'FIB4', 'GFR',
    # Confounders
    'is_hypertension', 'is_dyslipidemia','HAR1','HSSEX',
    ]]

# Filter out records where ucod_leading is 4
df_subset = df_subset[df_subset['ucod_leading'] != 4]

# Rename columns to more descriptive names
df_subset = df_subset.rename(columns={
    'HSAGEIR': 'Age_in_years_at_screening', 
    'GHP': 'Glycohemoglobin (%)',
    'ATPSI': 'Alanine Aminotransferase (ALT) (U/L)', 
    'ASPSI': 'Aspartate Aminotransferase (AST) (U/L)',
    'PLP': 'Platelet count (1000 cells/uL)',
    'BMPBMI': 'Body Mass Index (kg/m**2)'
})

index_cols = ['SEQN', 'mortstat', 'ucod_leading', 'permth_exm', 'FIB4',
                'is_hypertension', 'is_dyslipidemia','HAR1','HSSEX']

# Set index and drop missing values
df_subset = df_subset.set_index(index_cols).dropna()


# Load the model
with open("xgboost_model.pkl", 'rb') as file:
    model = pickle.load(file)

# Prepare data for predictions
features = [
    'Age_in_years_at_screening', 
    'Glycohemoglobin (%)', 
    'Alanine Aminotransferase (ALT) (U/L)',
    'Aspartate Aminotransferase (AST) (U/L)', 
    'Platelet count (1000 cells/uL)',
    'Body Mass Index (kg/m**2)',
    'GFR'
]


data_dmatrix = xgb.DMatrix(df_subset[features])
y_pred_proba = model.predict(data_dmatrix)

# Define cutoffs for cardiovascular-related mortality predictions
thresholds = {
    '85_spec': 0.28781,
    '90_spec': 0.32834,
    '95_spec': 0.42690,
    '99_spec': 0.64435
}

# Add predictions and probabilities to the dataframe for each threshold
for key, threshold in thresholds.items():
    df_subset[f'Prediction_{key}'] = (y_pred_proba >= threshold).astype(int)
    df_subset[f'Probability_{key}'] = y_pred_proba

# Reset index for further analysis
df_subset_reset = df_subset.reset_index()



# Create binary indicators for FIB4 categories
df_subset_reset['isfib4mod'] = (df_subset_reset['FIB4'] >= 1.3).astype(int)
df_subset_reset['isfib4high'] = (df_subset_reset['FIB4'] >= 2.67).astype(int)

# Create binary indicators for mortality types
df_subset_reset['is_cardiac_mortality'] = df_subset_reset['ucod_leading'].isin([1, 5]).astype(int)
df_subset_reset['is_malignancy_mortality'] = df_subset_reset['ucod_leading'].isin([2]).astype(int)

def fit_cox_model(df, event_column, covariates, follow_up_months=12):
    df['follow_up_months'] = df['permth_exm']
    df['event'] = df[event_column]
    
    # Filter for less than or equal to follow_up_months
    df = df[df['follow_up_months'] <= follow_up_months]

    cph = CoxPHFitter()
    try:
        cph.fit(df[covariates + ['follow_up_months', 'event']], duration_col='follow_up_months', event_col='event')
        summary_df = cph.summary
        summary_df['mortality_type'] = event_column
        summary_df['covariates'] = summary_df.index
        
        # Add the threshold suffix to the Prediction covariate
        for key in thresholds.keys():
            if f'Prediction_{key}' in covariates:
                summary_df['covariates'] = summary_df['covariates'].str.replace(f'Prediction_{key}', f'Prediction_{key}')
        
        # Add N and N_event columns
        N_event = cph.event_observed.sum()
        N = cph.event_observed.shape[0]
        summary_df['N'] = N
        summary_df['N_event'] = N_event
        
        return summary_df
    except Exception as e:
        print(f"Skipping {event_column} due to error: {e}")
        return None

mortality_columns = ['mortstat', 'is_cardiac_mortality']
covariate_sets = [
    ['Prediction_85_spec', 'Age_in_years_at_screening', 'HSSEX'],
    ['Prediction_90_spec', 'Age_in_years_at_screening', 'HSSEX'],
    ['Prediction_95_spec', 'Age_in_years_at_screening', 'HSSEX'],
    ['Prediction_99_spec', 'Age_in_years_at_screening', 'HSSEX'],
    ['isfib4mod', 'Age_in_years_at_screening', 'HSSEX'],
    ['isfib4high', 'Age_in_years_at_screening', 'HSSEX']
]



summary_list = []
for covariates in covariate_sets:
    for mortality in mortality_columns:
        summary = fit_cox_model(df_subset_reset, mortality, covariates, follow_up_months=12*10)
        if summary is not None:
            summary_list.append(summary)

final_summary_df = pd.concat(summary_list, ignore_index=True) if summary_list else pd.DataFrame()

final_summary_df = final_summary_df[~final_summary_df['covariates'].str.contains('Age_in_years_at_screening|HSSEX|HAR1|is_hypertension|is_dyslipidemia|is_body|is_diabetes')][['exp(coef)', 'exp(coef) lower 95%', 'exp(coef) upper 95%', 'p', 'mortality_type', 'covariates', 'N', 'N_event']]
# final_summary_df = final_summary_df[['exp(coef)', 'exp(coef) lower 95%', 'exp(coef) upper 95%', 'p', 'mortality_type', 'covariates', 'N', 'N_event']]

final_summary_df['mortality_type'] = final_summary_df['mortality_type'].replace({
    'mortstat': 'All-cause mortality',
    'is_cardiac_mortality': 'Cardiovascular-related mortality'})

final_summary_df['covariates'] = final_summary_df['covariates'].replace({
    'Prediction_90_spec': 'FibroX-90',
    'Prediction_95_spec': 'FibroX-95',
    'Prediction_85_spec': 'FibroX-85',
    'Prediction_99_spec': 'FibroX-99',
    'isfib4mod': 'FIB-4 ≥1.3',
    'isfib4high': 'FIB-4 ≥2.67'
})

# Add significance indicator column
final_summary_df['p<0.05'] = final_summary_df['p'].apply(lambda x: '*' if x < 0.05 else '')


final_summary_df = final_summary_df.rename(columns={
    'covariates': 'Model',
    'mortality_type': '10-Year Mortality',
    'N': 'Total',
    'N_event': 'Deceased',
    'exp(coef)': 'HR',
    'exp(coef) lower 95%': 'Lower CI',
    'exp(coef) upper 95%': 'Upper CI'
})

final_summary_df = final_summary_df[['Model', '10-Year Mortality', 'Total', 'Deceased', 'HR', 'Lower CI', 'Upper CI', 'p', 'p<0.05']]


final_summary_df

# final_summary_df.to_csv('age_gender_adjusted_hr_10_year.csv',index=False)

# final_summary_df.to_csv('age_sex_framingham_adjusted_hazard_ratios.csv',index=False)

# Export selected columns to CSV
export_columns = [
    'mortstat', 
    'is_cardiac_mortality',
    'Prediction_95_spec',
    'Age_in_years_at_screening',
    'HSSEX', 
    'isfib4mod',
    'isfib4high',
    'permth_exm'
]

df_mort = df_subset_reset[export_columns]


import pandas as pd
import matplotlib.pyplot as plt
from lifelines import CoxPHFitter

# Load the dataset
df = df_mort

# Cap the time column at 120 months (10 years) and convert to years
df['time'] = df['permth_exm'].clip(upper=12*500) / 12  # Convert months to years

# Prepare the data for Cox regression
df = df[['time', 'mortstat', 'Prediction_95_spec', 'Age_in_years_at_screening', 'HSSEX']].dropna()
df.rename(columns={'Age_in_years_at_screening': 'Age', 'HSSEX': 'Sex'}, inplace=True)

# Fit the Cox Proportional Hazards model
cph = CoxPHFitter()
cph.fit(df, duration_col='time', event_col='mortstat')

# Print the summary of the Cox model
print(cph.summary)

# Check proportional hazards assumption
cph.check_assumptions(df)

# Plot scaled Schoenfeld residuals
cph.plot_partial_effects_on_outcome(covariates='Prediction_95_spec', values=[0, 1], cmap='coolwarm')

# Add plot details
plt.title("Adjusted Cox Proportial Hazards (30-Year Follow-Up)")
plt.xlabel("Years")
plt.ylabel("Survival Probability")
plt.legend(["FibroX <95% Specificity Cutoff (<0.4269)", "FibroX ≥95% Specificity Cutoff (≥0.4269)"])
plt.grid(True)
plt.show()



























import pandas as pd
import matplotlib.pyplot as plt
from lifelines import CoxPHFitter

# Load the dataset
df = df_mort

# Cap the time column at 120 months (10 years) and convert to years
df['time'] = df['permth_exm'].clip(upper=12*500) / 12  # Convert months to years

# Prepare the data for Cox regression
df = df[['time', 'mortstat', 'isfib4mod', 'Age_in_years_at_screening', 'HSSEX']].dropna()
df.rename(columns={'Age_in_years_at_screening': 'Age', 'HSSEX': 'Sex'}, inplace=True)

# Fit the Cox Proportional Hazards model
cph = CoxPHFitter()
cph.fit(df, duration_col='time', event_col='mortstat')

# Print the summary of the Cox model
print(cph.summary)

# Plot the adjusted survival curves by isfib4mod
cph.plot_partial_effects_on_outcome(covariates='isfib4mod', values=[0, 1], cmap='coolwarm')
# Add plot details
plt.title("Adjusted Cox Proportial Hazards (30-Year Follow-Up)")
plt.xlabel("Years")
plt.ylabel("Survival Probability")
plt.legend(["FIB-4 <1.30", "FIB-4 ≥1.30"])
plt.grid(True)
plt.show()


import pandas as pd
import matplotlib.pyplot as plt
from lifelines import CoxPHFitter

# Load the dataset
df = df_mort

# Cap the time column at 120 months (10 years) and convert to years
df['time'] = df['permth_exm'].clip(upper=12*500) / 12  # Convert months to years

# Prepare the data for Cox regression
df = df[['time', 'mortstat', 'isfib4high', 'Age_in_years_at_screening', 'HSSEX']].dropna()
df.rename(columns={'Age_in_years_at_screening': 'Age', 'HSSEX': 'Sex'}, inplace=True)

# Fit the Cox Proportional Hazards model
cph = CoxPHFitter()
cph.fit(df, duration_col='time', event_col='mortstat')

# Print the summary of the Cox model
print(cph.summary)

# Plot the adjusted survival curves by isfib4high
cph.plot_partial_effects_on_outcome(covariates='isfib4high', values=[0, 1], cmap='coolwarm')
# Add plot details
plt.title("Adjusted Cox Proportial Hazards (30-Year Follow-Up)")
plt.xlabel("Years")
plt.ylabel("Survival Probability")
plt.legend(["FIB-4 <2.67", "FIB-4 ≥2.67"])
plt.grid(True)
plt.show()

