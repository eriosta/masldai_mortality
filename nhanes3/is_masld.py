import numpy as np
import pandas as pd

def is_masld(
    HSSEX, BMPBMI, BMPWAIST, G1P, GHP, HAD1, HAD6, HAD10,
    PEPMNK1R, PEPMNK5R, HAE5A, TGP, HAE9D, HDP
):
    """
    Determines if a patient meets MASLD criteria based on NHANES III variables.

    Parameters (NHANES III):
        HSSEX: Sex (1 = Male, 2 = Female)
        BMPBMI: Body-mass index
        BMPWAIST: Waist circumference (cm)
        G1P: Plasma glucose (mg/dL)
        GHP: Glycated hemoglobin (HbA1c)
        HAD1: History of diabetes (1 = Yes, 2 = No)
        HAD6: Currently taking insulin (1 = Yes, 2 = No)
        HAD10: Currently taking diabetes pills (1 = Yes, 2 = No)
        PEPMNK1R: Systolic blood pressure
        PEPMNK5R: Diastolic blood pressure
        HAE5A: Taking medication for high blood pressure (1 = Yes, 2 = No)
        TGP: Serum triglycerides (mg/dL)
        HAE9D: Taking cholesterol-lowering medication (1 = Yes, 2 = No)
        HDP: Serum HDL cholesterol (mg/dL)

    Returns:
        dict: Dictionary with MASLD criteria results and individual criteria flags.
    """

    # Criterion 1, body: BMI or waist circumference
    bmi_criteria = BMPBMI >= 25
    wc_criteria = BMPWAIST > (94 if HSSEX == 1 else 80)
    is_body = int(bmi_criteria or wc_criteria)

    # Criterion 2, diabetes: Blood glucose or diabetes history/treatment
    is_diabetes = int(
        G1P >= 100 or
        GHP >= 5.7 or
        HAD1 == 1 or
        HAD6 == 1 or
        HAD10 == 1
    )

    # Criterion 3, hypertension: Blood pressure or antihypertensive treatment
    is_hypertension = int(
        PEPMNK1R >= 130 or
        PEPMNK5R >= 85 or
        HAE5A == 1
    )

    # Criterion 4 and 5, dyslipidemia: Triglycerides, HDL cholesterol, or lipid-lowering treatment
    is_dyslipidemia = int(
        TGP >= 150 or
        HAE9D == 1 or
        (HDP <= 40 if HSSEX == 1 else HDP <= 50) or
        HAE9D == 1
    )

    # MASLD criteria met if at least one of the criteria is true
    is_masld = int(any([is_body, is_diabetes, is_hypertension, is_dyslipidemia]))

    return {
        'is_masld': is_masld,
        'is_body': is_body,
        'is_diabetes': is_diabetes,
        'is_hypertension': is_hypertension,
        'is_dyslipidemia': is_dyslipidemia
    }


def replace_blank_and_dont_know_with_na(data):
    """
    Replaces 'Blank but applicable' and 'Don't know' values with NA (None) in the provided NHANES III variables.

    Parameters:
        data (dict): Dictionary with NHANES III variable names and values.

    Returns:
        dict: Dictionary with 'Blank but applicable' and 'Don't know' values replaced with NA (None).
    """
    # Define the values for 'Blank but applicable' and 'Don't know' by variable
    blank_values = {
        'BMPBMI': 8888,        # Body-mass index
        'BMPWAIST': 88888,     # Waist circumference
        'G1P': 88888,          # Plasma glucose
        'GHP': 8888,           # HbA1c
        'HAD1': [8, 9],        # Diabetes history (8 = Blank, 9 = Don't know)
        'HAD6': 8,             # Insulin treatment (8 = Blank)
        'HAD10': [8, 9],       # Diabetes pills (8 = Blank, 9 = Don't know)
        'PEPMNK1R': 888,       # Systolic blood pressure
        'PEPMNK5R': 888,       # Diastolic blood pressure
        'HAE5A': 8,            # High blood pressure treatment (8 = Blank)
        'TGP': 8888,           # Serum triglycerides
        'HAE9D': [8, 9],       # Cholesterol medication (8 = Blank, 9 = Don't know)
        'HDP': 888,            # Serum HDL cholesterol
        'HAR3': 8,             # Actively smoking 
        'HAR1': 8,             # History of smoking 
        'TCP': 888             # Total serum cholesterol
    }

    # Replace each 'Blank but applicable' or 'Don't know' value with None (NA) if it matches the criteria
    for key, blank_value in blank_values.items():
        if isinstance(blank_value, list):
            # If the blank value is a list (for both Blank and Don't know), check for any match
            if data.get(key) in blank_value:
                data[key] = np.nan
        else:
            # Otherwise, check if the value matches the single blank indicator
            if data.get(key) == blank_value:
                data[key] = np.nan

    return data

def apply_is_masld(row):
    # First, replace blank and don't know values with NA
    row_dict = replace_blank_and_dont_know_with_na(row.to_dict())
    
    # Then apply the is_masld function
    result = is_masld(
        row_dict['HSSEX'], row_dict['BMPBMI'], row_dict['BMPWAIST'],
        row_dict['G1P'], row_dict['GHP'], row_dict['HAD1'],
        row_dict['HAD6'], row_dict['HAD10'], row_dict['PEPMNK1R'],
        row_dict['PEPMNK5R'], row_dict['HAE5A'], row_dict['TGP'],
        row_dict['HAE9D'], row_dict['HDP']
    )
    return pd.Series(result)

# Read the CSV file
df = pd.read_csv('nhanes3/nhanes3.csv')

# Apply the is_masld function to each row
result = df.apply(apply_is_masld, axis=1)
df[['is_masld', 'is_body', 'is_diabetes', 'is_hypertension', 'is_dyslipidemia']] = result

def replace_with_nan(df, column, value):
    df[column] = df[column].replace(value, np.nan)
    return df

# Filter for rows where is_masld is 1
df = df[df['is_masld'] == 1]

# Define columns and their respective values to be replaced with NaN
columns_to_replace = {
    'ASPSI': 888,  # AST
    'ATPSI': 888,  # ALT
    'GGPSI': 8888,  # GGT
    'PLP': 88888,  # platelets
    'HSAGEIR': 888,  # Age
    'BMPBMI': 8888,  # BMI
    'AMP': 888,  # albumin
    'GHP': 8888, # A1c
    'CEP': 8888,
    'BMPWAIST': 88888 
}

for column, value in columns_to_replace.items():
    df = replace_with_nan(df, column, value)

# Function to calculate FIB4
def calculate_fib4(age, ast, alt, platelets):
    if pd.isna(age) or pd.isna(ast) or pd.isna(alt) or pd.isna(platelets) or alt == 0 or platelets == 0:  # Check for NA values and avoid division by zero
        return np.nan
    return (age * ast) / (platelets * np.sqrt(alt))

# Function to calculate NFS
def calculate_nfs(age, bmi, diabetes, ast, alt, platelets, albumin):
    if pd.isna(age) or pd.isna(bmi) or pd.isna(diabetes) or pd.isna(ast) or pd.isna(alt) or pd.isna(platelets) or pd.isna(albumin) or alt == 0:  # Check for NA values and avoid division by zero
        return np.nan
    ast_alt_ratio = ast / alt  # No need to check for alt here since we already checked above
    return (-1.675 + 
            (0.037 * age) + 
            (0.094 * bmi) + 
            (1.13 * diabetes) + 
            (0.99 * ast_alt_ratio) - 
            (0.013 * platelets) - 
            (0.66 * albumin))
# Apply the calculations to the DataFrame
df['FIB4'] = df.apply(lambda row: calculate_fib4(row['HSAGEIR'], row['ASPSI'], row['ATPSI'], row['PLP']), axis=1)
df['NFS'] = df.apply(lambda row: calculate_nfs(row['HSAGEIR'], row['BMPBMI'], row['is_diabetes'], row['ASPSI'], row['ATPSI'], row['PLP'], row['AMP']), axis=1)

# SUBSET BY COMPLETE FIB4 AND NFS SCORE TO COMPARE HEAD TO HEAD
# df = df.dropna(subset=['FIB4', 'NFS'])

# df['is_high_risk_f3_fib4'] = (df['FIB4'] > 2.67).astype(int)
df['is_high_risk_f3_fib4'] = (df['FIB4'] > 1.30).astype(int)
df['is_high_risk_f3_nfs'] = (df['NFS'] > 0.676).astype(int)

def calculate_gfr(serum_cr, age, is_female):
    gfr = 175 * (serum_cr ** -1.154) * (age ** -0.203)
    if is_female:
        gfr *= 0.742
    return gfr

df['is_female'] = (df['HSSEX'] == 2).astype(int)
df['GFR'] = df.apply(lambda row: calculate_gfr(row['CEP'], row['HSAGEIR'], row['is_female']), axis=1)


df = df.apply(lambda row: replace_blank_and_dont_know_with_na(row.to_dict()), axis=1, result_type='expand')


def calculate_framingham_risk_score(row):
    is_female = row['is_female']
    age = row['HSAGEIR']
    total_chol = row['TCP']
    hdl_chol = row['HDP']
    syst_bp = row['PEPMNK1R']
    bp_treated = 0 if pd.isna(row['HAE5A']) else (0 if row['HAE5A'] == 2 else row['HAE5A'])
    smoker = 0 if 'HAR3' in row and pd.isna(row['HAR3']) else (0 if row['HAR3'] == 2 else row['HAR3']) if 'HAR3' in row else (0 if pd.isna(row['HAR1']) else (0 if row['HAR1'] == 2 else row['HAR1']))

    # Check for any NaN values
    if any(pd.isna([age, total_chol, hdl_chol, syst_bp, bp_treated, smoker])):
        return np.nan

    if is_female == 1:
        # Calculate for Women

        if age <= 0 or total_chol <= 0 or hdl_chol <= 0 or syst_bp <= 0:
            return np.nan

        ln_age = np.log(age)
        ln_total_chol = np.log(total_chol)
        ln_hdl_chol = np.log(hdl_chol)
        ln_syst_bp = np.log(syst_bp)

        # Adjust ln_age for smoker interaction
        if age > 78:
            ln_age_smoker = np.log(78) * smoker
        else:
            ln_age_smoker = ln_age * smoker

        LWomen = (
            31.764001 * ln_age +
            22.465206 * ln_total_chol +
            (-1.187731) * ln_hdl_chol +
            2.552905 * ln_syst_bp +
            0.420251 * bp_treated +
            13.07543 * smoker +
            (-5.060998) * ln_age * ln_total_chol +
            (-2.996945) * ln_age_smoker -
            146.5933061
        )

        PWomen = 1 - 0.98767 * np.exp(LWomen)
        return PWomen

    else:
        # Calculate for Men

        if age <= 0 or total_chol <= 0 or hdl_chol <= 0 or syst_bp <= 0:
            return np.nan

        ln_age = np.log(age)
        ln_total_chol = np.log(total_chol)
        ln_hdl_chol = np.log(hdl_chol)
        ln_syst_bp = np.log(syst_bp)

        # Adjust ln_age for smoker interaction
        if age > 70:
            ln_age_smoker = np.log(70) * smoker
        else:
            ln_age_smoker = ln_age * smoker

        # Calculate ln_age squared
        ln_age_sq = ln_age ** 2

        LMen = (
            52.00961 * ln_age +
            20.014077 * ln_total_chol +
            (-0.905964) * ln_hdl_chol +
            1.305784 * ln_syst_bp +
            0.241549 * bp_treated +
            12.096316 * smoker +
            (-4.605038) * ln_age * ln_total_chol +
            (-2.84367) * ln_age_smoker +
            (-2.93323) * ln_age_sq -
            172.300168
        )

        PMen = 1 - 0.9402 * np.exp(LMen)
        return PMen

# Apply the function to create the Framingham Risk Score column
df['Framingham_Risk_Score'] = df.apply(calculate_framingham_risk_score, axis=1)

df[[
    'TCP','HDP','PEPMNK1R','HAR1'
]].info()



df.to_csv('nhanes3/nhanes3_masld.csv', index=False)

mortality_df = pd.read_csv('nhanes3/NHANES_III_2019.csv')
df['seqn'] = df['SEQN'].astype(int)
merge_df = pd.merge(mortality_df, df, on='seqn', how='inner')
merge_df = merge_df[merge_df['eligstat'] != 2]
merge_df.to_csv('nhanes3/nhanes3_masld_mortality.csv')



