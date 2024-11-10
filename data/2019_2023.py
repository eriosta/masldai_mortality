import pandas as pd
import requests
from io import BytesIO

def download_and_process_data():
    # URLs and columns to retain with descriptive names
    urls = {
        'DEMO': ('https://wwwn.cdc.gov/Nchs/Nhanes/2021-2022/DEMO_L.XPT', 
                 {'RIAGENDR': 'Gender', 'RIDAGEYR': 'Age_in_years', 'RIDRETH1': 'Race_Hispanic_origin', 'RIDRETH3': 'Race_Hispanic_origin_NH_Asian'}),
        'LUX': ('https://wwwn.cdc.gov/Nchs/Nhanes/2021-2022/LUX_L.XPT', 
                {'LUAXSTAT': 'Elastography_exam', 'LUANMVGP': 'Complete_measures', 'LUXSMED': 'Median_stiffness_kPa', 'LUXSIQR': 'Stiffness_IQR', 
                 'LUXSIQRM': 'Stiffness_IQR_ratio', 'LUXCAPM': 'Median_CAP_dBm', 'LUXCPIQR': 'CAP_IQR'}),
        'BMX': ('https://wwwn.cdc.gov/Nchs/Nhanes/2021-2022/BMX_L.XPT', 
                {'BMXBMI': 'BMI', 'BMXWAIST': 'Waist_Circumference_cm'}),
        'BPX': ('https://wwwn.cdc.gov/Nchs/Nhanes/2021-2022/BPXO_L.XPT', 
                {'BPXOSY1': 'Systolic_1st', 'BPXODI1': 'Diastolic_1st', 'BPXOSY2': 'Systolic_2nd', 'BPXODI2': 'Diastolic_2nd', 
                 'BPXOSY3': 'Systolic_3rd', 'BPXODI3': 'Diastolic_3rd'}),
        'HDL': ('https://wwwn.cdc.gov/Nchs/Nhanes/2021-2022/HDL_L.XPT', 
                {'LBDHDD': 'Direct_HDL_Cholesterol'}),
        'CBC': ('https://wwwn.cdc.gov/Nchs/Nhanes/2021-2022/CBC_L.XPT', 
                {'LBXPLTSI': 'Platelet_count'}),
        'GHB': ('https://wwwn.cdc.gov/Nchs/Nhanes/2021-2022/GHB_L.XPT', 
                {'LBXGH': 'Glycohemoglobin'}),
        'HEPB': ('https://wwwn.cdc.gov/Nchs/Nhanes/2021-2022/HEPB_S_L.XPT', 
                 {'LBXHBS': 'Hepatitis_B_Surface_Antibody'}),
        'HEQ': ('https://wwwn.cdc.gov/Nchs/Nhanes/2021-2022/HEQ_L.XPT', 
                {'HEQ010': 'Ever_told_have_Hepatitis_B'}),
        'GLU': ('https://wwwn.cdc.gov/Nchs/Nhanes/2021-2022/GLU_L.XPT', 
                {'LBXGLU': 'Fasting_Glucose'}),
        'ALQ': ('https://wwwn.cdc.gov/Nchs/Nhanes/2021-2022/ALQ_L.XPT', 
                {'ALQ130': 'Avg_alcoholic_drinks_per_day'}),
        'BPQ': ('https://wwwn.cdc.gov/Nchs/Nhanes/2021-2022/BPQ_L.XPT', 
                {'BPQ150': 'Taking_BP_medication', 'BPQ101D': 'Taking_cholesterol_medication'}),
        'DIQ': ('https://wwwn.cdc.gov/Nchs/Nhanes/2021-2022/DIQ_L.XPT', 
                {'DIQ010': 'Doctor_told_have_diabetes', 'DIQ050': 'Taking_insulin_now', 'DIQ070': 'Take_diabetic_pills'})
    }

    # Initialize empty dictionary for DataFrames
    dfs = {}
    for key, (url, col_map) in urls.items():
        response = requests.get(url)
        df = pd.read_sas(BytesIO(response.content), format='xport')
        df = df[['SEQN'] + list(col_map.keys())]
        df.rename(columns=col_map, inplace=True)
        dfs[key] = df

    # Merge dataframes starting with LUX as main
    main_df = dfs['LUX']
    for key, df in dfs.items():
        if key != 'LUX':
            main_df = main_df.merge(df, on='SEQN', how='left')

    return main_df

# Download and process data
main_df = download_and_process_data()


main_df['Avg_Systolic_BP'] = main_df[['Systolic_1st', 'Systolic_2nd', 'Systolic_3rd']].mean(axis=1)
main_df['Avg_Diastolic_BP'] = main_df[['Diastolic_1st', 'Diastolic_2nd', 'Diastolic_3rd']].mean(axis=1)


print(main_df.columns.tolist())

main_df = main_df[(main_df['Age_in_years'] >= 18) & 
                  (main_df['Ever_told_have_Hepatitis_B'] == 2) | 
                  (main_df['Ever_told_have_Hepatitis_B'].isna()) & 
                  (main_df['Elastography_exam'] == 1)]


def is_masld(
    HSSEX, BMPBMI, BMPWAIST, G1P, GHP, HAD1, HAD6, HAD10,
    PEPMNK1R, PEPMNK5R, HAE5A, HAE9D, HDP
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

    # Criterion 4 and 5, dyslipidemia: HDL cholesterol, or lipid-lowering treatment
    is_dyslipidemia = int(
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

result = main_df.apply(
    lambda row: is_masld(
        HSSEX=row['Gender'], 
        BMPBMI=row['BMI'], 
        BMPWAIST=row['Waist_Circumference_cm'], 
        G1P=row['Fasting_Glucose'], 
        GHP=row['Glycohemoglobin'], 
        HAD1=row['Doctor_told_have_diabetes'], 
        HAD6=row['Taking_insulin_now'], 
        HAD10=row['Take_diabetic_pills'], 
        PEPMNK1R=row['Avg_Systolic_BP'], 
        PEPMNK5R=row['Avg_Diastolic_BP'], 
        HAE5A=row['Taking_BP_medication'], 
        HAE9D=row['Taking_cholesterol_medication'], 
        HDP=row['Direct_HDL_Cholesterol']
    ), axis=1)

# Unpack the result into separate columns in the DataFrame
main_df = main_df.join(pd.DataFrame(result.tolist()))

main_df = main_df[main_df['is_masld'] == 1]
