import requests
from io import BytesIO 
import numpy as np
import pandas as pd

def calculate_fib4(age, ast, alt, platelets):
    """
    Calculates the FIB4 score based on age, AST, ALT, and platelet count.

    Parameters:
        age: Age of the patient
        ast: Aspartate Aminotransferase (AST) level
        alt: Alanine Aminotransferase (ALT) level
        platelets: Platelet count

    Returns:
        float: FIB4 score or NaN if inputs are invalid.
    """
    if pd.isna(age) or pd.isna(ast) or pd.isna(alt) or pd.isna(platelets) or alt == 0 or platelets == 0:  # Check for NA values and avoid division by zero
        return np.nan
    return (age * ast) / (platelets * np.sqrt(alt))

def calculate_nfs(age, bmi, diabetes, ast, alt, platelets, albumin):
    """
    Calculates the NFS score based on age, BMI, diabetes history, AST, ALT, platelet count, and albumin.

    Parameters:
        age: Age of the patient
        bmi: Body Mass Index (BMI)
        diabetes: Diabetes history
        ast: Aspartate Aminotransferase (AST) level
        alt: Alanine Aminotransferase (ALT) level
        platelets: Platelet count
        albumin: Albumin level

    Returns:
        float: NFS score or NaN if inputs are invalid.
    """
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

def calculate_gfr(serum_cr, age, is_female):
    """
    Calculate GFR using the MDRD equation.

    Args:
        serum_cr: Serum creatinine in mg/dL
        age: Age in years  
        is_female: Boolean indicating if patient is female
    
    Returns:
        Calculated GFR value
    """
    gfr = 175 * (serum_cr ** -1.154) * (age ** -0.203)
    if is_female:
        gfr *= 0.742
    return gfr

def calculate_gfr_ckdepi(serum_cr, age, is_female):
    """
    Calculate GFR using the CKD-EPI equation.
    
    Args:
        serum_cr: Serum creatinine in mg/dL
        age: Age in years  
        is_female: Boolean indicating if patient is female
    
    Returns:
        Calculated GFR value
    """
    if is_female:
        if serum_cr <= 0.7:
            A = 0.7
            B = -0.241
        else:
            A = 0.7 
            B = -1.2
    else:
        if serum_cr <= 0.9:
            A = 0.9
            B = -0.302
        else:
            A = 0.9
            B = -1.2
            
    gfr = 142 * ((serum_cr/A)**B) * (0.9938**age)
    
    if is_female:
        gfr *= 1.012
        
    return gfr

def is_masld(sex, body_mass_index, waist_circumference, plasma_glucose, glycated_hemoglobin, diabetes_history, taking_insulin, taking_diabetes_pills, systolic_blood_pressure, diastolic_blood_pressure, taking_bp_medication, taking_cholesterol_medication, hdl_cholesterol, avg_alcoholic_drinks=None, hep_b_core_ab=None, hep_c_ab=None):
    """
    Determines if a patient meets MASLD criteria.

    Parameters:
        sex: Sex (1 = Male, 2 = Female)
        body_mass_index: Body-mass index
        waist_circumference: Waist circumference (cm)
        plasma_glucose: Plasma glucose (mg/dL)
        glycated_hemoglobin: Glycated hemoglobin (HbA1c)
        diabetes_history: History of diabetes (1 = Yes, 2 = No)
        taking_insulin: Currently taking insulin (1 = Yes, 2 = No)
        taking_diabetes_pills: Currently taking diabetes pills (1 = Yes, 2 = No)
        systolic_blood_pressure: Systolic blood pressure
        diastolic_blood_pressure: Diastolic blood pressure
        taking_bp_medication: Taking medication for high blood pressure (1 = Yes, 2 = No)
        taking_cholesterol_medication: Taking cholesterol-lowering medication (1 = Yes, 2 = No)
        hdl_cholesterol: Serum HDL cholesterol (mg/dL)
        avg_alcoholic_drinks: Average number of alcoholic drinks per day
        hep_b_core_ab: Hepatitis B core antibody status
        hep_c_ab: Hepatitis C antibody status

    Returns:
        dict: Dictionary with MASLD criteria results and individual criteria flags.
    """
    # Criterion 1, body: BMI or waist circumference
    bmi_criteria = body_mass_index >= 25
    wc_criteria = waist_circumference > (94 if sex == 1 else 80)
    is_body = int(bmi_criteria or wc_criteria)

    # Criterion 2, diabetes: Blood glucose or diabetes history/treatment
    is_diabetes = int(
        plasma_glucose >= 100 or
        glycated_hemoglobin >= 5.7 or
        diabetes_history == 1 or
        taking_insulin == 1 or
        taking_diabetes_pills == 1
    )

    # Criterion 3, hypertension: Blood pressure or antihypertensive treatment
    is_hypertension = int(
        systolic_blood_pressure >= 130 or
        diastolic_blood_pressure >= 85 or
        taking_bp_medication == 1
    )

    # Criterion 4 and 5, dyslipidemia: HDL cholesterol, or lipid-lowering treatment
    is_dyslipidemia = int(
        taking_cholesterol_medication == 1 or
        (hdl_cholesterol <= 40 if sex == 1 else hdl_cholesterol <= 50) or
        taking_cholesterol_medication == 1
    )

    # MASLD criteria met if at least one of the criteria is true
    is_masld = int(any([is_body, is_diabetes, is_hypertension, is_dyslipidemia]))

    if sex == 2:
        is_masld = int(is_masld and (avg_alcoholic_drinks <= 1 or pd.isna(avg_alcoholic_drinks)))
    elif sex == 1:
        is_masld = int(is_masld and (avg_alcoholic_drinks <= 2 or pd.isna(avg_alcoholic_drinks)))

    is_masld = int(is_masld and (pd.isna(hep_b_core_ab) or hep_b_core_ab == 2 or pd.isna(hep_c_ab) or hep_c_ab == 2))

    return {
        'is_masld': is_masld,
        'is_body': is_body,
        'is_diabetes': is_diabetes,
        'is_hypertension': is_hypertension,
        'is_dyslipidemia': is_dyslipidemia
    }

def determine_fibrosis_stage(row):
    """
    Determines the fibrosis stage based on FIB4, NFS, and median stiffness.

    Parameters:
        row: A row from the DataFrame containing FIB4, NFS, and median stiffness values.

    Returns:
        str: Fibrosis stage ('F1', 'F3', 'F4', 'CSPH') or NaN if criteria are not met.
    """
    if row['FIB4'] < 1.3 or row['NFS'] < -1.455:
        if row['Median_stiffness_kPa'] < 5:
            return 'F1'
    elif 1.3 <= row['FIB4'] <= 2.67 or -1.455 <= row['NFS'] <= 0.676:
        if row['Median_stiffness_kPa'] >= 15:
            return 'F4'
        elif row['Median_stiffness_kPa'] >= 8:
            return 'F3'
    elif row['FIB4'] > 2.67 or row['NFS'] > 0.676:
        if row['Median_stiffness_kPa'] >= 25:
            return 'CSPH'
        elif row['Median_stiffness_kPa'] >= 15:
            return 'F4'
        elif row['Median_stiffness_kPa'] >= 12:
            return 'F3'
    return np.nan

def download_and_process_data():
    """
    Downloads and processes NHANES data from specified URLs and retains specific columns.

    Column Names:
        - Gender
        - Age_in_years_at_screening
        - Race_Hispanic_origin
        - Race_Hispanic_origin_w_NH_Asian
        - Elastography_exam_status
        - Count_complete_measures_from_final_wand
        - Count_measures_attempted_with_final_wand
        - Median_stiffness_kPa
        - Stiffness_E_interquartile_range
        - Ratio_Stiffness_IQRe_median_E
        - Median_CAP_dB_per_meter
        - CAP_interquartile_range
        - Body Mass Index (kg/m**2)
        - Waist Circumference (cm)
        - Systolic - 1st oscillometric reading
        - Diastolic - 1st oscillometric reading
        - Systolic - 2nd oscillometric reading
        - Diastolic - 2nd oscillometric reading
        - Systolic - 3rd oscillometric reading
        - Diastolic - 3rd oscillometric reading
        - Direct HDL-Cholesterol (mg/dL)
        - Triglyceride (mg/dL)
        - Platelet count (1000 cells/uL)
        - Glycohemoglobin (%)
        - Hepatitis_B_core_antibody
        - Hepatitis_B_surface_antigen
        - Hepatitis C RNA
        - Hepatitis C Antibody (confirmed)
        - Fasting Glucose (mg/dL)
        - Alanine Aminotransferase (ALT) (U/L)
        - Aspartate Aminotransferase (AST) (U/L)
        - Albumin, refrigerated serum (g/dL)
        - Gamma Glutamyl Transferase (GGT) (IU/L)
        - Avg # alcoholic drinks/day - past 12 mos
        - Taking prescription for hypertension
        - Told to take prescription for cholesterol
        - Doctor_told_you_have_diabetes
        - Taking_insulin_now
        - Take_diabetic_pills_to_lower_blood_sugar
        - Overnight_hospital_patient_in_last_year
    """
    # URLs and columns to retain with descriptive names
    urls = {
        'DEMO': ('https://wwwn.cdc.gov/Nchs/Nhanes/2017-2018/P_DEMO.XPT', 
                 {'RIAGENDR': 'Gender', 'RIDAGEYR': 'Age (years)', 'RIDRETH1': 'Race_Hispanic_origin', 'RIDRETH3': 'Race_Hispanic_origin_w_NH_Asian'}),
        
        'LUX': ('https://wwwn.cdc.gov/Nchs/Nhanes/2017-2018/P_LUX.XPT', 
                {'LUAXSTAT': 'Elastography_exam_status', 
                 'LUANMVGP': 'Count_complete_measures_from_final_wand', 
                 'LUANMTGP': 'Count_measures_attempted_with_final_wand', 
                 'LUXSMED': 'Median_stiffness_kPa', 
                 'LUXSIQR': 'Stiffness_E_interquartile_range', 
                 'LUXSIQRM': 'Ratio_Stiffness_IQRe_median_E', 
                 'LUXCAPM': 'Median_CAP_dB_per_meter', 
                 'LUXCPIQR': 'CAP_interquartile_range'}),
        'BMX': ('https://wwwn.cdc.gov/Nchs/Nhanes/2017-2018/P_BMX.XPT', 
                {'BMXBMI': 'Body-mass index (kg/m**2)', 'BMXWAIST': 'Waist Circumference (cm)'}),
        'BPX': ('https://wwwn.cdc.gov/Nchs/Nhanes/2017-2018/P_BPXO.XPT', 
                {'BPXOSY1': 'Systolic - 1st oscillometric reading', 'BPXODI1': 'Diastolic - 1st oscillometric reading', 
                 'BPXOSY2': 'Systolic - 2nd oscillometric reading', 'BPXODI2': 'Diastolic - 2nd oscillometric reading', 
                 'BPXOSY3': 'Systolic - 3rd oscillometric reading', 'BPXODI3': 'Diastolic - 3rd oscillometric reading'}),
       
        'HDL': ('https://wwwn.cdc.gov/Nchs/Nhanes/2017-2018/P_HDL.XPT', 
                {'LBDHDD': 'Direct HDL-Cholesterol (mg/dL)'}),
        'TRIG': ('https://wwwn.cdc.gov/Nchs/Nhanes/2017-2018/P_TRIGLY.XPT', 
                {'LBXTR': 'Triglyceride (mg/dL)'}),
        'CBC': ('https://wwwn.cdc.gov/Nchs/Nhanes/2017-2018/P_CBC.XPT', 
                {'LBXPLTSI': 'Platelet count (1000 cells/µL)'}),
        'GHB': ('https://wwwn.cdc.gov/Nchs/Nhanes/2017-2018/P_GHB.XPT', 
                {'LBXGH': 'Glycohemoglobin (%)'}),
        'HEPBD': ('https://wwwn.cdc.gov/Nchs/Nhanes/2017-2018/P_HEPBD.XPT', 
                   {'LBXHBC': 'Hepatitis_B_core_antibody', 
                    'LBDHBG': 'Hepatitis_B_surface_antigen'}),
        'HEPC': ('https://wwwn.cdc.gov/Nchs/Nhanes/2017-2018/P_HEPC.XPT', 
                  {'LBXHCR': 'Hepatitis C RNA', 
                   'LBDHCI': 'Hepatitis C Antibody (confirmed)'}),
        'GLU': ('https://wwwn.cdc.gov/Nchs/Nhanes/2017-2018/P_GLU.XPT', 
                {'LBXGLU': 'Fasting Glucose (mg/dL)'}),
        'BIOPRO': ('https://wwwn.cdc.gov/Nchs/Nhanes/2017-2018/P_BIOPRO.XPT', 
                   {'LBXSATSI': 'Alanine aminotransferase (U/L)', 
                    'LBXSASSI': 'Aspartate aminotransferase (U/L)', 
                    'LBXSAL': 'Albumin, refrigerated serum (g/dL)',
                    'LBXSCR': 'Creatinine, refrigerated serum (mg/dL)',
                    'LBXSGTSI': 'Gamma Glutamyl Transferase (GGT) (IU/L)'}),

        'ALQ': ('https://wwwn.cdc.gov/Nchs/Nhanes/2017-2018/P_ALQ.XPT', 
                {'ALQ130': 'Avg # alcoholic drinks/day - past 12 mos'}),
        'BPQ': ('https://wwwn.cdc.gov/Nchs/Nhanes/2017-2018/P_BPQ.XPT', 
                {'BPQ040A': 'Taking prescription for hypertension', 'BPQ090D': 'Told to take prescription for cholesterol'}),
        'DIQ': ('https://wwwn.cdc.gov/Nchs/Nhanes/2017-2018/P_DIQ.XPT', 
                {'DIQ010': 'Doctor_told_you_have_diabetes', 'DIQ050': 'Taking_insulin_now', 'DIQ070': 'Take_diabetic_pills_to_lower_blood_sugar'}),
        'HUQ': ('https://wwwn.cdc.gov/Nchs/Nhanes/2017-2018/P_HUQ.XPT',
                {'HUQ071': 'Overnight_hospital_patient_in_last_year'}),
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
    original_row_count = main_df.shape[0]  # Track original number of rows with LUX
    print(f"Original rows with LUX: {original_row_count}")

    for key, df in dfs.items():
        if key != 'LUX':
            main_df = main_df.merge(df, on='SEQN', how='left')

    return main_df

