import pandas as pd
import numpy as np
from utils import download_and_process_data, is_masld, calculate_fib4, calculate_nfs, determine_fibrosis_stage, calculate_gfr, calculate_gfr_ckdepi

class DataPreparation:
    def __init__(self, data_path=None, column_names=None, cohort_type='derivation'):
        self.data_path = data_path
        self.cohort_type = cohort_type
        self.column_names = column_names or self.get_default_column_names()

    def get_default_column_names(self):
        if self.cohort_type == 'derivation':
            return {
                'gender': 'Gender',
                'bmi': 'Body-mass index (kg/m**2)',
                'waist_circumference': 'Waist Circumference (cm)',
                'plasma_glucose': 'Fasting Glucose (mg/dL)',
                'glycated_hemoglobin': 'Glycohemoglobin (%)',
                'diabetes_history': 'Doctor_told_you_have_diabetes',
                'taking_insulin': 'Taking_insulin_now',
                'taking_diabetes_pills': 'Take_diabetic_pills_to_lower_blood_sugar',
                'systolic_bp': 'Avg_Systolic_BP',
                'diastolic_bp': 'Avg_Diastolic_BP',
                'taking_bp_medication': 'Taking prescription for hypertension',
                'taking_cholesterol_medication': 'Told to take prescription for cholesterol',
                'hdl_cholesterol': 'Direct HDL-Cholesterol (mg/dL)',
                'avg_alcoholic_drinks': 'Avg # alcoholic drinks/day - past 12 mos',
                'hep_b_core_ab': 'Hepatitis_B_core_antibody',
                'hep_c_ab': 'Hepatitis C Antibody (confirmed)',
                'age': 'Age (years)',
                'ast': 'Aspartate aminotransferase (U/L)',
                'alt': 'Alanine aminotransferase (U/L)',
                'platelet_count': 'Platelet count (1000 cells/µL)',
                'albumin': 'Albumin, refrigerated serum (g/dL)'
            }
        else:  # validation cohort
            return {
                'gender': 'HSSEX',
                'bmi': 'BMPBMI',
                'waist_circumference': 'BMPWAIST',
                'plasma_glucose': 'G1P',
                'glycated_hemoglobin': 'GHP',
                'diabetes_history': 'HAD1',
                'taking_insulin': 'HAD6',
                'taking_diabetes_pills': 'HAD10',
                'systolic_bp': 'PEPMNK1R',
                'diastolic_bp': 'PEPMNK5R',
                'taking_bp_medication': 'HAE5A',
                'taking_cholesterol_medication': 'HAE9D',
                'hdl_cholesterol': 'HDP',
                'age': 'HSAGEIR',
                'ast': 'ASPSI',
                'alt': 'ATPSI',
                'platelet_count': 'PLP',
                'albumin': 'AMP'
            }

    def load_data(self):
        if self.data_path:
            data = pd.read_csv(self.data_path)
        else:
            data = download_and_process_data()
        return data

    def preprocess_data(self, data):
        if self.cohort_type == 'derivation':
            data[self.column_names['systolic_bp']] = data[['Systolic - 1st oscillometric reading', 'Systolic - 2nd oscillometric reading', 'Systolic - 3rd oscillometric reading']].mean(axis=1)
            data[self.column_names['diastolic_bp']] = data[['Diastolic - 1st oscillometric reading', 'Diastolic - 2nd oscillometric reading', 'Diastolic - 3rd oscillometric reading']].mean(axis=1)

            data = data[(data[self.column_names['age']] >= 18) & 
                        ((data[self.column_names['hep_b_core_ab']] == 2) | (data[self.column_names['hep_b_core_ab']].isna())) & 
                        ((data[self.column_names['hep_c_ab']] == 2) | (data[self.column_names['hep_c_ab']].isna()))]

            data = data[
                (data['Count_complete_measures_from_final_wand'] >= 10) & 
                (data['Ratio_Stiffness_IQRe_median_E'] < 30) & 
                (data['Median_CAP_dB_per_meter'] >= 275) & 
                (data['Elastography_exam_status'] == 1)
            ]
        else:  # validation cohort
            data = data[(data[self.column_names['age']] >= 18)]

        result = data.apply(
            lambda row: is_masld(
                sex=row[self.column_names['gender']], 
                body_mass_index=row[self.column_names['bmi']], 
                waist_circumference=row[self.column_names['waist_circumference']], 
                plasma_glucose=row[self.column_names['plasma_glucose']], 
                glycated_hemoglobin=row[self.column_names['glycated_hemoglobin']], 
                diabetes_history=row[self.column_names['diabetes_history']], 
                taking_insulin=row[self.column_names['taking_insulin']], 
                taking_diabetes_pills=row[self.column_names['taking_diabetes_pills']], 
                systolic_blood_pressure=row[self.column_names['systolic_bp']], 
                diastolic_blood_pressure=row[self.column_names['diastolic_bp']], 
                taking_bp_medication=row[self.column_names['taking_bp_medication']], 
                taking_cholesterol_medication=row[self.column_names['taking_cholesterol_medication']], 
                hdl_cholesterol=row[self.column_names['hdl_cholesterol']],
                avg_alcoholic_drinks=row.get(self.column_names.get('avg_alcoholic_drinks', np.nan)),
                hep_b_core_ab=row.get(self.column_names.get('hep_b_core_ab', np.nan)),
                hep_c_ab=row.get(self.column_names.get('hep_c_ab', np.nan))
            ), axis=1)

        data = data.join(pd.DataFrame(result.tolist()))
        data = data[(data['is_masld'] == 1)]

        data['GFR'] = data.apply(lambda row: calculate_gfr(row[self.column_names['serum_creatinine']], row[self.column_names['age']], row[self.column_names['gender']] == 2), axis=1)

        data['FIB4'] = data.apply(lambda row: calculate_fib4(row[self.column_names['age']], row[self.column_names['ast']], row[self.column_names['alt']], row[self.column_names['platelet_count']]), axis=1)
        data['NFS'] = data.apply(lambda row: calculate_nfs(row[self.column_names['age']], row[self.column_names['bmi']], row[self.column_names['diabetes_history']], row[self.column_names['ast']], row[self.column_names['alt']], row[self.column_names['platelet_count']], row[self.column_names['albumin']]), axis=1)

        data = data.dropna(subset=['FIB4', 'NFS'])
        data['fibrosis_stage'] = data.apply(determine_fibrosis_stage, axis=1)

        processed_data = data.dropna()
        return processed_data