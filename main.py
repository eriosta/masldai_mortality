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
                {'BMXBMI': 'BMI'}),
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
        'GLU': ('https://wwwn.cdc.gov/Nchs/Nhanes/2021-2022/GLU_L.XPT', 
                {'LBXGLU': 'Fasting_Glucose'}),
        'ALQ': ('https://wwwn.cdc.gov/Nchs/Nhanes/2021-2022/ALQ_L.XPT', 
                {'ALQ130': 'Avg_alcoholic_drinks_per_day'}),
        'BPQ': ('https://wwwn.cdc.gov/Nchs/Nhanes/2021-2022/BPQ_L.XPT', 
                {'BPQ150': 'Taking_BP_medication', 'BPQ101D': 'Taking_cholesterol_medication'})
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
