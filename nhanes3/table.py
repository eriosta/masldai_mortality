import numpy as np
import pandas as pd
from tableone import TableOne


df = pd.read_csv("nhanes3/nhanes3_masld_mortality.csv").dropna(subset=['mortstat', 'FIB4'])

df['FIB4']

df_subset = df[['HSAGEIR', 'HSSEX','DMARETHN','DMARACER',
                'ASPSI','ATPSI', 'GGPSI', 'PLP','CEP','BMPWAIST',  'GHP', 'BMPBMI', 
                'FIB4','Framingham_Risk_Score',
                'mortstat', 'ucod_leading']]

df_subset = df_subset[df_subset['ucod_leading'] != 4].rename(columns={
    'HSAGEIR': 'Age_in_years_at_screening', 
    'GHP': 'Glycohemoglobin (%)',
    'ATPSI': 'Alanine Aminotransferase (ALT) (U/L)', 
    'ASPSI': 'Aspartate Aminotransferase (AST) (U/L)',
    'PLP': 'Platelet count (1000 cells/uL)',
    'BMPBMI': 'Body Mass Index (kg/m**2)',
    'CEP':'Serum creatinine (mg/dL)',
    'HSSEX': 'Gender',
    'DMARETHN':'Ethnicity',
    'BMPWAIST': 'Waist Circumference (cm)',
    'DMARACER':'Race'}).dropna(subset=['Age_in_years_at_screening', 
                                        'Glycohemoglobin (%)', 
                                        'Alanine Aminotransferase (ALT) (U/L)',
                                        'Aspartate Aminotransferase (AST) (U/L)',
                                            'Platelet count (1000 cells/uL)', 
                                            'Body Mass Index (kg/m**2)'])


df_filtered = df_subset


# Define the categorical variables
df_filtered['Gender'] = pd.Categorical(df_filtered['Gender'].astype(str))
df_filtered['Ethnicity'] = pd.Categorical(df_filtered['Ethnicity'].astype(str))
df_filtered['Race'] = pd.Categorical(df_filtered['Race'].astype(str))
df_filtered['ucod_leading'] = pd.Categorical(df_filtered['ucod_leading'].astype(str))
df_filtered['mortstat'] = df_filtered['mortstat'].astype(str)

df_filtered.info()

# Create a TableOne object comparing mortstat == 1 vs. mortstat == 0
table = TableOne(df_filtered,
                    groupby='mortstat', 
                    pval=True)

# # Export the table to an Excel file
table.to_excel("nhanes3/mortstat_comparison.xlsx")
