import pandas as pd
import pickle
import xgboost as xgb
from lifelines import CoxPHFitter

class NHANES3Predictor:
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
        df_subset = df[['SEQN', 'HSAGEIR', 'GHP', 'ATPSI', 'ASPSI', 'PLP', 'BMPBMI', 'GFR', 'mortstat', 'ucod_leading', 'permth_exm', 'FIB4']]
        df_subset = df_subset[df_subset['ucod_leading'] != 4].rename(columns={
            'HSAGEIR': 'Age (years)',
            'GHP': 'Glycohemoglobin (%)',
            'ATPSI': 'Alanine aminotransferase (U/L)',
            'ASPSI': 'Aspartate aminotransferase (U/L)',
            'PLP': 'Platelet count (1000 cells/µL)',
            'BMPBMI': 'Body-mass index (kg/m**2)',
        }).set_index(['SEQN', 'mortstat', 'ucod_leading', 'permth_exm', 'FIB4']).dropna()
        return df_subset

    def prepare_output_folder(self):
        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)

    def predict(self):
        data_dmatrix = xgb.DMatrix(self.df[self.features])
        return self.model.predict(data_dmatrix)

    def perform_hr_analysis(self, follow_up_years):
        df = self.df.reset_index()
        df['time'] = df['permth_exm'].clip(upper=12*follow_up_years) / 12  # Convert months to years

        # Create binary indicators for FIB4 categories
        df['isfib4mod'] = (df['FIB4'] >= 1.3).astype(int)
        df['isfib4high'] = (df['FIB4'] >= 2.67).astype(int)

        # Create binary indicators for mortality types
        df['is_cardiac_mortality'] = df['ucod_leading'].isin([1, 5]).astype(int)
        df['is_malignancy_mortality'] = df['ucod_leading'].isin([2]).astype(int)

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
        covariate_sets = [
            ['Prediction_95_spec', 'Age (years)', 'HSSEX'],
            ['isfib4mod', 'Age (years)', 'HSSEX'],
            ['isfib4high', 'Age (years)', 'HSSEX']
        ]

        summary_list = []
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
            'Prediction_95_spec': 'FibroX-95',
            'isfib4mod': 'FIB-4 ≥1.3',
            'isfib4high': 'FIB-4 ≥2.67'
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

    def hr_analysis_10_year(self):
        return self.perform_hr_analysis(10)

    def hr_analysis_20_year(self):
        return self.perform_hr_analysis(20)

    def hr_analysis_30_year(self):
        return self.perform_hr_analysis(30)

# Example usage:
# predictor = NHANES3Predictor(model_path='path/to/model.pkl', data_path='path/to/data.csv')
# hr_10_year = predictor.hr_analysis_10_year()
# hr_20_year = predictor.hr_analysis_20_year()
# hr_30_year = predictor.hr_analysis_30_year()
# print(hr_10_year)
# print(hr_20_year)
# print(hr_30_year)
