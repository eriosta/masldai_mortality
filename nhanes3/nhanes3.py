import requests
import pandas as pd
import os
import wget

class NHANES3Data:
    def __init__(self, file_ext, data_files):
        self.file_ext = file_ext
        self.data_files = data_files
        self.urls = [f"https://wwwn.cdc.gov/nchs/data/nhanes3/{ext}/{file}" for ext in self.file_ext for file in self.data_files]
        self.out = pd.DataFrame(data=[0], columns=['SEQN'])

    def download_files(self):
        for url in self.urls:
            try:
                response = requests.get(url)
                if response.status_code == 200:
                    filename = url.split('/')[-1]
                    with open(filename, 'wb') as f:
                        f.write(response.content)
                    print(f"Downloaded: {filename}")
                else:
                    print(f"Failed to download: {url}")
            except Exception as e:
                print(f"Error downloading {url}: {str(e)}")

    def merge_files(self):
        for f in self.data_files:
            tmp = pd.read_sas(f)
            self.out = self.out.merge(tmp, on='SEQN', how='outer') 

    def clean_directory(self):
        cwd = os.getcwd()
        files = os.listdir(cwd)
        for file in files:
            if file.endswith(".xpt"):
                os.remove(os.path.join(cwd, file))

    def get_nhanes3(self):
        self.download_files()
        print("Downloading SAS files...")
        self.merge_files()
        self.out.to_csv('data.csv')
        print("Creating dataframe...")
        self.clean_directory()
        print("Cleaning directory...")
        return self.out

# Call function to get data
exts = ['34a','34B']
files = ['HGUHS.xpt','HGUHSSE.xpt']
dat = NHANES3Data(file_ext=exts,data_files=files).get_nhanes3()

# Select columns of interest
dat = dat[['SEQN','GUPHSPF']]

# Load other datasets
labs = pd.read_csv('nhanes3/files/files/labs.csv')
exam = pd.read_csv('nhanes3/files/files/exam.csv')
adult = pd.read_csv('nhanes3/files/files/adult.csv')

adult = adult.query('HSAGEIR >= 18')
labs = labs.query('((HBP == 2 or HBP == 8 or HBP.isna()) or (HCP == 2 or HCP == 8 or HCP.isna()))')
exam = exam.query('(HSAGEIR >= 18)')

adult = adult[(adult['HAZA2A2'] == 2) | (adult['HAZA2A2'] == 8) | (adult['HAZA2A2'].isna())]
exam = exam[(exam['PEP6C'] == 2) | (exam['PEP6C'] == 8) | (exam['PEP6C'].isna())]

# Inner joing datasets together by SEQN 
dat = dat.merge(adult, on='SEQN', how='inner')
dat = dat.merge(labs, on='SEQN', how='left')
dat = dat.merge(exam, on='SEQN', how='left')

dat = dat[dat['GUPHSPF'].isin([2, 3, 4])]

dat.to_csv('nhanes3/files/files/nhanes3.csv',index=False)