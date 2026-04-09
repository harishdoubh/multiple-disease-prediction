import os
import urllib.request
from sklearn.datasets import make_classification
import pandas as pd
import numpy as np

def download_diabetes():
    url = "https://raw.githubusercontent.com/plotly/datasets/master/diabetes.csv"
    output_path = os.path.join("data", "diabetes.csv")
    if not os.path.exists(output_path):
        try:
            urllib.request.urlretrieve(url, output_path)
            print(f"Downloaded diabetes dataset to {output_path}")
        except Exception as e:
            print(f"Failed to download diabetes dataset: {e}")

def create_synthetic_heart():
    output_path = os.path.join("data", "heart.csv")
    if not os.path.exists(output_path):
        print("Creating a synthetic Heart Disease dataset...")
        # 13 features: age,sex,cp,trestbps,chol,fbs,restecg,thalach,exang,oldpeak,slope,ca,thal
        feature_names = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 
                         'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']
        X, y = make_classification(n_samples=500, n_features=13, n_informative=8, 
                                   n_redundant=2, random_state=42, class_sep=1.5)
        df = pd.DataFrame(X, columns=feature_names)
        # make some values discrete roughly matching real data
        df['age'] = np.clip(np.round(df['age'] * 15 + 50), 20, 90).astype(int)
        df['sex'] = np.clip(np.round((df['sex'] + 1) / 2), 0, 1).astype(int)
        df['cp'] = np.clip(np.round((df['cp'] + 3) / 1.5), 0, 3).astype(int)
        df['target'] = y
        df.to_csv(output_path, index=False)
        print(f"Saved synthetic heart dataset to {output_path}")

def create_synthetic_parkinsons():
    output_path = os.path.join("data", "parkinsons.csv")
    if not os.path.exists(output_path):
        print("Creating a synthetic Parkinson's Disease dataset...")
        feature_names = ['MDVP:Fo(Hz)','MDVP:Fhi(Hz)','MDVP:Flo(Hz)','MDVP:Jitter(%)','MDVP:Jitter(Abs)','MDVP:RAP','MDVP:PPQ','Jitter:DDP','MDVP:Shimmer','MDVP:Shimmer(dB)','Shimmer:APQ3','Shimmer:APQ5','MDVP:APQ','Shimmer:DDA','NHR','HNR','RPDE','DFA','spread1','spread2','D2','PPE']
        # 22 features
        X, y = make_classification(n_samples=300, n_features=22, n_informative=12, 
                                   n_redundant=4, random_state=42, class_sep=1.5)
        df = pd.DataFrame(X, columns=feature_names)
        df['status'] = y
        df.to_csv(output_path, index=False)
        print(f"Saved synthetic parkinsons dataset to {output_path}")

if __name__ == '__main__':
    os.makedirs('data', exist_ok=True)
    download_diabetes()
    create_synthetic_heart()
    create_synthetic_parkinsons()
