import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def handle_diabetes(df):
    columns_with_zeros = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    for col in columns_with_zeros:
        if col in df.columns:
            df[col] = df[col].replace(0, np.nan)
            
    # Resolve null values with median imputation to preserve data volume 
    # (helps push accuracy higher)
    df.fillna(df.median(), inplace=True)
    
    target_col = 'Outcome'
    X = df.drop(columns=[target_col])
    y = df[target_col]
    return X, y

def handle_heart(df):
    target_col = 'target'
    if target_col not in df.columns:
        if 'num' in df.columns:
            target_col = 'num'
        elif 'Outcome' in df.columns:
            target_col = 'Outcome'
        elif 'class' in df.columns:
            target_col = 'class'
            
    # Heart dataset typically has some missing values represented as '?'
    df.replace('?', np.nan, inplace=True)
    df = df.apply(pd.to_numeric, errors='coerce')
    
    # Resolve null values
    df.fillna(df.median(), inplace=True)
    
    X = df.drop(columns=[target_col])
    y = df[target_col]
    return X, y

def handle_parkinsons(df):
    # Drop irrelevant columns
    if 'name' in df.columns:
        df = df.drop(columns=['name'])
        
    target_col = 'status'
    if target_col not in df.columns:
        if 'class' in df.columns:
            target_col = 'class'
            
    # Resolve null values
    df.replace('?', np.nan, inplace=True)
    df.fillna(df.median(), inplace=True)
    
    X = df.drop(columns=[target_col])
    y = df[target_col]
    return X, y

def load_and_preprocess_data(disease_name, filepath):
    """
    Loads dataset and performs preprocessing.
    """
    try:
        df = pd.read_csv(filepath)
    except FileNotFoundError:
        print(f"Error: Dataset not found at {filepath}.")
        return None, None
        
    if disease_name == 'diabetes':
        X, y = handle_diabetes(df)
    elif disease_name == 'heart':
        X, y = handle_heart(df)
    elif disease_name == 'parkinsons':
        X, y = handle_parkinsons(df)
    else:
        print(f"Unknown disease: {disease_name}")
        return None, None

    return X, y

def get_scaler(X_train):
    # Normalize/scale features using StandardScaler
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=X_train.columns)
    return X_train_scaled_df, scaler

def scale_test_data(X_test, scaler):
    X_test_scaled = scaler.transform(X_test)
    X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=X_test.columns)
    return X_test_scaled_df

if __name__ == '__main__':
    pass
