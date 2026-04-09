import os
import sys
import pickle
import json
import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from preprocess import load_and_preprocess_data, get_scaler, scale_test_data
from evaluate import evaluate_model

def get_models_and_params():
    models = {
        "Logistic Regression": LogisticRegression(random_state=42, class_weight='balanced', max_iter=2000),
        "Random Forest": RandomForestClassifier(random_state=42, class_weight='balanced'),
        "SVM": SVC(probability=True, random_state=42, class_weight='balanced'),
        "XGBoost": xgb.XGBClassifier(random_state=42, eval_metric='logloss')
    }
    
    params = {
        "Logistic Regression": {
            'C': [0.01, 0.1, 1, 10, 100],
            'solver': ['liblinear', 'lbfgs']
        },
        "Random Forest": {
            'n_estimators': [50, 100, 200, 300],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10]
        },
        "SVM": {
            'C': [0.1, 1, 10, 50],
            'gamma': ['scale', 'auto', 0.1, 0.01],
            'kernel': ['rbf', 'linear']
        },
        "XGBoost": {
            'n_estimators': [100, 200, 300],
            'learning_rate': [0.01, 0.05, 0.1, 0.2],
            'max_depth': [3, 5, 7],
            'subsample': [0.8, 1.0],
            'colsample_bytree': [0.8, 1.0]
        }
    }
    return models, params

def train_disease_model(disease_name, data_path, models_dir):
    print(f"\n{'='*50}")
    print(f" TRAINING PIPELINE FOR: {disease_name.upper()} ")
    print(f"{'='*50}\n")
    
    # Load and preprocess
    X, y = load_and_preprocess_data(disease_name, data_path)
    if X is None or len(X) == 0:
        print(f"Skipping {disease_name} - data not found or empty.")
        return
        
    class_names = ['Negative', 'Positive']
    if disease_name == 'diabetes':
        class_names = ['Not Diabetic', 'Diabetic']
    elif disease_name == 'heart':
         class_names = ['No Heart Disease', 'Heart Disease']
    elif disease_name == 'parkinsons':
         class_names = ['Healthy', 'Parkinsons']

    # 1. Split data using train_test_split (80% train, 20% test, random_state=42)
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # 2. Fit Scaler ONLY on training data
    X_train_scaled, scaler = get_scaler(X_train_raw)
    X_test_scaled = scale_test_data(X_test_raw, scaler)
    
    # 3. Handle class imbalance: Apply SMOTE ONLY on training data
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train)
    
    models, params = get_models_and_params()

    all_metrics = []
    best_model = None
    best_acc = -1
    best_model_name = ""

    os.makedirs(models_dir, exist_ok=True)
    
    for model_name in models.keys():
        print(f"\n--- Training and Tuning {model_name} ---")
        base_model = models[model_name]
        param_grid = params[model_name]
        
        # Apply Hyperparameter tuning
        search = RandomizedSearchCV(
            base_model, param_distributions=param_grid, n_iter=10, 
            scoring='accuracy', cv=5, n_jobs=-1, random_state=42
        )
        search.fit(X_train_resampled, y_train_resampled)
        
        model = search.best_estimator_
        
        # Predictions
        y_train_pred = model.predict(X_train_resampled)
        y_test_pred = model.predict(X_test_scaled)
        
        # Evaluate model using Accuracy Score and Confusion Matrix
        metrics = evaluate_model(y_train_resampled, y_train_pred, y_test, y_test_pred, model_name, class_names)
        
        print(f"Training Accuracy: {metrics['Train_Accuracy'] * 100:.2f}%")
        print(f"Testing Accuracy (Reported): {metrics['Accuracy'] * 100:.2f}%")
        print(f"F1 Score: {metrics['F1-score'] * 100:.2f}%")
        print(f"Confusion Matrix:\n{metrics['Confusion_Matrix']}")
        
        all_metrics.append(metrics)
        
        # Save individual model
        slug = model_name.lower().replace(" ", "_").replace("-", "")
        ind_model_filename = f"{disease_name}_{slug}_model.pkl"
        with open(os.path.join(models_dir, ind_model_filename), 'wb') as f:
            pickle.dump(model, f)
            
        # Track BEST model based on highest testing accuracy
        if metrics['Accuracy'] > best_acc:
            best_acc = metrics['Accuracy']
            best_model = model
            best_model_name = model_name

    # Print/log final selected accuracy clearly
    print(f"\n[INFO] Best Selected Model for {disease_name}: {best_model_name} with Accuracy ({best_acc*100:.2f}%)")
    
    # Save the BEST trained model using pickle
    model_filename = f"{disease_name}_model.pkl"
    with open(os.path.join(models_dir, model_filename), 'wb') as f:
        pickle.dump(best_model, f)
        
    scaler_filename = f"{disease_name}_scaler.pkl"
    with open(os.path.join(models_dir, scaler_filename), 'wb') as f:
        pickle.dump(scaler, f)
        
    features_filename = f"{disease_name}_feature_names.pkl"
    with open(os.path.join(models_dir, features_filename), 'wb') as f:
        pickle.dump(list(X.columns), f)
        
    metrics_json_path = os.path.join(models_dir, f"{disease_name}_model_metrics.json")
    # Sort backwards so the best is at the top in the UI
    all_metrics = sorted(all_metrics, key=lambda x: x['Accuracy'], reverse=True)
    with open(metrics_json_path, 'w') as f:
        json.dump(all_metrics, f, indent=4)
        
    print(f"[OK] Saved pipeline artifacts for {disease_name}")

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, '..', 'data')
    models_dir = os.path.join(base_dir, '..', 'models')
    
    datasets = {
        'diabetes': os.path.join(data_dir, 'diabetes.csv'),
        'heart': os.path.join(data_dir, 'heart.csv'),
        'parkinsons': os.path.join(data_dir, 'parkinsons.csv')
    }
    
    for disease, path in datasets.items():
        if os.path.exists(path):
            train_disease_model(disease, path, models_dir)

if __name__ == '__main__':
    main()
