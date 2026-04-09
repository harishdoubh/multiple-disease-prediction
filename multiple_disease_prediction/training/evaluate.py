from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, precision_score, recall_score, f1_score
import pandas as pd
import numpy as np

def evaluate_model(y_train, y_train_pred, y_test, y_test_pred, model_name, class_names=None):
    if class_names is None:
        class_names = ['Negative', 'Positive']
        
    train_accuracy = accuracy_score(y_train, y_train_pred)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    
    precision = precision_score(y_test, y_test_pred, zero_division=0)
    recall = recall_score(y_test, y_test_pred, zero_division=0)
    f1 = f1_score(y_test, y_test_pred, zero_division=0)
    
    cm = confusion_matrix(y_test, y_test_pred)
    
    return {
        'Model Name': model_name,
        'Train_Accuracy': float(train_accuracy),
        'Accuracy': float(test_accuracy),
        'Precision': float(precision),
        'Recall': float(recall),
        'F1-score': float(f1),
        'Confusion_Matrix': cm.tolist()
    }
