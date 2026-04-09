import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import streamlit as st

def plot_confusion_matrix(cm, model_name):
    """
    Plots a confusion matrix using matplotlib.
    """
    fig, ax = plt.subplots(figsize=(6, 4))
    cax = ax.matshow(cm, cmap=plt.cm.Blues)
    fig.colorbar(cax)
    
    # Set labels
    classes = ['Not Diabetic', 'Diabetic']
    ax.set_xticks(np.arange(len(classes)))
    ax.set_yticks(np.arange(len(classes)))
    ax.set_xticklabels(classes)
    ax.set_yticklabels(classes)
    
    # Add text annotations inside the boxes
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(j, i, str(cm[i][j]), ha='center', va='center', color='black')
            
    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')
    ax.set_title(f'Confusion Matrix: {model_name}', pad=20)
    
    return fig

def plot_feature_importance(feature_importances_df, model_name):
    """
    Plots feature importances for models that support it (Random Forest, Logistic Regression).
    Highlights top 3 features.
    """
    if feature_importances_df is None or feature_importances_df.empty:
         return None
         
    fig, ax = plt.subplots(figsize=(8, 5))
    
    # Sort for bar chart display
    df_sorted = feature_importances_df.sort_values('Importance', ascending=True)
    
    # Create colors (highlight top 3)
    colors = ['#1f77b4'] * (len(df_sorted) - 3) + ['#ff7f0e'] * 3
    
    bars = ax.barh(df_sorted['Feature'], df_sorted['Importance'], color=colors)
    
    ax.set_xlabel('Importance Score')
    ax.set_ylabel('Features')
    ax.set_title(f'Feature Importances: {model_name}')
    
    return fig
