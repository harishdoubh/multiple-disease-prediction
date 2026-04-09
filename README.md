# AI-Based Multi Disease Prediction System

A robust, enterprise-grade machine learning application designed to predict the likelihood of multiple chronic diseases (Diabetes, Heart Disease, and Parkinson's) based on clinical and physiological data. This system utilizes advanced Machine Learning pipelines, an ensemble modeling approach, and a professional-grade Streamlit web dashboard to provide accurate health assessments.

---

## 🚀 Features

- **Multi-Disease Prediction:** Dynamically predicts Diabetes, Heart Disease, and Parkinson's through dedicated machine learning models.
- **Intelligent Model Selection:** Trains multiple algorithms (XGBoost, Support Vector Machines (SVM), Random Forest, and Logistic Regression) and automatically selects the highest-performing model for disease predictions using GridSearchCV/RandomizedSearchCV.
- **Advanced Pre-processing:** Implements standard preprocessing steps, including handling class imbalances with SMOTE, dealing with zero/null values, and scaling using `StandardScaler` for reliable metrics and avoiding data leaks.
- **Dynamic Dashboard UI (Streamlit):** Engaging, multi-page (Tabs) web interface with dynamic custom CSS for patient input, medical insights, model analytics cross-validation, and diagnostic outcome representation visually showing probability via Risk Levels (Low, Moderate, High).
- **Downloadable Clinical Reports:** Automatically generates comprehensive, timestamped clinical PDF/text reports summarising patient inputs, prediction probabilities, AI confidence scores, and specific medical precautions/next steps.

---

## 📁 Project Structure

The project employs a highly modular architecture separating data logic, training scripts, artifacts, and frontend views:

```text
multiple_disease_prediction/
│
├── data/
│   ├── diabetes.csv           <-- Pima Indians Diabetes Dataset
│   ├── heart.csv              <-- Heart Disease Dataset
│   └── parkinsons.csv         <-- Parkinson's Medical Dataset
│
├── training/
│   ├── preprocess.py          <-- Data loading, missing value imputation, scaling
│   ├── train_model.py         <-- Model tuning, SMOTE, training (LR, RF, SVM, XGB), selection
│   └── evaluate.py            <-- Accuracy score, confusion matrix, classification reports
│
├── models/
│   ├── [disease]_model.pkl             <-- Best trained model artifact per disease
│   ├── [disease]_scaler.pkl            <-- Standard Scaler per disease 
│   ├── [disease]_feature_names.pkl     <-- Track columns for prediction compatibility
│   └── [disease]_model_metrics.json    <-- Tracking history & scores for analytics tab
│
├── utils/                     <-- Future generic helpers and abstractions
│
├── app.py                     <-- Main Streamlit web application & UI
├── requirements.txt           <-- Project python dependencies
└── README.md                  <-- Project documentation
```

---

## 🛠️ Step-by-Step Execution Instructions

### 1. Setup Environment
It is highly recommended to use a Python virtual environment to manage dependencies locally.
```bash
python -m venv venv
# On macOS/Linux:
source venv/bin/activate  
# On Windows:
venv\Scripts\activate
```

### 2. Install Project Dependencies
Install the required machine learning and web dependencies:
```bash
pip install -r requirements.txt
```

### 3. Ensure Datasets are Present
Ensure the data files (`diabetes.csv`, `heart.csv`, and `parkinsons.csv`) reside in the `data/` directory. If they don't, you may need to run `python download.py` (if available) or download them from a reliable repository (e.g., Kaggle/UCI) locally.

### 4. Train the ML Models & Build Artifacts
Run the comprehensive training pipeline. This will process the datasets, apply SMOTE, perform hyperparameter optimization for all algorithms, and save the *best* performing configurations natively to the `models/` directory.
```bash
python training/train_model.py
```
*(This will take a moment as it evaluates multiple models across RandomizedSearchCV)*

### 5. Launch the Web Application
Once the `.pkl` and `.json` files are properly generated in the `models/` folder, begin interacting with the AI engine:
```bash
streamlit run app.py
```

---

## 🧠 Technical Workflow & Models Explored

### Machine Learning Models Evaluated
- **XGBoost:** An optimized distributed gradient boosting library highly efficient, flexible, and portable, operating exceptionally well on tabular datasets.
- **Support Vector Machine (SVM):** Efficiently performs non-linear classification utilizing various kernel tricks (RBF, linear) for strict boundary creation.
- **Random Forest:** An ensemble learning model aggregating decision trees to drastically minimize variance and over-fitting while ensuring consistency.
- **Logistic Regression:** A reliable baseline statistical method leveraged for probabilistic binary classification.

### Pipeline Best Practices
- **Class Imbalance (SMOTE):** Employed Synthetic Minority Over-sampling Technique exclusively on the training sets to protect against class imbalance bias while completely guarding against data leakage on validation sets.
- **Tuning:** Deployed `RandomizedSearchCV` on each model instance.
- **Metrics:** Prioritized maximizing `Accuracy` & `F1-Score` to best represent reliability dynamically loaded into the Model Analytics Dashboard. 

---

## 👨‍💻 Developer
**Developed by Kavya**  
*Powered by Machine Learning*
