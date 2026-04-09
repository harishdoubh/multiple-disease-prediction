import os
import pickle
import streamlit as st
import pandas as pd
import numpy as np
import time
import json
from datetime import datetime

# ==========================================
# 1. PAGE CONFIGURATION & CUSTOM CSS
# ==========================================
def setup_page():
    st.set_page_config(
        page_title="AI-Based Multi Disease Prediction System",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded"
    )

def inject_custom_css():
    st.markdown("""
        <style>
        .stApp { background: linear-gradient(135deg, #0f172a 0%, #1e1e2d 100%); color: #f1f5f9; }
        .dashboard-container { background: rgba(30, 41, 59, 0.7); border: 1px solid rgba(255, 255, 255, 0.1); border-radius: 12px; padding: 20px; text-align: center; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3); margin-bottom: 2rem; }
        .metric-title { color: #94a3b8; font-size: 1rem; font-weight: 500; margin-bottom: 0.5rem; }
        .metric-value { font-size: 2rem; font-weight: 700; background: -webkit-linear-gradient(45deg, #38bdf8, #818cf8); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
        .result-box-success { background: rgba(16, 185, 129, 0.15); border: 2px solid #10b981; padding: 30px; border-radius: 16px; margin: 20px 0; text-align: center; animation: fadeIn 0.6s ease; }
        .result-box-warning { background: rgba(245, 158, 11, 0.15); border: 2px solid #f59e0b; padding: 30px; border-radius: 16px; margin: 20px 0; text-align: center; animation: fadeIn 0.6s ease; }
        .result-box-error { background: rgba(239, 68, 68, 0.15); border: 2px solid #ef4444; padding: 30px; border-radius: 16px; margin: 20px 0; text-align: center; animation: fadeIn 0.6s ease; }
        div.stButton > button:first-child { width: 100%; border-radius: 12px; height: 60px; font-size: 22px; font-weight: 700; background: linear-gradient(90deg, #2563eb 0%, #3b82f6 100%); color: white; border: none; box-shadow: 0 4px 14px 0 rgba(37, 99, 235, 0.39); transition: all 0.3s ease; }
        div.stButton > button:first-child:hover { transform: translateY(-2px); box-shadow: 0 6px 20px rgba(37, 99, 235, 0.6); }
        h1 { font-family: 'Inter', sans-serif; font-weight: 800; color: #f8fafc; text-align: center; margin-bottom: 0.2rem; }
        .sub-header { color: #94a3b8; text-align: center; font-size: 1.25rem; font-weight: 400; margin-bottom: 2.5rem; }
        @keyframes fadeIn { from { opacity: 0; transform: translateY(10px); } to { opacity: 1; transform: translateY(0); } }
        .advice-card { background: rgba(255,255,255,0.05); padding: 15px; border-radius: 8px; margin-top: 10px; border-left: 4px solid #38bdf8; }
        .footer { text-align: center; padding: 40px 0 20px 0; color: #64748b; font-size: 0.95rem; border-top: 1px solid rgba(255,255,255,0.05); margin-top: 60px; }
        </style>
    """, unsafe_allow_html=True)

# ==========================================
# 2. MEDICAL KNOWLEDGE BASE
# ==========================================
HEALTH_DATA = {
    "Diabetes Prediction": {
        "precautions": ["Reduce refined sugar and carbohydrate intake", "Aim for 30 minutes of aerobic exercise daily", "Monitor blood glucose levels frequently", "Avoid highly processed foods"],
        "insights": "Diabetes is a chronic metabolic disease characterized by elevated levels of blood glucose. Early detection can prevent severe damage to the heart, blood vessels, eyes, kidneys, and nerves."
    },
    "Heart Disease Prediction": {
        "precautions": ["Adopt a low-cholesterol, low-sodium diet", "Completely avoid smoking and tobacco", "Engage in regular cardiovascular exercise", "Practice daily stress reduction techniques"],
        "insights": "Cardiovascular diseases are the leading cause of mortality globally. Maintaining healthy blood pressure and cholesterol levels significantly mitigates acute risk factors."
    },
    "Parkinson’s Prediction": {
        "precautions": ["Engage in consistent speech and vocal therapy", "Maintain a regular physical therapy schedule", "Consume an antioxidant-rich, balanced diet", "Schedule routine neurological consultations"],
        "insights": "Parkinson's is a progressive nervous system disorder affecting movement. Acoustic alterations like dysphonia are often early clinical indicators of the condition."
    }
}

def generate_report(disease, is_disease, confidence, input_data, precautions, next_steps):
    status = "HIGH RISK DETECTED" if is_disease else "LOW RISK / HEALTHY"
    
    report = f"==========================================\n"
    report += f"   AI HEALTH DIAGNOSTIC REPORT\n"
    report += f"==========================================\n"
    report += f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
    report += f"Module: {disease}\n\n"
    
    report += f"[ DIAGNOSTIC RESULT ]\n"
    report += f"Status: {status}\n"
    report += f"Model Confidence: {confidence*100:.2f}%\n\n"
    
    report += f"[ PATIENT INPUT METRICS ]\n"
    for key, val in input_data.items():
        report += f"- {key}: {val}\n"
        
    report += f"\n[ RECOMMENDED PRECAUTIONS ]\n"
    for p in precautions:
        report += f"- {p}\n"
        
    report += f"\n[ NEXT STEPS ]\n"
    for step in next_steps:
        report += f"- {step}\n"
        
    report += f"\n==========================================\n"
    report += f"Disclaimer: This is an AI-generated predictive report.\n"
    report += f"Do not use this as a substitute for professional medical advice.\n"
    return report

# ==========================================
# 3. CORE LOGIC
# ==========================================
@st.cache_resource(show_spinner=False)
def load_all_artifacts(selected_disease_dropdown):
    di_map = {
        "Diabetes Prediction": "diabetes",
        "Heart Disease Prediction": "heart",
        "Parkinson’s Prediction": "parkinsons"
    }
    prefix = di_map[selected_disease_dropdown]
    models_dir = os.path.join(os.path.dirname(__file__), 'models')
    artifacts = {}
    try:
        with open(os.path.join(models_dir, f'{prefix}_model.pkl'), 'rb') as f:
            artifacts['model'] = pickle.load(f)
        with open(os.path.join(models_dir, f'{prefix}_scaler.pkl'), 'rb') as f:
            artifacts['scaler'] = pickle.load(f)
        with open(os.path.join(models_dir, f'{prefix}_feature_names.pkl'), 'rb') as f:
            artifacts['feature_names'] = pickle.load(f)
        
        metrics_path = os.path.join(models_dir, f'{prefix}_model_metrics.json')
        if os.path.exists(metrics_path):
            with open(metrics_path, 'r') as f:
                artifacts['metrics_df'] = pd.DataFrame(json.load(f))
        else:
            artifacts['metrics_df'] = pd.DataFrame()
    except Exception as e:
        return None
    return artifacts

# ==========================================
# 4. SIDEBAR NAVIGATION
# ==========================================
def render_sidebar():
    st.sidebar.markdown("## 🧭 Navigation")
    selected = st.sidebar.selectbox(
        "Select Diagnostic Engine",
        ["Diabetes Prediction", "Heart Disease Prediction", "Parkinson’s Prediction"]
    )
    st.sidebar.markdown("<br><hr>", unsafe_allow_html=True)
    st.sidebar.markdown("### 📋 Active Configuration")
    st.sidebar.info(f"**Loaded Module:**\n{selected}")
    st.sidebar.caption("Powered by Machine Learning (XGBoost).")
    return selected

# ==========================================
# 5. INPUT FORMS
# ==========================================
def render_input_form(selected_disease, feature_names):
    input_data = {}
    st.markdown("### 📋 Patient Input Metrics")
    
    if selected_disease == "Diabetes Prediction":
        st.markdown("#### 👤 Basic Info")
        c1, c2, c3 = st.columns(3)
        with c1: input_data['Age'] = st.number_input("Age", 1.0, 120.0, 30.0)
        with c2: input_data['Pregnancies'] = st.number_input("Pregnancies", 0.0, 20.0, 0.0)
        with c3: input_data['BMI'] = st.number_input("BMI", 10.0, 80.0, 25.0)

        st.markdown("#### 🔬 Medical Metrics")
        c4, c5, c6 = st.columns(3)
        with c4: input_data['Glucose'] = st.number_input("Glucose", 0.0, 400.0, 120.0)
        with c5: input_data['BloodPressure'] = st.number_input("Blood Pressure", 0.0, 250.0, 70.0)
        with c6: input_data['Insulin'] = st.number_input("Insulin", 0.0, 1200.0, 79.0)
        
        c7, c8 = st.columns(2)
        with c7: input_data['SkinThickness'] = st.number_input("Skin Thickness", 0.0, 100.0, 20.0)
        with c8: input_data['DiabetesPedigreeFunction'] = st.number_input("Diabetes Pedigree Function", 0.0, 3.0, 0.47, step=0.01)
        
    elif selected_disease == "Heart Disease Prediction":
        st.markdown("#### 👤 Patient Info")
        c1, c2 = st.columns(2)
        with c1: input_data['age'] = st.number_input("Age", 1.0, 120.0, 50.0)
        with c2: input_data['sex'] = st.selectbox("Sex (1=Male, 0=Female)", [1.0, 0.0])
        
        st.markdown("#### 🫀 Cardiac Metrics")
        c3, c4, c5 = st.columns(3)
        with c3: input_data['cp'] = st.selectbox("Chest Pain Type (0-3)", [0.0, 1.0, 2.0, 3.0])
        with c4: input_data['trestbps'] = st.number_input("Resting Blood Pressure", 50.0, 300.0, 120.0)
        with c5: input_data['chol'] = st.number_input("Cholesterol", 100.0, 600.0, 200.0)
        
        c6, c7, c8 = st.columns(3)
        with c6: input_data['fbs'] = st.selectbox("Fasting Blood Sugar > 120", [0.0, 1.0])
        with c7: input_data['restecg'] = st.selectbox("Resting ECG (0-2)", [0.0, 1.0, 2.0])
        with c8: input_data['thalach'] = st.number_input("Max Heart Rate", 50.0, 250.0, 150.0)
        
        c9, c10, c11 = st.columns(3)
        with c9: input_data['exang'] = st.selectbox("Exercise Induced Angina", [0.0, 1.0])
        with c10: input_data['oldpeak'] = st.number_input("ST Depression", 0.0, 10.0, 1.0, step=0.1)
        with c11: input_data['slope'] = st.selectbox("Slope of Peak ST (0-2)", [0.0, 1.0, 2.0])
        
        c12, c13 = st.columns(2)
        with c12: input_data['ca'] = st.number_input("Major Vessels (0-4)", 0.0, 4.0, 0.0)
        with c13: input_data['thal'] = st.selectbox("Thalassemia (0-3)", [0.0, 1.0, 2.0, 3.0])

    elif selected_disease == "Parkinson’s Prediction":
        st.markdown("#### 🎙️ Voice Frequency")
        cols = st.columns(3)
        with cols[0]: input_data['MDVP:Fo(Hz)'] = st.number_input("MDVP:Fo(Hz)", value=120.0, step=0.1)
        with cols[1]: input_data['MDVP:Fhi(Hz)'] = st.number_input("MDVP:Fhi(Hz)", value=157.0, step=0.1)
        with cols[2]: input_data['MDVP:Flo(Hz)'] = st.number_input("MDVP:Flo(Hz)", value=75.0, step=0.1)
            
        st.markdown("#### 〰️ Jitter & Shimmer")
        cols2 = st.columns(4)
        with cols2[0]: input_data['MDVP:Jitter(%)'] = st.number_input("Jitter(%)", value=0.006, step=0.0001, format="%.5f")
        with cols2[1]: input_data['MDVP:Jitter(Abs)'] = st.number_input("Jitter(Abs)", value=0.00005, step=0.00001, format="%.6f")
        with cols2[2]: input_data['MDVP:RAP'] = st.number_input("RAP", value=0.003, step=0.001, format="%.4f")
        with cols2[3]: input_data['MDVP:PPQ'] = st.number_input("PPQ", value=0.003, step=0.001, format="%.4f")
        
        cols2_b = st.columns(4)
        with cols2_b[0]: input_data['Jitter:DDP'] = st.number_input("Jitter:DDP", value=0.01, step=0.001, format="%.4f")
        with cols2_b[1]: input_data['MDVP:Shimmer'] = st.number_input("Shimmer", value=0.03, step=0.001, format="%.4f")
        with cols2_b[2]: input_data['MDVP:Shimmer(dB)'] = st.number_input("Shimmer(dB)", value=0.28, step=0.01, format="%.3f")
        with cols2_b[3]: input_data['Shimmer:APQ3'] = st.number_input("Shimmer:APQ3", value=0.015, step=0.001, format="%.4f")
        
        st.markdown("#### 📊 Advanced Acoustic Metrics")
        cols3 = st.columns(4)
        adv = {'Shimmer:APQ5': 0.02, 'MDVP:APQ': 0.025, 'Shimmer:DDA': 0.045, 'NHR': 0.02, 'HNR': 21.0, 'RPDE': 0.5, 'DFA': 0.7, 'spread1': -5.0, 'spread2': 0.2, 'D2': 2.3, 'PPE': 0.2}
        i = 0
        for feat in feature_names:
            if feat not in input_data:
                default_val = float(adv.get(feat, 0.1))
                with cols3[i % 4]: 
                    input_data[feat] = st.number_input(feat, value=default_val, step=0.01)
                i += 1
            
    st.markdown("<br>", unsafe_allow_html=True)
    return input_data

# ==========================================
# MAIN EXECUTION
# ==========================================
def main():
    setup_page()
    inject_custom_css()
    
    # Header Module
    st.markdown("<h1>🧠💉❤️ AI-Based Multi Disease Prediction System</h1>", unsafe_allow_html=True)
    st.markdown("<div class='sub-header'>Predict Diabetes, Heart Disease & Parkinson’s using Machine Learning</div>", unsafe_allow_html=True)
    
    # Render Sidebar
    selected_disease = render_sidebar()
    
    # Setup Tabs
    tab_dashboard, tab_analytics, tab_insights = st.tabs(["🚀 Prediction", "📊 Model Analytics", "🫀 Health Insights"])
    
    # Load Backend Data
    artifacts = load_all_artifacts(selected_disease)
    
    with tab_dashboard:
        if artifacts is None:
            st.error("⚠️ Critical Module Offline. Extracted data models missing. Please run model compilation.")
            st.stop()
            
        # Top Cards
        df_met = artifacts['metrics_df']
        if not df_met.empty:
            c1, c2, c3 = st.columns(3)
            with c1:
                st.markdown(f"""<div class="dashboard-container"><div class="metric-title">Training Accuracy</div><div class="metric-value">{df_met.iloc[0].get('Train_Accuracy',0)*100:.1f}%</div></div>""", unsafe_allow_html=True)
            with c2:
                st.markdown(f"""<div class="dashboard-container"><div class="metric-title">Testing Accuracy</div><div class="metric-value">{df_met.iloc[0].get('Accuracy',0)*100:.1f}%</div></div>""", unsafe_allow_html=True)
            with c3:
                st.markdown(f"""<div class="dashboard-container"><div class="metric-title">Validation F1-Score</div><div class="metric-value">{df_met.iloc[0].get('F1-score',0)*100:.1f}%</div></div>""", unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Form Input
        input_data = render_input_form(selected_disease, artifacts['feature_names'])
        
        # PREDICTION BUTTON LOGIC
        if st.button("Predict"):
            model = artifacts['model']
            scaler = artifacts['scaler']
            feature_names = artifacts['feature_names']
            
            with st.spinner("Analyzing neural pathways & metrics..."):
                time.sleep(1)
                
                input_df = pd.DataFrame([input_data], columns=feature_names)
                scaled_input = scaler.transform(input_df)
                scaled_df = pd.DataFrame(scaled_input, columns=feature_names)
                
                prediction = model.predict(scaled_df)[0]
                probability_array = model.predict_proba(scaled_df)[0]
                
                is_disease = (prediction == 1)
                confidence = probability_array[1] if is_disease else probability_array[0]
                disease_prob = probability_array[1]
                
                target_name = selected_disease.replace(" Prediction", "")
                
            # Render visually outside spinner context
            st.markdown("### 🧬 Diagnostic Results")
            
            if disease_prob < 0.3: 
                box_class, status, next_steps = "result-box-success", "Low Risk / Healthy", ["Maintain routine checkups", "Continue preventive health measures"]
            elif disease_prob < 0.6: 
                box_class, status, next_steps = "result-box-warning", "Moderate Risk", ["Consult a physician for preventive screening", "Adopt lifestyle changes immediately"]
            else: 
                box_class, status, next_steps = "result-box-error", "High Risk Detected", ["Consult a specialist doctor immediately", "Undergo suggested clinical tests", "Enact strict lifestyle changes"]
            
            icon = "✅" if not is_disease else "⚠️"
            color = "#10b981" if not is_disease else "#ef4444"
            
            st.markdown(f"""
            <div class="{box_class}">
                <h1 style="color: {color}; margin: 0; font-size: 2.8rem;">{icon} {status}</h1>
                <hr style="border-color: rgba(255,255,255,0.1); margin: 25px 0;">
                <h3 style="color: #f1f5f9; margin-bottom:5px;">Confidence Score: <span style="color:{color};">{confidence*100:.1f}%</span></h3>
                <p style="color: #94a3b8;">Base Probability Factor: {disease_prob*100:.1f}%</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.progress(float(disease_prob), text=f"Disease Probability Vector ({disease_prob*100:.1f}%)")
            
            st.markdown("---")
            cc1, cc2 = st.columns(2)
            with cc1:
                with st.expander("💊 View Specific Precautions", expanded=True):
                    for p in HEALTH_DATA[selected_disease]['precautions']:
                        st.markdown(f"<div class='advice-card'>• {p}</div>", unsafe_allow_html=True)
            with cc2:
                with st.expander("📋 Recommended Actions", expanded=True):
                    for step in next_steps:
                        st.markdown(f"<div class='advice-card' style='border-left-color: #f59e0b;'>• {step}</div>", unsafe_allow_html=True)
            
            st.markdown("---")
            st.markdown("#### 🧠 Why this prediction?")
            st.caption("Top 3 most important clinical features driving this specific analysis:")
            
            try:
                if hasattr(model, 'feature_importances_'):
                    importances = model.feature_importances_
                    feat_imp_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances}).sort_values(by='Importance', ascending=False)
                    st.bar_chart(feat_imp_df.head(3).set_index('Feature'), color=["#38bdf8"])
                else:
                    st.info("Feature impact tracking unavailable for this model architecture.")
            except Exception as e:
                pass
                
            report_str = generate_report(selected_disease, is_disease, confidence, input_data, HEALTH_DATA[selected_disease]['precautions'], next_steps)
            st.download_button("📄 Download Health Report", data=report_str, file_name=f"Health_Report_{selected_disease.replace(' ','_')}.txt", mime="text/plain")

    with tab_analytics:
        if artifacts and not df_met.empty:
            st.markdown("### 📊 Performance Cross-Validation")
            display_df = df_met.drop(columns=['Confusion_Matrix'], errors='ignore')
            st.dataframe(display_df.style.highlight_max(subset=['Accuracy', 'F1-score'], color='rgba(16, 185, 129, 0.3)', axis=0).format({'Train_Accuracy': '{:.2%}', 'Accuracy': '{:.2%}', 'Precision': '{:.2%}', 'Recall': '{:.2%}', 'F1-score': '{:.2%}'}), use_container_width=True)
            st.markdown(f"#### Architecture Stability Map")
            try:
                st.bar_chart(display_df.set_index("Model Name")[["Train_Accuracy", "Accuracy", "F1-score"]], color=["#10b981", "#3b82f6", "#eab308"])
            except Exception:
                pass

    with tab_insights:
        st.markdown(f"### 🧬 General Medical Context: {selected_disease}")
        st.info(HEALTH_DATA[selected_disease]['insights'])
        st.markdown("#### Universal Disease Prevention Framework:")
        st.markdown("- **Nutrition:** Focus on whole foods, lean proteins, and critical micronutrients.")
        st.markdown("- **Activity:** 150 minutes of moderate aerobic activity per week recommended globally.")
        st.markdown("- **Screening:** Early detection drastically improves survival and management probabilities across all major illnesses.")

    st.markdown("""<div class="footer">Developed by Kavya<br>Powered by Machine Learning</div>""", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
