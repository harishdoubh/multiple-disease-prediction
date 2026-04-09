import streamlit as st

def get_medical_advice(prediction_result):
    """
    Returns a disclaimer based on the prediction result.
    """
    if prediction_result == "Diabetic":
        return "Disclaimer: This is a predictive tool, not a medical diagnosis. Please consult a healthcare professional for proper testing and advice."
    else:
        return "Disclaimer: This result indicates a lower risk, but maintaining a healthy lifestyle is always recommended. Consult a doctor for any health concerns."

def format_percentage(confidence):
    """
    Formats the confidence probability into a percentage string.
    """
    return f"{confidence * 100:.2f}%"

def render_sidebar():
    """
    Renders sidebar elements and returns useful navigational facts.
    """
    st.sidebar.title("Diabetes Prediction System")
    st.sidebar.info(
        "Phase 1 of a Multi-Disease Prediction System.\n\n"
        "Currently supports: **Diabetes**\n\n"
        "Future Updates: Heart Disease, Parkinson's."
    )
    st.sidebar.markdown("---")

def categorize_risk(probability):
    """
    Categorizes risk based on confidence probability.
    Returns the risk text and its corresponding Streamlit color status.
    """
    if probability < 0.30:
        return "Low Risk", "success"
    elif probability < 0.60:
        return "Moderate Risk", "warning"
    else:
        return "High Risk", "error"

def generate_recommendations(risk_level):
    """
    Generates personalized recommendations based on the predicted risk level.
    """
    if risk_level == "Low Risk":
        return [
            "Maintain balanced diet",
            "Regular exercise (30 mins daily)",
            "Monitor blood sugar annually",
            "Avoid excessive sugar intake"
        ]
    elif risk_level == "Moderate Risk":
        return [
            "Reduce refined carbs and sugary foods",
            "Increase physical activity",
            "Monitor glucose every 3–6 months",
            "Maintain healthy BMI",
            "Manage stress levels"
        ]
    else:
        return [
            "Consult a healthcare professional immediately",
            "Strict low-sugar diet",
            "Regular glucose monitoring",
            "Weight management plan",
            "Lifestyle modification",
            "Avoid smoking and alcohol"
        ]
