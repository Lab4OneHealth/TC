import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
import numpy as np

# Page configuration - wide layout for side-by-side arrangement
st.set_page_config(page_title="Thyroid Cancer Recurrence Prediction", layout="wide")

# -------------------------- Load Resources --------------------------
@st.cache_resource
def load_resources():
    model = joblib.load("stacking_model.pkl")
    features = joblib.load("features.pkl")
    label_encoders = joblib.load("label_encoders.pkl")
    shap_background = joblib.load("shap_background.pkl")
    return model, features, label_encoders, shap_background

stacking_model, features, label_encoders, shap_background = load_resources()

@st.cache_resource
def init_shap_explainer():
    explainer = shap.Explainer(
        lambda x: stacking_model.predict_proba(x),
        shap_background,
        seed=42
    )
    return explainer

explainer = init_shap_explainer()

# -------------------------- Categories and Mappings --------------------------
categories = {
    "Gender": ["F", "M"],
    "Smoking": ["No", "Yes"],
    "Hx Smoking": ["No", "Yes"],
    "Hx Radiothreapy": ["No", "Yes"],
    "Thyroid Function": [
        "Clinical Hyperthyroidism", "Clinical Hypothyroidism", "Euthyroid",
        "Subclinical Hyperthyroidism", "Subclinical Hypothyroidism"
    ],
    "Physical Examination": [
        "Diffuse goiter", "Multinodular goiter", "Normal",
        "Single nodular goiter-left", "Single nodular goiter-right"
    ],
    "Adenopathy": ["Bilateral", "Extensive", "Left", "No", "Posterior", "Right"],
    "Pathology": ["Follicular", "Hurthel cell", "Micropapillary", "Papillary"],
    "Focality": ["Multi-Focal", "Uni-Focal"],
    "Risk": ["High", "Intermediate", "Low"],
    "T": ["T1a", "T1b", "T2", "T3a", "T3b", "T4a", "T4b"],
    "N": ["N0", "N1a", "N1b"],
    "M": ["M0", "M1"],
    "Stage": ["I", "II", "III", "IVA", "IVB"],
    "Response": ["Biochemical Incomplete", "Excellent", "Indeterminate", "Structural Incomplete"]
}

risk_mapping = {"High": 2, "Intermediate": 1, "Low": 0}
response_mapping = {
    "Biochemical Incomplete": 2,
    "Excellent": 0,
    "Indeterminate": 1,
    "Structural Incomplete": 3
}

default_values = {
    "Age": 45,
    "Gender": "F",
    "Smoking": "No",
    "Hx Smoking": "Yes",
    "Hx Radiothreapy": "No",
    "Thyroid Function": "Euthyroid",
    "Physical Examination": "Diffuse goiter",
    "Adenopathy": "Posterior",
    "Pathology": "Papillary",
    "Focality": "Multi-Focal",
    "Risk": "Intermediate",
    "T": "T1b",
    "N": "N1a",
    "M": "M0",
    "Stage": "III",
    "Response": "Structural Incomplete"
}

# -------------------------- Main Page Layout --------------------------
st.title("🧠 Thyroid Cancer Recurrence Risk Prediction System")
st.divider()

# Side-by-side columns: form on left, results on right
left_col, right_col = st.columns([0.45, 0.55])  # Left slightly narrower, right slightly wider

# -------------------------- Left Column: Input Form --------------------------
with left_col:
    st.subheader("📋 Input Patient Clinical Feature")
    with st.form("input_form", clear_on_submit=False):
        input_data = {}

        # Compact grouped arrangement
        col1, col2 = st.columns(2)
        with col1:
            input_data["Age"] = st.number_input(
                "Age", min_value=15, max_value=82,
                value=default_values["Age"], step=1, format="%d"
            )
        with col2:
            
            selected = st.selectbox(
                "Gender", categories["Gender"],
                index=categories["Gender"].index(default_values["Gender"]),
                label_visibility="visible"  
            )
            input_data["Gender"] = label_encoders["Gender"].transform([selected])[0]

        # Lifestyle habits
        st.markdown("### Lifestyle Habits")
        col1, col2 = st.columns(2)
        with col1:
            selected = st.selectbox("Smoking", categories["Smoking"],
                index=categories["Smoking"].index(default_values["Smoking"]))
            input_data["Smoking"] = label_encoders["Smoking"].transform([selected])[0]
        with col2:
            selected = st.selectbox("Hx Smoking", categories["Hx Smoking"],
                index=categories["Hx Smoking"].index(default_values["Hx Smoking"]))
            input_data["Hx Smoking"] = label_encoders["Hx Smoking"].transform([selected])[0]
        selected = st.selectbox("Hx Radiothreapy", categories["Hx Radiothreapy"],
            index=categories["Hx Radiothreapy"].index(default_values["Hx Radiothreapy"]))
        input_data["Hx Radiothreapy"] = label_encoders["Hx Radiothreapy"].transform([selected])[0]

        # Thyroid function
        st.markdown("### Thyroid Function")
        selected = st.selectbox("Thyroid Function", categories["Thyroid Function"],
            index=categories["Thyroid Function"].index(default_values["Thyroid Function"]))
        input_data["Thyroid Function"] = label_encoders["Thyroid Function"].transform([selected])[0]
        selected = st.selectbox("Physical Examination", categories["Physical Examination"],
            index=categories["Physical Examination"].index(default_values["Physical Examination"]))
        input_data["Physical Examination"] = label_encoders["Physical Examination"].transform([selected])[0]

        # Pathological features
        st.markdown("### Pathological Features")
        col1, col2 = st.columns(2)
        with col1:
            selected = st.selectbox("Adenopathy", categories["Adenopathy"],
                index=categories["Adenopathy"].index(default_values["Adenopathy"]))
            input_data["Adenopathy"] = label_encoders["Adenopathy"].transform([selected])[0]
        with col2:
            selected = st.selectbox("Pathology", categories["Pathology"],
                index=categories["Pathology"].index(default_values["Pathology"]))
            input_data["Pathology"] = label_encoders["Pathology"].transform([selected])[0]
        selected = st.selectbox("Focality", categories["Focality"],
            index=categories["Focality"].index(default_values["Focality"]))
        input_data["Focality"] = label_encoders["Focality"].transform([selected])[0]

        # Staging and risk
        st.markdown("### Staging and Risk")
        col1, col2 = st.columns(2)
        with col1:
            selected = st.selectbox("Risk", categories["Risk"],
                index=categories["Risk"].index(default_values["Risk"]))
            input_data["Risk"] = risk_mapping[selected]
        with col2:
            selected = st.selectbox("Stage", categories["Stage"],
                index=categories["Stage"].index(default_values["Stage"]))
            input_data["Stage"] = label_encoders["Stage"].transform([selected])[0]

        col1, col2, col3 = st.columns(3)
        with col1:
            selected = st.selectbox("T", categories["T"],
                index=categories["T"].index(default_values["T"]))
            input_data["T"] = label_encoders["T"].transform([selected])[0]
        with col2:
            selected = st.selectbox("N", categories["N"],
                index=categories["N"].index(default_values["N"]))
            input_data["N"] = label_encoders["N"].transform([selected])[0]
        with col3:
            selected = st.selectbox("M", categories["M"],
                index=categories["M"].index(default_values["M"]))
            input_data["M"] = label_encoders["M"].transform([selected])[0]

        # Treatment response
        selected = st.selectbox("Response", categories["Response"],
            index=categories["Response"].index(default_values["Response"]))
        input_data["Response"] = response_mapping[selected]

        # Submit button
        submitted = st.form_submit_button("🚀 Submit & Predict", use_container_width=True)

# -------------------------- Right Column: Results and Visualization --------------------------
with right_col:
    if not submitted:
        # Show prompt when not submitted
        st.info("Please fill in patient clinical feature on the left and click the predict button")
    else:
        try:
            # Construct input data
            X_input = pd.DataFrame([input_data]).reindex(columns=features, fill_value=0)

            # Model prediction
            y_pred = stacking_model.predict(X_input)
            y_prob = stacking_model.predict_proba(X_input)[:, 1] if hasattr(stacking_model, "predict_proba") else [0]
            y_label = label_encoders["Recurred"].inverse_transform(y_pred)[0]

            # Display prediction results
            st.subheader("🎯 Prediction Results")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Predicted Class", y_label)
            with col2:
                st.metric("Recurrence Probability", f"{y_prob[0] * 100:.1f}%")

            # SHAP explanation (compact version)
            st.subheader("🔍 Model Explanation")
            #st.markdown("Red indicates increased recurrence risk, blue indicates decreased recurrence risk")

            with st.spinner("Generating explanation..."):
                shap_values = explainer(X_input)
                single_shap = shap_values[0, :, 1]

                # Adjust plot size to fit right column
                plt.figure(figsize=(8, 6))  # Width fits right column, height compressed
                shap.plots.waterfall(
                    single_shap,
                    max_display=20, 
                    show=False
                )
                plt.tight_layout(pad=0.5)  # Minimize margins
                st.pyplot(plt.gcf())
                plt.close()

        except Exception as e:
            st.error(f"Error processing request: {str(e)}")

# Footer information (global display)
st.divider()
