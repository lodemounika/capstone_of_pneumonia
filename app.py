import streamlit as st
import pandas as pd
import numpy as np
import joblib
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

# ==========================
# Page Config
# ==========================
st.set_page_config(
    page_title="AI Healthcare System",
    layout="wide"
)

st.title("🩺 AI-Driven Healthcare Decision Support System")
st.caption("Diabetes Risk Prediction & Chest X-ray Disease Detection")

# ==========================
# Load Models
# ==========================
@st.cache_resource
def load_diabetes_model():
    model = joblib.load("diabetes_rf_model.pkl")
    scaler = joblib.load("scaler.pkl")
    return model, scaler

@st.cache_resource
def load_xray_model():
    return load_model("chest_model.h5")

diabetes_model, scaler = load_diabetes_model()
xray_model = load_xray_model()

# ==========================
# Tabs
# ==========================
tabs = st.tabs([
    "🧪 Diabetes Prediction",
    "🫁 Chest X-ray Detection",
    "📊 KPIs",
    "📘 About Project"
])

# ==================================================
# TAB 1: DIABETES
# ==================================================
with tabs[0]:
    st.subheader("🧪 Diabetes Risk Prediction")

    col1, col2 = st.columns(2)

    with col1:
        pregnancies = st.number_input("Pregnancies", 0, 20, 3)
        glucose = st.number_input("Glucose", 50, 200, 120)
        bp = st.number_input("Blood Pressure", 40, 120, 70)
        skin = st.number_input("Skin Thickness", 0, 100, 20)

    with col2:
        insulin = st.number_input("Insulin", 0, 900, 79)
        bmi = st.number_input("BMI", 10.0, 70.0, 32.0)
        dpf = st.number_input("Diabetes Pedigree Function", 0.0, 2.5, 0.5)
        age = st.number_input("Age", 10, 100, 33)

    if st.button("🔍 Predict Diabetes Risk"):
        input_df = pd.DataFrame([[pregnancies, glucose, bp, skin,
                                  insulin, bmi, dpf, age]],
                                columns=[
                                    "Pregnancies", "Glucose", "BloodPressure",
                                    "SkinThickness", "Insulin", "BMI",
                                    "DiabetesPedigreeFunction", "Age"
                                ])

        X_scaled = scaler.transform(input_df)
        risk = diabetes_model.predict_proba(X_scaled)[0][1]

        st.metric("Diabetes Risk Probability", f"{risk*100:.2f}%")

        if risk > 0.6:
            st.error("⚠ High Risk of Diabetes")
        elif risk > 0.3:
            st.warning("⚠ Moderate Risk")
        else:
            st.success("✅ Low Risk")

# ==================================================
# TAB 2: CHEST X-RAY
# ==================================================
with tabs[1]:
    st.subheader("🫁 Chest X-ray Disease Detection")

    uploaded_image = st.file_uploader(
        "Upload Chest X-ray Image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_image:
        image = Image.open(uploaded_image).convert("RGB")
        st.image(image, caption="Uploaded X-ray", width=350)

        IMG_SIZE = (224, 224)
        img_array = img_to_array(image.resize(IMG_SIZE)) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        preds = xray_model.predict(img_array)
        class_names = ["COVID-19", "Normal", "Pneumonia"]

        predicted_class = class_names[np.argmax(preds)]
        confidence = np.max(preds) * 100

        st.metric("Prediction", predicted_class)
        st.metric("Confidence", f"{confidence:.2f}%")

# ==================================================
# TAB 3: KPIs
# ==================================================
with tabs[2]:
    st.subheader("📊 Key Performance Indicators")

    col1, col2, col3 = st.columns(3)

    col1.metric("Diabetes Accuracy", "78%")
    col2.metric("X-ray CNN Accuracy", "92%")
    col3.metric("Total Records", "768 Patients + 4500 Images")

# ==================================================
# TAB 4: ABOUT
# ==================================================
with tabs[3]:
    st.subheader("📘 About This Project")

    st.write("""
    **AI-Driven Healthcare Decision Support System**

    ✔ Machine Learning for Diabetes Prediction  
    ✔ Deep Learning (CNN + Transfer Learning) for Chest X-ray Analysis  
    ✔ Real-time Predictions via Web Interface  
    ✔ Built using Python, Scikit-Learn, TensorFlow & Streamlit  

    This system assists clinicians by providing fast, data-driven
    predictions to support early diagnosis and decision-making.
    """)

    st.success("🎯 Capstone-ready | Demo-ready | Viva-ready")