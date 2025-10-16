import streamlit as st
import tensorflow as tf
import cv2
import numpy as np
import matplotlib.pyplot as plt
from src.xai_engine import MedicalXAIEngine
from config import config
import tempfile
import os

# Load model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("models/best_model.h5")

model = load_model()
xai_engine = MedicalXAIEngine(model, config.CLASS_NAMES)

st.title("🏥 Medical XAI - Pneumonia Diagnosis")
st.write("Upload a chest X-ray image for AI diagnosis with explanations")

uploaded_file = st.file_uploader("Choose a chest X-ray image", type=['jpeg', 'jpg', 'png'])

if uploaded_file is not None:
    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix='.jpeg') as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_path = tmp_file.name
    
    # Generate explanation
    explanation = xai_engine.generate_explanation_report(tmp_path, "results/web_explanations/")
    
    if explanation:
        st.success(f"**Diagnosis**: {explanation['prediction']}")
        st.info(f"**Confidence**: {explanation['confidence']:.3f}")
        
        # Show the saved visualization
        viz_path = f"results/web_explanations/XAI_{explanation['prediction']}_{explanation['confidence']:.3f}_{uploaded_file.name.split('.')[0]}.png"
        if os.path.exists(viz_path):
            st.image(viz_path, caption="XAI Explanation", use_column_width=True)
    
    # Clean up
    os.unlink(tmp_path)