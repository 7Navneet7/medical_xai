import streamlit as st
import tensorflow as tf
import cv2
import numpy as np
import matplotlib.pyplot as plt
from src.xai_engine import MedicalXAIEngine
from config import config
import tempfile
import os
import time
import io
from PIL import Image

# Set page configuration
st.set_page_config(
    page_title="Medical XAI - Pneumonia Diagnosis",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .diagnosis-box {
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        font-weight: bold;
        text-align: center;
    }
    .normal {
        background-color: #d4edda;
        color: #155724;
        border: 2px solid #c3e6cb;
    }
    .pneumonia {
        background-color: #f8d7da;
        color: #721c24;
        border: 2px solid #f5c6cb;
    }
    .confidence-high {
        color: #28a745;
        font-weight: bold;
    }
    .confidence-medium {
        color: #ffc107;
        font-weight: bold;
    }
    .confidence-low {
        color: #dc3545;
        font-weight: bold;
    }
    .heatmap-container {
        border: 2px solid #e0e0e0;
        border-radius: 10px;
        padding: 10px;
        margin: 10px 0;
        background-color: #f8f9fa;
    }
</style>
""", unsafe_allow_html=True)

# Load model with caching
@st.cache_resource
def load_model():
    """Load the trained model with caching"""
    try:
        model = tf.keras.models.load_model("models/best_model.h5")
        return model
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        return None

def get_confidence_class(confidence):
    """Get CSS class for confidence display"""
    if confidence > 0.9:
        return "confidence-high"
    elif confidence > 0.7:
        return "confidence-medium"
    else:
        return "confidence-low"

def create_xai_visualization(original_image, processed_image, heatmap, prediction, confidence):
    """Create real-time XAI visualization"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Original image
    axes[0,0].imshow(original_image)
    axes[0,0].set_title('Original Chest X-Ray', fontsize=14, fontweight='bold')
    axes[0,0].axis('off')
    
    # Preprocessed image
    axes[0,1].imshow(processed_image)
    axes[0,1].set_title('Preprocessed Image', fontsize=14, fontweight='bold')
    axes[0,1].axis('off')
    
    # Heatmap
    im = axes[1,0].imshow(heatmap, cmap='jet')
    axes[1,0].set_title('Grad-CAM Heatmap', fontsize=14, fontweight='bold')
    axes[1,0].axis('off')
    plt.colorbar(im, ax=axes[1,0], fraction=0.046, pad=0.04)
    
    # Overlay
    overlay = create_heatmap_overlay(heatmap, original_image)
    axes[1,1].imshow(overlay)
    axes[1,1].set_title(f'Explanation Overlay\nPrediction: {prediction}\nConfidence: {confidence:.3f}', 
                       fontsize=14, fontweight='bold')
    axes[1,1].axis('off')
    
    plt.tight_layout()
    
    # Convert matplotlib figure to image
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=150, bbox_inches='tight', facecolor='white')
    buf.seek(0)
    plt.close()
    
    return buf

def create_heatmap_overlay(heatmap, original_image, alpha=0.4):
    """Create heatmap overlay on original image"""
    # Resize heatmap to match original image dimensions
    heatmap_resized = cv2.resize(heatmap, (original_image.shape[1], original_image.shape[0]))
    
    # Apply colormap
    heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    
    # Ensure both images have same data type
    if original_image.dtype != np.uint8:
        original_image_uint8 = (original_image * 255).astype(np.uint8)
    else:
        original_image_uint8 = original_image
    
    # Blend images
    overlayed = cv2.addWeighted(original_image_uint8, 1 - alpha, heatmap_colored, alpha, 0)
    
    return overlayed

def generate_real_time_explanation(model, image_path):
    """Generate XAI explanation in real-time"""
    try:
        # Load and preprocess image
        image = cv2.imread(image_path)
        if image is None:
            return None, "Could not load image"
            
        original_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        processed_image = cv2.resize(original_image, (224, 224))
        processed_image = processed_image.astype(np.float32) / 255.0
        
        # Get prediction
        pred = model.predict(np.expand_dims(processed_image, axis=0), verbose=0)[0]
        predicted_class = np.argmax(pred)
        confidence = pred[predicted_class]
        prediction_label = config.CLASS_NAMES[predicted_class]
        
        # Generate Grad-CAM heatmap
        xai_engine = MedicalXAIEngine(model, config.CLASS_NAMES)
        heatmap, _ = xai_engine.grad_cam(processed_image, class_idx=predicted_class)
        
        # Create visualization
        viz_buffer = create_xai_visualization(
            original_image, processed_image, heatmap, 
            prediction_label, confidence
        )
        
        explanation = {
            'prediction': prediction_label,
            'confidence': float(confidence),
            'all_probabilities': {name: float(prob) for name, prob in zip(config.CLASS_NAMES, pred)},
            'heatmap': heatmap,
            'visualization': viz_buffer
        }
        
        return explanation, None
        
    except Exception as e:
        return None, str(e)

def main():
    # Header
    st.markdown('<h1 class="main-header">🏥 Medical XAI - Pneumonia Diagnosis</h1>', 
                unsafe_allow_html=True)
    
    st.markdown("""
    Welcome to the Medical Explainable AI system! Upload a chest X-ray image, 
    and our AI will provide a diagnosis with **real-time visual explanations** showing exactly 
    **why** it made that decision.
    """)
    
    # Sidebar
    st.sidebar.title("About")
    st.sidebar.info("""
    This system uses Deep Learning with Explainable AI (XAI) to:
    - 🔍 Detect pneumonia from chest X-rays
    - 🎯 Provide accurate diagnoses
    - 👁️ Show real-time visual explanations (Grad-CAM heatmaps)
    - 🏥 Support clinical decision-making
    """)
    
    st.sidebar.title("Model Information")
    st.sidebar.text(f"Model: {config.MODEL_NAME}")
    st.sidebar.text(f"Image Size: {config.IMAGE_SIZE}")
    st.sidebar.text(f"Classes: {', '.join(config.CLASS_NAMES)}")
    
    st.sidebar.title("XAI Heatmap Guide")
    st.sidebar.markdown("""
    **Color Interpretation:**
    - 🔴 **Red/Orange**: High attention - critical diagnostic regions
    - 🟡 **Yellow**: Medium attention - supportive evidence
    - 🟢 **Green**: Low attention - normal tissue
    - 🔵 **Blue**: Minimal attention - background
    """)
    
    st.sidebar.title("Instructions")
    st.sidebar.markdown("""
    1. Upload a chest X-ray image (JPEG/PNG)
    2. Wait for real-time AI analysis
    3. Review diagnosis and confidence
    4. Examine the live XAI heatmaps
    5. Consult with healthcare professional
    """)
    
    # Main content
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📤 Upload X-Ray Image")
        uploaded_file = st.file_uploader(
            "Choose a chest X-ray image", 
            type=['jpeg', 'jpg', 'png'],
            help="Upload a frontal chest X-ray image for analysis"
        )
        
        if uploaded_file is not None:
            # Display uploaded image
            st.image(uploaded_file, caption="Uploaded X-Ray", use_column_width=True)
            
            # File info
            file_size = len(uploaded_file.getvalue()) / 1024
            st.text(f"File: {uploaded_file.name} ({file_size:.1f} KB)")
    
    with col2:
        if uploaded_file is not None:
            st.subheader("🔍 Analysis Results")
            
            # Create progress bar and status
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Load model
            status_text.text("🔄 Loading AI model...")
            model = load_model()
            progress_bar.progress(20)
            
            if model is not None:
                # Save uploaded file temporarily
                status_text.text("🔄 Processing image...")
                with tempfile.NamedTemporaryFile(delete=False, suffix='.jpeg') as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    tmp_path = tmp_file.name
                
                progress_bar.progress(40)
                
                try:
                    # Generate real-time explanation
                    status_text.text("🎨 Generating XAI heatmaps...")
                    explanation, error = generate_real_time_explanation(model, tmp_path)
                    
                    progress_bar.progress(80)
                    
                    if error:
                        st.error(f"❌ Analysis failed: {error}")
                    elif explanation:
                        progress_bar.progress(100)
                        status_text.text("✅ Analysis complete!")
                        
                        # Display results
                        st.success("✅ Real-time Analysis Completed!")
                        
                        # Diagnosis box
                        diagnosis_class = "pneumonia" if explanation['prediction'] == 'PNEUMONIA' else "normal"
                        confidence_class = get_confidence_class(explanation['confidence'])
                        
                        st.markdown(
                            f'<div class="diagnosis-box {diagnosis_class}">'
                            f'<h2>Diagnosis: {explanation["prediction"]}</h2>'
                            f'<h3 class="{confidence_class}">Confidence: {explanation["confidence"]:.1%}</h3>'
                            f'</div>', 
                            unsafe_allow_html=True
                        )
                        
                        # Probabilities
                        st.subheader("📊 Probability Distribution")
                        probs = explanation['all_probabilities']
                        
                        col_prob1, col_prob2 = st.columns(2)
                        with col_prob1:
                            for class_name, prob in probs.items():
                                progress_val = prob
                                st.write(f"**{class_name}**: {prob:.1%}")
                                st.progress(progress_val)
                        
                        # Display real-time XAI visualization
                        st.subheader("👁️ Real-time XAI Explanation")
                        
                        # Create tabs for different views
                        tab1, tab2, tab3, tab4 = st.tabs([
                            "📊 Full Explanation", 
                            "🖼️ Original", 
                            "🔥 Heatmap", 
                            "🎯 Overlay"
                        ])
                        
                        with tab1:
                            st.markdown('<div class="heatmap-container">', unsafe_allow_html=True)
                            st.image(explanation['visualization'], 
                                   caption="Complete XAI Explanation", 
                                   use_column_width=True)
                            st.markdown('</div>', unsafe_allow_html=True)
                        
                        with tab2:
                            # Show original image
                            original_img = cv2.imread(tmp_path)
                            original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
                            st.image(original_img, caption="Original X-Ray", use_column_width=True)
                        
                        with tab3:
                            # Show heatmap only
                            fig, ax = plt.subplots(figsize=(8, 6))
                            im = ax.imshow(explanation['heatmap'], cmap='jet')
                            ax.set_title('Grad-CAM Heatmap', fontweight='bold')
                            ax.axis('off')
                            plt.colorbar(im, ax=ax)
                            st.pyplot(fig)
                            plt.close()
                        
                        with tab4:
                            # Show overlay only
                            original_img = cv2.imread(tmp_path)
                            original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
                            overlay = create_heatmap_overlay(explanation['heatmap'], original_img)
                            st.image(overlay, caption="Heatmap Overlay", use_column_width=True)
                        
                        # Interpretation guide
                        st.subheader("🎓 How to Interpret the Heatmaps")
                        
                        col_guide1, col_guide2 = st.columns(2)
                        
                        with col_guide1:
                            st.markdown("""
                            **Color Meaning:**
                            - 🔴 **Red/Orange**: Critical diagnostic regions
                            - 🟡 **Yellow**: Supportive evidence areas  
                            - 🟢 **Green**: Normal tissue patterns
                            - 🔵 **Blue**: Background/unimportant areas
                            
                            **Clinical Insight:**
                            - The AI highlights medically relevant lung regions
                            - Patterns should match radiological expertise
                            - Verify the AI focuses on correct anatomical areas
                            """)
                        
                        with col_guide2:
                            st.markdown("""
                            **For Pneumonia:**
                            - Look for intense red areas in lung fields
                            - Should highlight consolidation patterns
                            - May show asymmetrical lung involvement
                            
                            **For Normal Cases:**
                            - More diffuse, lower-intensity heatmaps
                            - Focus on clear lung fields
                            - Less concentrated activation
                            """)
                        
                        # Heatmap statistics
                        st.subheader("📈 Heatmap Analysis")
                        heatmap_data = explanation['heatmap']
                        
                        col_stat1, col_stat2, col_stat3 = st.columns(3)
                        
                        with col_stat1:
                            st.metric("Max Intensity", f"{np.max(heatmap_data):.3f}")
                        
                        with col_stat2:
                            st.metric("Mean Intensity", f"{np.mean(heatmap_data):.3f}")
                        
                        with col_stat3:
                            activation_ratio = np.sum(heatmap_data > 0.5) / heatmap_data.size
                            st.metric("High Activation %", f"{activation_ratio:.1%}")
                    
                    else:
                        st.error("❌ Failed to generate explanation. Please try another image.")
                        
                except Exception as e:
                    st.error(f"❌ Error during analysis: {str(e)}")
                    st.info("💡 Please ensure you uploaded a valid chest X-ray image")
                
                finally:
                    # Clean up temporary file
                    if os.path.exists(tmp_path):
                        os.unlink(tmp_path)
            
            # Add a small delay for better UX
            time.sleep(1)
            progress_bar.empty()
            status_text.empty()
        
        else:
            # Show example when no file uploaded
            st.info("👆 Please upload a chest X-ray image to begin real-time analysis")
            
            # Example section
            st.subheader("🚀 Real-time XAI Features")
            st.markdown("""
            This system provides **instant visual explanations**:
            - **Live Heatmap Generation**: See why the AI makes each decision
            - **Interactive Tabs**: Switch between different visualization views
            - **Statistical Analysis**: Quantitative heatmap metrics
            - **Clinical Guidance**: Professional interpretation help
            """)

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center'>
        <p><strong>Medical XAI System</strong> - Real-time Explainable AI for Pneumonia Diagnosis</p>
        <p>Always consult with healthcare professionals for medical diagnoses</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    # Create directories
    os.makedirs("results/web_explanations/", exist_ok=True)
    main()