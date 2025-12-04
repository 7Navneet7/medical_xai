import streamlit as st
import tensorflow as tf
import cv2
import numpy as np
import matplotlib.pyplot as plt
from src.xai_engine import EnhancedMedicalXAIEngine
from config import config
import tempfile
import os
import time
import io
from PIL import Image
from skimage.segmentation import mark_boundaries
import platform



st.set_page_config(
    page_title="Medical XAI - Pneumonia Diagnosis",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

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
    .enhanced-feature {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 10px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .method-status {
        padding: 8px;
        border-radius: 5px;
        margin: 5px 0;
        font-weight: bold;
    }
    .method-success {
        background-color: #d4edda;
        color: #155724;
        border-left: 4px solid #28a745;
    }
    .method-warning {
        background-color: #fff3cd;
        color: #856404;
        border-left: 4px solid #ffc107;
    }
    .method-error {
        background-color: #f8d7da;
        color: #721c24;
        border-left: 4px solid #dc3545;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    model_paths = [
        "models/best_model.h5",
        "models/final_model.h5", 
        "best_model.h5",
        "final_model.h5",
        "models/emergency_model.h5"
    ]
    
    existing_models = [path for path in model_paths if os.path.exists(path)]
    
    if not existing_models:
        st.error("❌ No model files found! Please train the model first.")
        st.info("💡 Run: python train_model.py to train the model")
        return create_emergency_model()
    
    st.info(f"🔍 Found {len(existing_models)} model file(s): {existing_models}")
    
    for model_path in existing_models:
        try:
            st.info(f"🔄 Loading model from: {model_path}")
            
            try:
                model = tf.keras.models.load_model(model_path)
                st.success(f"✅ Model loaded successfully using standard method!")
            except Exception as e1:
                st.warning(f"⚠️ Standard load failed: {e1}")
                
                try:
                    model = tf.keras.models.load_model(
                        model_path,
                        custom_objects=None,
                        compile=False
                    )
                    st.success(f"✅ Model loaded successfully with compile=False!")
                except Exception as e2:
                    st.warning(f"⚠️ Custom objects load failed: {e2}")
                    
                    try:
                        from tensorflow.keras.applications import DenseNet121
                        from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
                        from tensorflow.keras.models import Model
                        
                        base_model = DenseNet121(
                            weights='imagenet' if config.PRETRAINED else None,
                            include_top=False,
                            input_shape=(224, 224, 3)
                        )
                        x = base_model.output
                        x = GlobalAveragePooling2D()(x)
                        predictions = Dense(config.NUM_CLASSES, activation='softmax')(x)
                        model = Model(inputs=base_model.input, outputs=predictions)
                        
                        model.load_weights(model_path)
                        st.success(f"✅ Model weights loaded successfully!")
                    except Exception as e3:
                        st.warning(f"⚠️ Weights-only load failed: {e3}")
                        continue
            
            try:
                test_input = np.random.random((1, 224, 224, 3)).astype(np.float32)
                test_pred = model.predict(test_input, verbose=0)
                st.success(f"✅ Model test passed! Output shape: {test_pred.shape}")
                return model
            except Exception as e:
                st.warning(f"⚠️ Model test failed: {e}")
                return model
                
        except Exception as e:
            st.warning(f"⚠️ Failed to load from {model_path}: {str(e)}")
            continue
    
    st.error("❌ All loading methods failed!")
    return create_emergency_model()

def create_emergency_model():
    st.warning("🚨 Creating emergency demo model for testing...")
    try:
        from tensorflow.keras.applications import DenseNet121
        from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
        from tensorflow.keras.models import Model
        
        base_model = DenseNet121(
            weights='imagenet',
            include_top=False,
            input_shape=(224, 224, 3)
        )
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        predictions = Dense(config.NUM_CLASSES, activation='softmax')(x)
        model = Model(inputs=base_model.input, outputs=predictions)
        
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        os.makedirs("models", exist_ok=True)
        model.save("models/emergency_model.h5")
        st.success("✅ Emergency model created and saved!")
        return model
        
    except Exception as e:
        st.error(f"❌ Emergency model creation failed: {e}")
        return None

def get_confidence_class(confidence):
    if confidence > 0.9:
        return "confidence-high"
    elif confidence > 0.7:
        return "confidence-medium"
    else:
        return "confidence-low"

def create_xai_visualization(original_image, processed_image, heatmap, prediction, confidence):
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    axes[0,0].imshow(original_image)
    axes[0,0].set_title('Original Chest X-Ray', fontsize=14, fontweight='bold')
    axes[0,0].axis('off')
    
    axes[0,1].imshow(processed_image)
    axes[0,1].set_title('Preprocessed Image', fontsize=14, fontweight='bold')
    axes[0,1].axis('off')
    
    im = axes[1,0].imshow(heatmap, cmap='jet')
    axes[1,0].set_title('Grad-CAM Heatmap', fontsize=14, fontweight='bold')
    axes[1,0].axis('off')
    plt.colorbar(im, ax=axes[1,0], fraction=0.046, pad=0.04)
    
    overlay = create_heatmap_overlay(heatmap, original_image)
    axes[1,1].imshow(overlay)
    axes[1,1].set_title(f'Explanation Overlay\nPrediction: {prediction}\nConfidence: {confidence:.3f}', 
                       fontsize=14, fontweight='bold')
    axes[1,1].axis('off')
    
    plt.tight_layout()
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=150, bbox_inches='tight', facecolor='white')
    buf.seek(0)
    plt.close()
    
    return buf

def create_heatmap_overlay(heatmap, original_image, alpha=0.4):
    heatmap_resized = cv2.resize(heatmap, (original_image.shape[1], original_image.shape[0]))
    
    heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    
    if original_image.dtype != np.uint8:
        original_image_uint8 = (original_image * 255).astype(np.uint8)
    else:
        original_image_uint8 = original_image
    
    overlayed = cv2.addWeighted(original_image_uint8, 1 - alpha, heatmap_colored, alpha, 0)
    
    return overlayed

def generate_real_time_explanation(model, image_path):
    try:
        image = cv2.imread(image_path)
        if image is None:
            return None, "Could not load image"
            
        original_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        processed_image = cv2.resize(original_image, (224, 224))
        processed_image = processed_image.astype(np.float32) / 255.0
        
        pred = model.predict(np.expand_dims(processed_image, axis=0), verbose=0)[0]
        predicted_class = np.argmax(pred)
        confidence = pred[predicted_class]
        prediction_label = config.CLASS_NAMES[predicted_class]
        
        xai_engine = EnhancedMedicalXAIEngine(model, config.CLASS_NAMES)
        heatmap, _ = xai_engine.grad_cam(processed_image, class_idx=predicted_class)
        
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

def debug_xai_methods(model, image_path):
    st.write("## 🔧 XAI Method Debug")
    
    try:
        xai_engine = EnhancedMedicalXAIEngine(model, config.CLASS_NAMES)
        st.markdown('<div class="method-status method-success">✅ EnhancedMedicalXAIEngine initialized</div>', unsafe_allow_html=True)
    except Exception as e:
        st.markdown(f'<div class="method-status method-error">❌ XAI Engine failed: {e}</div>', unsafe_allow_html=True)
        return
    
    try:
        image = cv2.imread(image_path)
        st.markdown(f'<div class="method-status method-success">✅ Image loaded: {image.shape}</div>', unsafe_allow_html=True)
    except Exception as e:
        st.markdown(f'<div class="method-status method-error">❌ Image loading failed: {e}</div>', unsafe_allow_html=True)
        return
    
    try:
        import shap
        st.markdown('<div class="method-status method-success">✅ SHAP imported successfully</div>', unsafe_allow_html=True)
    except Exception as e:
        st.markdown(f'<div class="method-status method-error">❌ SHAP import failed: {e}</div>', unsafe_allow_html=True)
    
    try:
        import lime
        st.markdown('<div class="method-status method-success">✅ LIME imported successfully</div>', unsafe_allow_html=True)
    except Exception as e:
        st.markdown(f'<div class="method-status method-error">❌ LIME import failed: {e}</div>', unsafe_allow_html=True)

def safe_comprehensive_explanation(model, image_path):
    try:
        image = cv2.imread(image_path)
        original_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        processed_image = cv2.resize(original_image, (224, 224))
        processed_image = processed_image.astype(np.float32) / 255.0
        
        pred = model.predict(np.expand_dims(processed_image, axis=0), verbose=0)[0]
        predicted_class = np.argmax(pred)
        confidence = pred[predicted_class]
        prediction_label = config.CLASS_NAMES[predicted_class]
        
        explanation = {
            'prediction': prediction_label,
            'confidence': float(confidence),
            'all_probabilities': {name: float(prob) for name, prob in zip(config.CLASS_NAMES, pred)},
            'grad_cam': None,
            'shap': None,
            'lime': None
        }
        
        try:
            xai_engine = EnhancedMedicalXAIEngine(model, config.CLASS_NAMES)
            heatmap, _ = xai_engine.grad_cam(processed_image, class_idx=predicted_class)
            explanation['grad_cam'] = {
                'heatmap': heatmap,
                'overlay': xai_engine.overlay_heatmap(heatmap, original_image)
            }
            st.markdown('<div class="method-status method-success">✅ Grad-CAM analysis completed</div>', unsafe_allow_html=True)
        except Exception as e:
            st.markdown(f'<div class="method-status method-warning">⚠️ Grad-CAM failed: {e}</div>', unsafe_allow_html=True)
        
        try:
            import shap
            st.info("🔄 Computing SHAP values...")
            
            background = np.random.normal(0.5, 0.2, (10, 224, 224, 3))
            
            explainer = shap.GradientExplainer(model, background)
            
            image_batch = np.expand_dims(processed_image, axis=0)
            shap_values = explainer.shap_values(image_batch)
            
            if isinstance(shap_values, list):
                shap_values = shap_values[predicted_class]
            
            if len(shap_values.shape) == 4:
                shap_values = shap_values[0]
            
            if shap_values.size == 0:
                st.warning("SHAP values are empty")
                raise ValueError("Empty SHAP values")
            
            shap_abs = np.abs(shap_values)
            
            explanation['shap'] = {
                'shap_values': shap_values,
                'feature_importance': {
                    'total_impact': np.sum(shap_abs),
                    'positive_impact': np.sum(np.maximum(shap_values, 0)),
                    'negative_impact': np.sum(np.minimum(shap_values, 0)),
                    'mean_impact': np.mean(shap_abs),
                    'max_impact': np.max(shap_abs)
                }
            }
            st.markdown('<div class="method-status method-success">✅ SHAP analysis completed</div>', unsafe_allow_html=True)
            
        except Exception as e:
            st.markdown(f'<div class="method-status method-warning">⚠️ SHAP failed: {e}</div>', unsafe_allow_html=True)
        
        try:
            import lime.lime_image
            explainer = lime.lime_image.LimeImageExplainer()
            
            def model_predict(images):
                return model.predict(images, verbose=0)
            
            lime_exp = explainer.explain_instance(
                processed_image.astype(np.double),
                model_predict,
                top_labels=2,
                hide_color=0,
                num_samples=500
            )
            
            explanation['lime'] = {
                'explanation': lime_exp,
                'feature_importance': {
                    'top_labels': lime_exp.top_labels
                }
            }
            st.markdown('<div class="method-status method-success">✅ LIME analysis completed</div>', unsafe_allow_html=True)
        except Exception as e:
            st.markdown(f'<div class="method-status method-warning">⚠️ LIME failed: {e}</div>', unsafe_allow_html=True)
        
        if explanation['grad_cam']:
            viz_buffer = create_xai_visualization(
                original_image, processed_image, 
                explanation['grad_cam']['heatmap'], 
                explanation['prediction'], explanation['confidence']
            )
            explanation['visualization'] = viz_buffer
        
        return explanation, None
        
    except Exception as e:
        return None, f"Comprehensive analysis failed: {str(e)}"

def display_comprehensive_results(explanation, tmp_path):
    try:
        predicted_class = 0 if explanation['prediction'] == 'NORMAL' else 1
        
        methods_used = []
        if explanation.get('grad_cam'): methods_used.append("Grad-CAM")
        if explanation.get('shap'): methods_used.append("SHAP") 
        if explanation.get('lime'): methods_used.append("LIME")
        
        diagnosis_class = "pneumonia" if explanation['prediction'] == 'PNEUMONIA' else "normal"
        confidence_class = get_confidence_class(explanation['confidence'])
        
        st.markdown(
            f'<div class="diagnosis-box {diagnosis_class}">'
            f'<h2>Diagnosis: {explanation["prediction"]}</h2>'
            f'<h3 class="{confidence_class}">Confidence: {explanation["confidence"]:.1%}</h3>'
            f'<p>XAI Methods: {", ".join(methods_used) if methods_used else "Basic"}</p>'
            f'</div>', 
            unsafe_allow_html=True
        )
        
        if len(methods_used) > 1:
            st.markdown(
                '<div class="enhanced-feature">'
                '🔬 <strong>Enhanced XAI Analysis Active</strong> - Multiple methods used'
                '</div>',
                unsafe_allow_html=True
            )
        
        st.subheader("📊 Probability Distribution")
        probs = explanation['all_probabilities']
        
        col_prob1, col_prob2 = st.columns(2)
        with col_prob1:
            for class_name, prob in probs.items():
                progress_val = prob
                st.write(f"**{class_name}**: {prob:.1%}")
                st.progress(progress_val)
        
        st.subheader("👁️ Comprehensive XAI Explanation")
        
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📊 Full Analysis", 
            "🖼️ Original", 
            "🔥 Grad-CAM", 
            "📈 SHAP",
            "🎯 LIME"
        ])
        
        with tab1:
            if 'visualization' in explanation:
                st.markdown('<div class="heatmap-container">', unsafe_allow_html=True)
                st.image(explanation['visualization'], 
                       caption="Complete XAI Explanation Dashboard", 
                       use_column_width=True)
                st.markdown('</div>', unsafe_allow_html=True)
            else:
                st.warning("No visualization available")
        
        with tab2:
            original_img = cv2.imread(tmp_path)
            original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
            st.image(original_img, caption="Original X-Ray", use_column_width=True)
        
        with tab3:
            if explanation.get('grad_cam'):
                fig, ax = plt.subplots(figsize=(8, 6))
                im = ax.imshow(explanation['grad_cam']['heatmap'], cmap='jet')
                ax.set_title('Grad-CAM Heatmap', fontweight='bold')
                ax.axis('off')
                plt.colorbar(im, ax=ax)
                st.pyplot(fig)
                plt.close()
            else:
                st.warning("Grad-CAM analysis not available")
        
        with tab4:
            if explanation.get('shap'):
                st.success("🔍 SHAP Analysis Active")
                shap_info = explanation['shap'].get('feature_importance', {})
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Total Impact", f"{shap_info.get('total_impact', 0):.3f}")
                    st.metric("Positive Impact", f"{shap_info.get('positive_impact', 0):.3f}")
                with col2:
                    st.metric("Negative Impact", f"{shap_info.get('negative_impact', 0):.3f}")
                
                if 'shap_values' in explanation['shap']:
                    shap_values = explanation['shap']['shap_values']
                    
                    st.write(f"SHAP values shape: {shap_values.shape}")
                    
                    try:
                        if len(shap_values.shape) == 3:
                            if shap_values.shape[2] == 2:
                                shap_heatmap = np.abs(shap_values[:, :, predicted_class])
                                st.write(f"Using predicted class {predicted_class} SHAP values")
                            else:
                                shap_heatmap = np.mean(np.abs(shap_values), axis=2)
                                st.write("Using mean across channels")
                        elif len(shap_values.shape) == 2:
                            shap_heatmap = np.abs(shap_values)
                            st.write("Using 2D SHAP values as is")
                        else:
                            st.error(f"Unexpected SHAP shape: {shap_values.shape}")
                            shap_heatmap = np.abs(shap_values).mean(axis=tuple(range(2, len(shap_values.shape))))
                        
                        if shap_heatmap.size == 0:
                            st.error("SHAP heatmap is empty!")
                            return
                        
                        st.write(f"SHAP heatmap shape: {shap_heatmap.shape}")
                        
                        target_shape = (224, 224)
                        if shap_heatmap.shape[:2] != target_shape and shap_heatmap.size > 0:
                            try:
                                if np.any(np.isnan(shap_heatmap)):
                                    shap_heatmap = np.nan_to_num(shap_heatmap)
                                
                                if shap_heatmap.shape[0] > 0 and shap_heatmap.shape[1] > 0:
                                    shap_heatmap = cv2.resize(shap_heatmap, target_shape)
                                    st.write(f"Resized SHAP heatmap to: {shap_heatmap.shape}")
                                else:
                                    st.warning("Invalid SHAP heatmap dimensions for resizing")
                            except Exception as resize_error:
                                st.warning(f"Could not resize SHAP heatmap: {resize_error}")
                        
                        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
                        
                        original_img = cv2.imread(tmp_path)
                        if original_img is None:
                            st.error("Could not load original image")
                            return
                            
                        original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
                        original_img_resized = cv2.resize(original_img, target_shape)
                        
                        if shap_heatmap.size > 0 and not np.all(shap_heatmap == 0):
                            im1 = ax1.imshow(shap_heatmap, cmap='hot')
                            ax1.set_title('SHAP Values Heatmap\n(Absolute Importance)', fontweight='bold')
                            ax1.axis('off')
                            plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
                        else:
                            ax1.text(0.5, 0.5, 'No SHAP data\nto display', 
                                    ha='center', va='center', transform=ax1.transAxes, fontsize=12)
                            ax1.set_title('SHAP Values Heatmap', fontweight='bold')
                            ax1.axis('off')
                        
                        if shap_heatmap.size > 0 and not np.all(shap_heatmap == 0):
                            try:
                                shap_resized = cv2.resize(shap_heatmap, (original_img_resized.shape[1], original_img_resized.shape[0]))
                                shap_normalized = (shap_resized - np.min(shap_resized)) / (np.max(shap_resized) - np.min(shap_resized) + 1e-8)
                                
                                ax2.imshow(original_img_resized, alpha=0.7)
                                im2 = ax2.imshow(shap_normalized, cmap='hot', alpha=0.5)
                                ax2.set_title('SHAP Overlay on Original Image', fontweight='bold')
                                ax2.axis('off')
                                plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
                            except Exception as overlay_error:
                                ax2.imshow(original_img_resized)
                                ax2.text(0.5, 0.5, 'Overlay failed', 
                                        ha='center', va='center', transform=ax2.transAxes, fontsize=12)
                                ax2.set_title('Original Image', fontweight='bold')
                                ax2.axis('off')
                        else:
                            ax2.imshow(original_img_resized)
                            ax2.set_title('Original Image', fontweight='bold')
                            ax2.axis('off')
                        
                        plt.tight_layout()
                        st.pyplot(fig)
                        plt.close()
                        
                        if shap_heatmap.size > 0 and not np.all(shap_heatmap == 0):
                            st.subheader("📊 SHAP Feature Analysis")
                            
                            col_shap1, col_shap2, col_shap3 = st.columns(3)
                            
                            with col_shap1:
                                flat_shap = shap_heatmap.flatten()
                                if len(flat_shap) > 0:
                                    top_percentile = np.percentile(flat_shap, 95)
                                    high_impact_ratio = np.sum(flat_shap > top_percentile) / len(flat_shap)
                                    st.metric("High Impact Regions", f"{high_impact_ratio:.1%}")
                                else:
                                    st.metric("High Impact Regions", "N/A")
                            
                            with col_shap2:
                                if len(flat_shap) > 0:
                                    mean_impact = np.mean(flat_shap)
                                    st.metric("Mean Impact per Pixel", f"{mean_impact:.4f}")
                                else:
                                    st.metric("Mean Impact per Pixel", "N/A")
                            
                            with col_shap3:
                                if len(flat_shap) > 0 and np.sum(flat_shap) > 0:
                                    top_10_percent = np.sum(np.sort(flat_shap)[-len(flat_shap)//10:]) / np.sum(flat_shap)
                                    st.metric("Top 10% Concentration", f"{top_10_percent:.1%}")
                                else:
                                    st.metric("Top 10% Concentration", "N/A")
                    
                    except Exception as e:
                        st.error(f"Error processing SHAP visualization: {e}")
                        import traceback
                        st.code(traceback.format_exc())
            else:
                st.warning("SHAP analysis not available for this image")
        
        with tab5:
            if explanation.get('lime'):
                st.success("🎯 LIME Analysis Active")
                lime_info = explanation['lime'].get('feature_importance', {})
                if lime_info:
                    available_labels = list(lime_info.keys())
                    st.write(f"Available for {len(available_labels)} class(es)")
                    
                    try:
                        lime_exp = explanation['lime']['explanation']
                        temp, mask = lime_exp.get_image_and_mask(
                            lime_exp.top_labels[0],
                            positive_only=True,
                            num_features=10,
                            hide_rest=False
                        )
                        fig, ax = plt.subplots(figsize=(8, 6))
                        ax.imshow(mark_boundaries(temp, mask))
                        ax.set_title('LIME Explanation', fontweight='bold')
                        ax.axis('off')
                        st.pyplot(fig)
                        plt.close()
                    except Exception as e:
                        st.warning(f"Could not generate LIME visualization: {e}")
            else:
                st.warning("LIME analysis not available for this image")
        
        st.subheader("🎓 Comprehensive XAI Interpretation")
        
        col_guide1, col_guide2 = st.columns(2)
        
        with col_guide1:
            st.markdown("""
            **Multi-Method Analysis:**
            - 🔥 **Grad-CAM**: Visual attention heatmaps
            - 📊 **SHAP**: Feature importance values
            - 🎯 **LIME**: Local interpretable models
            - 📈 **Combined**: Cross-verification of results
            
            **Clinical Benefits:**
            - Multiple XAI methods increase reliability
            - Cross-validation of important regions
            - Quantitative feature importance scores
            """)
        
        with col_guide2:
            st.markdown("""
            **Confidence Indicators:**
            - **High Confidence**: All methods agree on important regions
            - **Medium Confidence**: Some disagreement between methods
            - **Low Confidence**: Significant method disagreement
            
            **Action Steps:**
            - Verify highlighted regions match clinical expertise
            - Check for consistent patterns across methods
            - Consult radiologist for ambiguous cases
            """)
        
        st.subheader("🤝 Method Agreement Analysis")
        
        agreement_col1, agreement_col2, agreement_col3 = st.columns(3)
        
        with agreement_col1:
            methods_count = sum(1 for method in ['grad_cam', 'shap', 'lime'] if explanation.get(method))
            st.metric("Methods Used", f"{methods_count}/3")
        
        with agreement_col2:
            if explanation['confidence'] > 0.8:
                agreement_status = "High"
            elif explanation['confidence'] > 0.6:
                agreement_status = "Medium"
            else:
                agreement_status = "Low"
            st.metric("Confidence Level", agreement_status)
        
        with agreement_col3:
            analysis_depth = "Enhanced" if methods_count > 1 else "Basic"
            st.metric("Analysis Depth", analysis_depth)
        
    except Exception as e:
        st.error(f"Error displaying comprehensive results: {e}")

def display_basic_results(explanation, tmp_path):
    diagnosis_class = "pneumonia" if explanation['prediction'] == 'PNEUMONIA' else "normal"
    confidence_class = get_confidence_class(explanation['confidence'])
    
    st.markdown(
        f'<div class="diagnosis-box {diagnosis_class}">'
        f'<h2>Diagnosis: {explanation["prediction"]}</h2>'
        f'<h3 class="{confidence_class}">Confidence: {explanation["confidence"]:.1%}</h3>'
        f'</div>', 
        unsafe_allow_html=True
    )
    
    st.subheader("📊 Probability Distribution")
    probs = explanation['all_probabilities']
    
    col_prob1, col_prob2 = st.columns(2)
    with col_prob1:
        for class_name, prob in probs.items():
            progress_val = prob
            st.write(f"**{class_name}**: {prob:.1%}")
            st.progress(progress_val)
    
    st.subheader("👁️ Real-time XAI Explanation")
    
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
        original_img = cv2.imread(tmp_path)
        original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
        st.image(original_img, caption="Original X-Ray", use_column_width=True)
    
    with tab3:
        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(explanation['heatmap'], cmap='jet')
        ax.set_title('Grad-CAM Heatmap', fontweight='bold')
        ax.axis('off')
        plt.colorbar(im, ax=ax)
        st.pyplot(fig)
        plt.close()
    
    with tab4:
        original_img = cv2.imread(tmp_path)
        original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
        overlay = create_heatmap_overlay(explanation['heatmap'], original_img)
        st.image(overlay, caption="Heatmap Overlay", use_column_width=True)


IS_MAC = platform.system() == "Darwin"

if IS_MAC:
    st.info("🖥️ Running on macOS - TensorFlow Metal acceleration enabled")
else:
    st.info("☁️ Running on Linux / Streamlit Cloud")      

def main():
    st.markdown('<h1 class="main-header">🏥 Medical XAI - Pneumonia Diagnosis</h1>', 
                unsafe_allow_html=True)
    
    st.markdown("""
    Welcome to the Medical Explainable AI system! Upload a chest X-ray image, 
    and our AI will provide a diagnosis with **real-time visual explanations** showing exactly 
    **why** it made that decision.
    """)
    
    model_files_exist = any(os.path.exists(f"models/{f}") for f in ["best_model.h5", "final_model.h5", "emergency_model.h5"])
    
    if not model_files_exist:
        st.warning("🚨 No trained model found!")
        if st.button("🛠️ Create Emergency Model for Testing"):
            with st.spinner("Creating emergency model..."):
                model = create_emergency_model()
                if model:
                    st.success("✅ Emergency model created successfully!")
                    st.rerun()
    
    st.sidebar.title("About")
    st.sidebar.info("""
    This system uses Deep Learning with Explainable AI (XAI) to:
    - 🔍 Detect pneumonia from chest X-rays
    - 🎯 Provide accurate diagnoses
    - 👁️ Show real-time visual explanations
    - 🔬 Support multiple XAI methods (Grad-CAM, SHAP, LIME)
    - 🏥 Support clinical decision-making
    """)
    
    st.sidebar.title("XAI Methods")
    st.sidebar.markdown("""
    **Available XAI Techniques:**
    - 🔥 **Grad-CAM**: Visual attention heatmaps
    - 📊 **SHAP**: Feature importance analysis  
    - 🎯 **LIME**: Local interpretable explanations
    - 🏥 **Multi-Method**: Comprehensive analysis
    """)
    
    st.sidebar.title("Model Information")
    st.sidebar.text(f"Model: {config.MODEL_NAME}")
    st.sidebar.text(f"Image Size: {config.IMAGE_SIZE}")
    st.sidebar.text(f"Classes: {', '.join(config.CLASS_NAMES)}")
    
    st.sidebar.title("Instructions")
    st.sidebar.markdown("""
    1. Upload a chest X-ray image (JPEG/PNG)
    2. Choose analysis type (Basic/Enhanced)
    3. Wait for real-time AI analysis
    4. Review diagnosis and confidence
    5. Examine comprehensive XAI explanations
    """)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📤 Upload X-Ray Image")
        uploaded_file = st.file_uploader(
            "Choose a chest X-ray image", 
            type=['jpeg', 'jpg', 'png'],
            help="Upload a frontal chest X-ray image for analysis"
        )
        
        if uploaded_file is not None:
            st.image(uploaded_file, caption="Uploaded X-Ray", use_column_width=True)
            
            file_size = len(uploaded_file.getvalue()) / 1024
            st.text(f"File: {uploaded_file.name} ({file_size:.1f} KB)")
            
            use_enhanced_xai = st.checkbox(
                "🔬 Enable Enhanced XAI Analysis", 
                value=False,
                help="Use multiple XAI methods (Grad-CAM + SHAP + LIME) for comprehensive analysis"
            )
    
    with col2:
        if uploaded_file is not None:
            st.subheader("🔍 Analysis Results")
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            status_text.text("🔄 Loading AI model...")
            model = load_model()
            progress_bar.progress(20)
            
            if model is None:
                st.error("❌ Cannot proceed without a model. Please create a demo model first.")
                progress_bar.empty()
                status_text.empty()
                return
            
            status_text.text("🔄 Processing image...")
            with tempfile.NamedTemporaryFile(delete=False, suffix='.jpeg') as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_path = tmp_file.name
            
            progress_bar.progress(40)
            
            with st.expander("🔧 Debug Information", expanded=False):
                debug_xai_methods(model, tmp_path)
            
            try:
                if use_enhanced_xai:
                    status_text.text("🔬 Generating comprehensive XAI analysis...")
                    
                    with st.expander("XAI Method Status", expanded=True):
                        st.write("🔄 Initializing enhanced analysis...")
                        st.write("• Grad-CAM: Loading...")
                        st.write("• SHAP: Loading...") 
                        st.write("• LIME: Loading...")
                    
                    explanation, error = safe_comprehensive_explanation(model, tmp_path)
                else:
                    status_text.text("🎨 Generating XAI heatmaps...")
                    explanation, error = generate_real_time_explanation(model, tmp_path)
                
                progress_bar.progress(80)
                
                if error:
                    st.error(f"❌ Analysis failed: {error}")
                elif explanation:
                    progress_bar.progress(100)
                    status_text.text("✅ Analysis complete!")
                    
                    st.success("✅ Real-time Analysis Completed!")
                    
                    if use_enhanced_xai:
                        display_comprehensive_results(explanation, tmp_path)
                    else:
                        display_basic_results(explanation, tmp_path)
                
                else:
                    st.error("❌ Failed to generate explanation. Please try another image.")
                    
            except Exception as e:
                st.error(f"❌ Error during analysis: {str(e)}")
                st.info("💡 Please ensure you uploaded a valid chest X-ray image")
            
            finally:
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)
        
        else:
            st.info("👆 Please upload a chest X-ray image to begin real-time analysis")

if __name__ == "__main__":
    os.makedirs("results/web_explanations/", exist_ok=True)
    os.makedirs("results/comprehensive_explanations/", exist_ok=True)
    main()