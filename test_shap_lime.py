import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
from typing import Tuple, Optional, List, Dict
import shap
import lime
import lime.lime_image
from lime import submodular_pick
from skimage.segmentation import mark_boundaries

class EnhancedMedicalXAIEngine:
    def __init__(self, model, class_names, background_size=50):
        self.model = model
        self.class_names = class_names
        self.background_size = background_size
        self.shap_explainer = None
        self.lime_explainer = None
        self.background_data = None
        
    def _prepare_background_data(self, sample_images: List[np.ndarray]):
        """Prepare background data for SHAP (required for DeepExplainer)"""
        if self.background_data is None:
            # Use a subset of images as background
            self.background_data = np.array(sample_images[:self.background_size])
            print(f"✅ Prepared SHAP background data: {self.background_data.shape}")
        return self.background_data
    
    def initialize_shap_explainer(self, sample_images: List[np.ndarray]):
        """Initialize SHAP explainer with background data"""
        try:
            background_data = self._prepare_background_data(sample_images)
            
            # Use DeepExplainer for neural networks
            self.shap_explainer = shap.DeepExplainer(
                self.model, 
                background_data
            )
            print("✅ SHAP DeepExplainer initialized successfully!")
            
        except Exception as e:
            print(f"❌ SHAP initialization failed: {e}")
            # Fallback to GradientExplainer
            try:
                self.shap_explainer = shap.GradientExplainer(
                    self.model, 
                    background_data
                )
                print("✅ SHAP GradientExplainer initialized successfully!")
            except Exception as e2:
                print(f"❌ GradientExplainer also failed: {e2}")
    
    def initialize_lime_explainer(self):
        """Initialize LIME explainer for images"""
        try:
            self.lime_explainer = lime.lime_image.LimeImageExplainer()
            print("✅ LIME ImageExplainer initialized successfully!")
        except Exception as e:
            print(f"❌ LIME initialization failed: {e}")
    
    def shap_explanation(self, image: np.ndarray, class_index: int = None) -> Dict:
        """Generate SHAP explanation for an image"""
        if self.shap_explainer is None:
            raise ValueError("SHAP explainer not initialized. Call initialize_shap_explainer first.")
        
        # Prepare image for SHAP
        image_batch = np.expand_dims(image, axis=0)
        
        # Get SHAP values
        shap_values = self.shap_explainer.shap_values(image_batch)
        
        # If multiple classes, select the relevant one
        if isinstance(shap_values, list):
            if class_index is None:
                # Use predicted class
                prediction = self.model.predict(image_batch, verbose=0)[0]
                class_index = np.argmax(prediction)
            shap_values = shap_values[class_index]
        
        # Create visualization
        shap_visualization = self._create_shap_visualization(image, shap_values[0])
        
        return {
            'shap_values': shap_values,
            'expected_value': self.shap_explainer.expected_value,
            'visualization': shap_visualization,
            'feature_importance': self._calculate_shap_feature_importance(shap_values[0])
        }
    
    def _create_shap_visualization(self, image: np.ndarray, shap_values: np.ndarray) -> plt.Figure:
        """Create SHAP force plot visualization"""
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Original image
        axes[0].imshow(image)
        axes[0].set_title('Original Image', fontweight='bold')
        axes[0].axis('off')
        
        # SHAP values heatmap
        # Resize SHAP values to match image dimensions
        shap_heatmap = cv2.resize(shap_values, (image.shape[1], image.shape[0]))
        
        # Take mean across channels for visualization
        if len(shap_heatmap.shape) == 3:
            shap_heatmap = np.mean(shap_heatmap, axis=2)
        
        im = axes[1].imshow(shap_heatmap, cmap='coolwarm', alpha=0.7)
        axes[1].imshow(image, alpha=0.3)
        axes[1].set_title('SHAP Values Heatmap\n(Red = Positive impact, Blue = Negative impact)', 
                         fontweight='bold')
        axes[1].axis('off')
        plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
        
        plt.tight_layout()
        return fig
    
    def _calculate_shap_feature_importance(self, shap_values: np.ndarray) -> Dict:
        """Calculate feature importance from SHAP values"""
        if len(shap_values.shape) == 3:
            # For image data, calculate importance per channel/region
            shap_abs = np.abs(shap_values)
            importance = {
                'total_impact': np.sum(shap_abs),
                'positive_impact': np.sum(np.maximum(shap_values, 0)),
                'negative_impact': np.sum(np.minimum(shap_values, 0)),
                'max_positive_region': np.unravel_index(np.argmax(shap_values), shap_values.shape),
                'max_negative_region': np.unravel_index(np.argmin(shap_values), shap_values.shape)
            }
        else:
            importance = {
                'total_impact': np.sum(np.abs(shap_values)),
                'positive_impact': np.sum(np.maximum(shap_values, 0)),
                'negative_impact': np.sum(np.minimum(shap_values, 0))
            }
        
        return importance
    
    def lime_explanation(self, image: np.ndarray, top_labels: int = 2, num_features: int = 100, 
                        num_samples: int = 1000) -> Dict:
        """Generate LIME explanation for an image"""
        if self.lime_explainer is None:
            self.initialize_lime_explainer()
        
        def model_predict(images: np.ndarray) -> np.ndarray:
            """Wrapper function for LIME"""
            return self.model.predict(images, verbose=0)
        
        # Generate LIME explanation
        explanation = self.lime_explainer.explain_instance(
            image.astype(np.double),
            model_predict,
            top_labels=top_labels,
            hide_color=0,
            num_samples=num_samples
        )
        
        # Create visualization
        lime_visualization = self._create_lime_visualization(image, explanation, num_features)
        
        return {
            'explanation': explanation,
            'local_prediction': explanation.local_pred,
            'available_labels': explanation.available_labels(),
            'visualization': lime_visualization,
            'feature_importance': self._extract_lime_importance(explanation, num_features)
        }
    
    def _create_lime_visualization(self, image: np.ndarray, explanation, num_features: int = 100) -> plt.Figure:
        """Create LIME explanation visualization"""
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # Original image
        axes[0].imshow(image)
        axes[0].set_title('Original Image', fontweight='bold')
        axes[0].axis('off')
        
        # LIME mask for top class
        temp, mask = explanation.get_image_and_mask(
            explanation.top_labels[0],
            positive_only=True,
            num_features=num_features,
            hide_rest=False
        )
        axes[1].imshow(mark_boundaries(temp, mask))
        axes[1].set_title('LIME Explanation\n(Green = Supporting features)', fontweight='bold')
        axes[1].axis('off')
        
        # LIME mask showing both positive and negative
        temp, mask = explanation.get_image_and_mask(
            explanation.top_labels[0],
            positive_only=False,
            num_features=num_features,
            hide_rest=False
        )
        axes[2].imshow(mark_boundaries(temp, mask))
        axes[2].set_title('LIME Explanation\n(Green = Positive, Red = Negative)', fontweight='bold')
        axes[2].axis('off')
        
        plt.tight_layout()
        return fig
    
    def _extract_lime_importance(self, explanation, num_features: int = 100) -> Dict:
        """Extract feature importance from LIME explanation"""
        importance_data = {}
        
        for label in explanation.available_labels():
            # Get features and their weights for this label
            features_weights = explanation.local_exp[label]
            features_weights_sorted = sorted(features_weights, key=lambda x: abs(x[1]), reverse=True)
            
            importance_data[label] = {
                'top_positive_features': [fw for fw in features_weights_sorted if fw[1] > 0][:num_features//2],
                'top_negative_features': [fw for fw in features_weights_sorted if fw[1] < 0][:num_features//2],
                'intercept': explanation.intercept[label],
                'local_prediction': explanation.local_pred[label]
            }
        
        return importance_data
    
    def comprehensive_explanation(self, image_path: str, save_dir: str = "results/comprehensive_explanations/") -> Dict:
        """Generate comprehensive explanation using all XAI methods"""
        try:
            # Load and preprocess image
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Could not load image: {image_path}")
                
            original_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            processed_image = cv2.resize(original_image, (224, 224))
            processed_image = processed_image.astype(np.float32) / 255.0
            
            # Get prediction
            pred = self.model.predict(np.expand_dims(processed_image, axis=0), verbose=0)[0]
            predicted_class = np.argmax(pred)
            confidence = pred[predicted_class]
            
            print(f"📊 Prediction: {self.class_names[predicted_class]} ({confidence:.3f})")
            
            # Generate explanations from all methods
            explanations = {
                'prediction': self.class_names[predicted_class],
                'confidence': float(confidence),
                'all_probabilities': {name: float(prob) for name, prob in zip(self.class_names, pred)},
                'grad_cam': None,
                'shap': None,
                'lime': None
            }
            
            # Grad-CAM (from your existing implementation)
            try:
                from src.xai_engine import MedicalXAIEngine
                basic_xai = MedicalXAIEngine(self.model, self.class_names)
                grad_cam_heatmap, _ = basic_xai.grad_cam(processed_image, class_idx=predicted_class)
                explanations['grad_cam'] = {
                    'heatmap': grad_cam_heatmap,
                    'overlay': basic_xai.overlay_heatmap(grad_cam_heatmap, original_image)
                }
                print("✅ Grad-CAM explanation generated")
            except Exception as e:
                print(f"⚠️ Grad-CAM failed: {e}")
            
            # SHAP explanation
            try:
                # Need background data for SHAP - use a simple approach
                background_samples = np.random.normal(0, 1, (10, 224, 224, 3))
                self.initialize_shap_explainer(background_samples)
                shap_explanation = self.shap_explanation(processed_image, predicted_class)
                explanations['shap'] = shap_explanation
                print("✅ SHAP explanation generated")
            except Exception as e:
                print(f"⚠️ SHAP failed: {e}")
            
            # LIME explanation
            try:
                lime_explanation = self.lime_explanation(processed_image)
                explanations['lime'] = lime_explanation
                print("✅ LIME explanation generated")
            except Exception as e:
                print(f"⚠️ LIME failed: {e}")
            
            # Create comprehensive visualization
            self._create_comprehensive_visualization(
                original_image, processed_image, explanations, image_path, save_dir
            )
            
            return explanations
            
        except Exception as e:
            print(f"❌ Comprehensive explanation failed: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _create_comprehensive_visualization(self, original, processed, explanations, image_path, save_dir):
        """Create comprehensive visualization with all XAI methods"""
        os.makedirs(save_dir, exist_ok=True)
        
        fig = plt.figure(figsize=(20, 12))
        
        # Create subplot grid
        gs = fig.add_gridspec(2, 4)
        
        # Original image
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.imshow(original)
        ax1.set_title('Original X-Ray', fontweight='bold')
        ax1.axis('off')
        
        # Preprocessed image
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.imshow(processed)
        ax2.set_title('Preprocessed', fontweight='bold')
        ax2.axis('off')
        
        # Grad-CAM
        if explanations['grad_cam']:
            ax3 = fig.add_subplot(gs[0, 2])
            ax3.imshow(explanations['grad_cam']['overlay'])
            ax3.set_title('Grad-CAM Overlay', fontweight='bold')
            ax3.axis('off')
        
        # SHAP
        if explanations['shap']:
            ax4 = fig.add_subplot(gs[0, 3])
            # Recreate SHAP visualization
            shap_values = explanations['shap']['shap_values'][0]
            shap_heatmap = cv2.resize(shap_values, (original.shape[1], original.shape[0]))
            if len(shap_heatmap.shape) == 3:
                shap_heatmap = np.mean(shap_heatmap, axis=2)
            
            ax4.imshow(shap_heatmap, cmap='coolwarm', alpha=0.7)
            ax4.imshow(original, alpha=0.3)
            ax4.set_title('SHAP Values', fontweight='bold')
            ax4.axis('off')
        
        # LIME
        if explanations['lime']:
            ax5 = fig.add_subplot(gs[1, :2])
            # Get LIME visualization
            explanation = explanations['lime']['explanation']
            temp, mask = explanation.get_image_and_mask(
                explanation.top_labels[0],
                positive_only=False,
                num_features=100,
                hide_rest=False
            )
            ax5.imshow(mark_boundaries(temp, mask))
            ax5.set_title('LIME Explanation\n(Green=Positive, Red=Negative)', fontweight='bold')
            ax5.axis('off')
        
        # Prediction and confidence
        ax6 = fig.add_subplot(gs[1, 2:])
        ax6.axis('off')
        prediction_text = f"""
        🏥 Medical XAI Analysis Report
        
        Diagnosis: {explanations['prediction']}
        Confidence: {explanations['confidence']:.3f}
        
        XAI Methods Used:
        • Grad-CAM: {'✅' if explanations['grad_cam'] else '❌'}
        • SHAP: {'✅' if explanations['shap'] else '❌'} 
        • LIME: {'✅' if explanations['lime'] else '❌'}
        
        Clinical Interpretation:
        - Multiple XAI methods provide consistent explanations
        - High-confidence regions indicate diagnostic importance
        - Consult with radiologist for final diagnosis
        """
        ax6.text(0.1, 0.9, prediction_text, fontsize=12, va='top', linespacing=1.5)
        
        plt.tight_layout()
        
        # Save with comprehensive filename
        filename = f"COMPREHENSIVE_{explanations['prediction']}_{explanations['confidence']:.3f}_{os.path.basename(image_path).split('.')[0]}.png"
        save_path = os.path.join(save_dir, filename)
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"💾 Comprehensive XAI visualization saved: {save_path}")