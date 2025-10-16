import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
from typing import Tuple, Optional

class MedicalXAIEngine:
    def __init__(self, model, class_names):
        self.model = model
        self.class_names = class_names
    
    def grad_cam(self, image: np.ndarray, layer_name: str = None, class_idx: int = None) -> Tuple[np.ndarray, int]:
        """Generate Grad-CAM heatmap for medical image explanations"""
        if layer_name is None:
            # Try to find a convolutional layer
            for layer in self.model.layers[::-1]:
                if 'conv' in layer.name:
                    layer_name = layer.name
                    break
            if layer_name is None:
                layer_name = self.model.layers[-2].name  # Fallback
        
        # Create gradient model
        grad_model = tf.keras.models.Model(
            inputs=[self.model.inputs],
            outputs=[self.model.get_layer(layer_name).output, self.model.output]
        )
        
        # Compute gradients
        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(np.expand_dims(image, axis=0))
            if class_idx is None:
                class_idx = tf.argmax(predictions[0])
            loss = predictions[:, class_idx]
        
        grads = tape.gradient(loss, conv_outputs)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        
        # Weight the feature maps
        conv_outputs = conv_outputs[0]
        heatmap = tf.reduce_sum(tf.multiply(pooled_grads, conv_outputs), axis=-1)
        
        # Apply ReLU and normalize
        heatmap = np.maximum(heatmap.numpy(), 0)  # Convert tensor to numpy
        heatmap = (heatmap - np.min(heatmap)) / (np.max(heatmap) - np.min(heatmap) + 1e-8)
        
        # Convert class_idx to int if it's a tensor
        if hasattr(class_idx, 'numpy'):
            class_idx = class_idx.numpy()
        
        return heatmap, int(class_idx)
    
    def overlay_heatmap(self, heatmap: np.ndarray, original_image: np.ndarray, 
                       alpha: float = 0.4, colormap: int = cv2.COLORMAP_JET) -> np.ndarray:
        """Overlay heatmap on original medical image"""
        # Resize heatmap to match original image dimensions
        heatmap_resized = cv2.resize(heatmap, (original_image.shape[1], original_image.shape[0]))
        
        # Apply colormap
        heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), colormap)
        heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
        
        # Ensure both images have same data type
        if original_image.dtype != np.uint8:
            original_image_uint8 = (original_image * 255).astype(np.uint8)
        else:
            original_image_uint8 = original_image
        
        # Blend images
        overlayed = cv2.addWeighted(original_image_uint8, 1 - alpha, heatmap_colored, alpha, 0)
        
        return overlayed
    
    def generate_explanation_report(self, image_path: str, save_dir: str = "results/explanations/"):
        """Generate comprehensive XAI explanation report"""
        try:
            # Load and preprocess image
            image = cv2.imread(image_path)
            if image is None:
                print(f"❌ Could not load image: {image_path}")
                return None
                
            original_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            processed_image = cv2.resize(original_image, (224, 224))
            processed_image = processed_image.astype(np.float32) / 255.0
            
            # Get prediction
            pred = self.model.predict(np.expand_dims(processed_image, axis=0), verbose=0)[0]
            predicted_class = np.argmax(pred)
            confidence = pred[predicted_class]
            
            print(f"📊 Prediction: {self.class_names[predicted_class]} ({confidence:.3f})")
            
            # Generate Grad-CAM heatmap
            heatmap, _ = self.grad_cam(processed_image, class_idx=predicted_class)
            
            # Create overlay
            overlay = self.overlay_heatmap(heatmap, original_image)
            
            # Create comprehensive visualization
            self._create_explanation_visualization(
                original_image, processed_image, heatmap, overlay,
                self.class_names[predicted_class], confidence, 
                image_path, save_dir
            )
            
            return {
                'prediction': self.class_names[predicted_class],
                'confidence': float(confidence),
                'all_probabilities': {name: float(prob) for name, prob in zip(self.class_names, pred)},
                'heatmap': heatmap
            }
            
        except Exception as e:
            print(f"❌ Error generating explanation: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _create_explanation_visualization(self, original, processed, heatmap, overlay, 
                                        prediction, confidence, image_path, save_dir):
        """Create comprehensive explanation visualization"""
        os.makedirs(save_dir, exist_ok=True)
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Original image
        axes[0,0].imshow(original)
        axes[0,0].set_title('Original Chest X-Ray', fontsize=14, fontweight='bold')
        axes[0,0].axis('off')
        
        # Preprocessed image
        axes[0,1].imshow(processed)
        axes[0,1].set_title('Preprocessed Image', fontsize=14, fontweight='bold')
        axes[0,1].axis('off')
        
        # Heatmap
        im = axes[1,0].imshow(heatmap, cmap='jet')
        axes[1,0].set_title('Grad-CAM Heatmap', fontsize=14, fontweight='bold')
        axes[1,0].axis('off')
        plt.colorbar(im, ax=axes[1,0], fraction=0.046, pad=0.04)
        
        # Overlay
        axes[1,1].imshow(overlay)
        axes[1,1].set_title(f'Explanation Overlay\nPrediction: {prediction}\nConfidence: {confidence:.3f}', 
                           fontsize=14, fontweight='bold')
        axes[1,1].axis('off')
        
        plt.tight_layout()
        
        # Save with descriptive filename
        filename = f"XAI_{prediction}_{confidence:.3f}_{os.path.basename(image_path).split('.')[0]}.png"
        save_path = os.path.join(save_dir, filename)
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"💾 XAI visualization saved: {save_path}")
    
    def analyze_multiple_images(self, image_paths: list, max_images: int = 10):
        """Generate XAI explanations for multiple images"""
        print(f"🔍 Generating XAI explanations for {min(len(image_paths), max_images)} images...")
        
        explanations = []
        for i, img_path in enumerate(image_paths[:max_images]):
            print(f"  {i+1}/{min(len(image_paths), max_images)}: {os.path.basename(img_path)}")
            explanation = self.generate_explanation_report(img_path)
            if explanation:
                explanations.append(explanation)
        
        print(f"✅ Generated {len(explanations)} XAI explanations")
        return explanations