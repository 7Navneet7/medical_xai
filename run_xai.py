import tensorflow as tf
import os
import sys
import glob

# Add src to path
sys.path.append('src')

from config import config
from src.xai_engine import MedicalXAIEngine

def main():
    print("🎯 Running XAI on Trained Model")
    print("=" * 50)
    
    # Check if model exists
    model_path = "models/best_model.h5"
    if not os.path.exists(model_path):
        print("❌ Model not found. Please train the model first.")
        return
    
    # Load the trained model
    print("📦 Loading trained model...")
    model = tf.keras.models.load_model(model_path)
    print("✅ Model loaded successfully!")
    
    # Initialize XAI engine
    xai_engine = MedicalXAIEngine(model, config.CLASS_NAMES)
    
    # Find test images from both classes
    print("\n🔍 Finding test images...")
    normal_images = glob.glob("data/raw/chest_xray/test/NORMAL/*.jpeg")[:2]
    pneumonia_images = glob.glob("data/raw/chest_xray/test/PNEUMONIA/*.jpeg")[:2]
    test_images = normal_images + pneumonia_images
    
    print(f"📁 Found {len(test_images)} test images:")
    for img in test_images:
        print(f"   - {os.path.basename(img)}")
    
    # Generate XAI explanations
    print("\n🎨 Generating XAI explanations...")
    explanations = []
    
    for i, img_path in enumerate(test_images):
        print(f"\n🔬 {i+1}/{len(test_images)}: Analyzing {os.path.basename(img_path)}")
        explanation = xai_engine.generate_explanation_report(img_path)
        if explanation:
            explanations.append(explanation)
            print(f"   ✅ Prediction: {explanation['prediction']}")
            print(f"   ✅ Confidence: {explanation['confidence']:.3f}")
        else:
            print(f"   ❌ Failed to generate explanation")
    
    print(f"\n🎉 XAI completed!")
    print(f"📊 Generated {len(explanations)} explanations")
    print(f"📁 Check 'results/explanations/' folder for visualizations")
    
    # Show summary
    if explanations:
        print("\n📈 Summary:")
        normal_count = sum(1 for exp in explanations if exp['prediction'] == 'NORMAL')
        pneumonia_count = sum(1 for exp in explanations if exp['prediction'] == 'PNEUMONIA')
        print(f"   NORMAL predictions: {normal_count}")
        print(f"   PNEUMONIA predictions: {pneumonia_count}")
        
        avg_confidence = sum(exp['confidence'] for exp in explanations) / len(explanations)
        print(f"   Average confidence: {avg_confidence:.3f}")

if __name__ == "__main__":
    main()