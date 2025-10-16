import tensorflow as tf
import numpy as np
import os
import glob
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def analyze_model_performance():
    print("📊 Analyzing Model Performance")
    print("=" * 40)
    
    # Load model
    model = tf.keras.models.load_model("models/best_model.h5")
    
    # Load some test data for quick analysis
    from src.data_preprocessing import MedicalDataPreprocessor
    preprocessor = MedicalDataPreprocessor((224, 224))
    
    # Sample images for quick test
    test_normal = glob.glob("data/raw/chest_xray/test/NORMAL/*.jpeg")[:50]
    test_pneumonia = glob.glob("data/raw/chest_xray/test/PNEUMONIA/*.jpeg")[:50]
    
    all_test_paths = test_normal + test_pneumonia
    all_test_labels = [0] * len(test_normal) + [1] * len(test_pneumonia)
    
    print(f"Testing on {len(all_test_paths)} images...")
    
    predictions = []
    confidences = []
    
    for i, (path, true_label) in enumerate(zip(all_test_paths, all_test_labels)):
        try:
            image = preprocessor.load_and_preprocess_image(path, augment=False)
            if image is not None:
                pred = model.predict(np.expand_dims(image, axis=0), verbose=0)[0]
                pred_class = np.argmax(pred)
                confidence = pred[pred_class]
                
                predictions.append(pred_class)
                confidences.append(confidence)
                
                if (i + 1) % 20 == 0:
                    print(f"Processed {i+1}/{len(all_test_paths)} images")
        except Exception as e:
            print(f"Error processing {path}: {e}")
            predictions.append(-1)  # Error marker
            confidences.append(0)
    
    # Filter out errors
    valid_indices = [i for i, pred in enumerate(predictions) if pred != -1]
    valid_predictions = [predictions[i] for i in valid_indices]
    valid_true = [all_test_labels[i] for i in valid_indices]
    valid_confidences = [confidences[i] for i in valid_indices]
    
    print(f"\n✅ Successfully processed {len(valid_predictions)} images")
    
    # Calculate accuracy
    accuracy = np.mean(np.array(valid_predictions) == np.array(valid_true))
    print(f"🎯 Sample Test Accuracy: {accuracy:.4f}")
    print(f"🎯 Average Confidence: {np.mean(valid_confidences):.4f}")
    
    # Classification report
    print("\n📋 Classification Report:")
    print(classification_report(valid_true, valid_predictions, 
                              target_names=['NORMAL', 'PNEUMONIA']))
    
    # Confusion matrix
    cm = confusion_matrix(valid_true, valid_predictions)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['NORMAL', 'PNEUMONIA'],
                yticklabels=['NORMAL', 'PNEUMONIA'])
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig('results/confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("💾 Confusion matrix saved to results/confusion_matrix.png")

if __name__ == "__main__":
    analyze_model_performance()