import tensorflow as tf
import numpy as np
import os
import glob
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# --- New function to evaluate performance with a custom threshold ---
def evaluate_with_threshold(true_labels, probabilities, threshold, target_names):
    """
    Recalculates predictions, metrics, and generates a confusion matrix 
    using a specified decision threshold for the positive class (PNEUMONIA, index 1).
    """
    # 1. Apply the new threshold
    # If the PNEUMONIA probability (prob) is >= threshold, the prediction is 1 (PNEUMONIA)
    # Otherwise, the prediction is 0 (NORMAL)
    threshold_predictions = (probabilities >= threshold).astype(int)

    # 2. Calculate accuracy
    accuracy = np.mean(threshold_predictions == np.array(true_labels))
    
    # 3. Classification report
    report = classification_report(true_labels, threshold_predictions, 
                                   target_names=target_names, output_dict=True)
    
    # 4. Confusion matrix
    cm = confusion_matrix(true_labels, threshold_predictions)
    
    return accuracy, report, cm

def analyze_model_performance():
    print("📊 Analyzing Model Performance")
    print("=" * 40)
    
    # --- Configuration ---
    # The current threshold is 0.5. We are testing a higher threshold to reduce False Positives.
    TARGET_THRESHOLD = 0.75
    TARGET_NAMES = ['NORMAL', 'PNEUMONIA']
    
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
    
    # We will store the raw PNEUMONIA probability (index 1) for threshold testing
    pneumonia_probabilities = []
    
    for i, path in enumerate(all_test_paths):
        try:
            image = preprocessor.load_and_preprocess_image(path, augment=False)
            if image is not None:
                # pred is a 2-element array: [P(NORMAL), P(PNEUMONIA)]
                pred = model.predict(np.expand_dims(image, axis=0), verbose=0)[0]
                
                # Store P(PNEUMONIA)
                pneumonia_probabilities.append(pred[1])
                
                if (i + 1) % 20 == 0:
                    print(f"Processed {i+1}/{len(all_test_paths)} images")
        except Exception as e:
            print(f"Error processing {path}: {e}")
            pneumonia_probabilities.append(np.nan) # Marker for error
    
    # Filter out errors
    valid_indices = [i for i, prob in enumerate(pneumonia_probabilities) if not np.isnan(prob)]
    valid_probabilities = np.array([pneumonia_probabilities[i] for i in valid_indices])
    valid_true = [all_test_labels[i] for i in valid_indices]
    
    print(f"\n✅ Successfully processed {len(valid_probabilities)} images")
    
    # --- 1. Evaluate Performance at Default Threshold (0.5) ---
    accuracy_05, report_05, cm_05 = evaluate_with_threshold(
        valid_true, valid_probabilities, 0.5, TARGET_NAMES
    )
    
    print("\n--- BASELINE PERFORMANCE (THRESHOLD 0.5) ---")
    print(f"🎯 Sample Test Accuracy: {accuracy_05:.4f}")
    print("\n📋 Classification Report:")
    print(classification_report(valid_true, (valid_probabilities >= 0.5).astype(int), 
                              target_names=TARGET_NAMES))
    
    # --- 2. Evaluate Performance at Target Threshold (e.g., 0.75) ---
    accuracy_target, report_target, cm_target = evaluate_with_threshold(
        valid_true, valid_probabilities, TARGET_THRESHOLD, TARGET_NAMES
    )
    
    print(f"\n--- RECALIBRATED PERFORMANCE (THRESHOLD {TARGET_THRESHOLD:.2f}) ---")
    print(f"🎯 Sample Test Accuracy: {accuracy_target:.4f}")
    print(f"NORMAL Recall (Specificity): {report_target['NORMAL']['recall']:.4f}")
    print(f"PNEUMONIA Recall (Sensitivity): {report_target['PNEUMONIA']['recall']:.4f}")
    
    print("\n📋 Classification Report:")
    print(classification_report(valid_true, (valid_probabilities >= TARGET_THRESHOLD).astype(int), 
                              target_names=TARGET_NAMES))
    
    # --- 3. Plot Recalibrated Confusion Matrix ---
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_target, annot=True, fmt='d', cmap='Blues',
                xticklabels=TARGET_NAMES,
                yticklabels=TARGET_NAMES)
    plt.title(f'Confusion Matrix (Threshold: {TARGET_THRESHOLD:.2f})')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    filename = f'results/confusion_matrix_t{int(TARGET_THRESHOLD*100)}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"💾 Recalibrated confusion matrix saved to {filename}")

if __name__ == "__main__":
    analyze_model_performance()
