import tensorflow as tf
import os
import sys
import glob
import random
from sklearn.model_selection import train_test_split
import numpy as np

# Add src to Python path
sys.path.append('src')

# Import your custom modules
try:
    from config import config
    from src.data_preprocessing import MedicalDataPreprocessor
    from src.model_builder import MedicalModelBuilder
    from src.xai_engine import MedicalXAIEngine
except ImportError as e:
    print(f"❌ Import Error: {e}")
    print("Please make sure all required files exist:")
    print("  - config.py")
    print("  - src/data_preprocessing.py") 
    print("  - src/model_builder.py")
    print("  - src/xai_engine.py")
    sys.exit(1)


def load_dataset_paths_corrected():
    """Load and properly split the dataset - CORRECTED VERSION"""
    print("📁 Loading dataset with proper validation split...")
    
    data_dir = "data/raw/chest_xray"
    
    # Check if dataset exists
    if not os.path.exists(data_dir):
        print(f"❌ Dataset not found at: {data_dir}")
        print("💡 Download from: https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia")
        print("💡 Extract to: data/raw/chest_xray/")
        return None
    
    # Load training data (we'll split this properly)
    train_normal = glob.glob(os.path.join(data_dir, "train", "NORMAL", "*.jpeg"))
    train_pneumonia = glob.glob(os.path.join(data_dir, "train", "PNEUMONIA", "*.jpeg"))
    
    # Load test data (keep separate for final evaluation)
    test_normal = glob.glob(os.path.join(data_dir, "test", "NORMAL", "*.jpeg"))
    test_pneumonia = glob.glob(os.path.join(data_dir, "test", "PNEUMONIA", "*.jpeg"))
    
    print("📊 Original dataset statistics:")
    print(f"   Train NORMAL: {len(train_normal)}")
    print(f"   Train PNEUMONIA: {len(train_pneumonia)}")
    print(f"   Test NORMAL: {len(test_normal)}")
    print(f"   Test PNEUMONIA: {len(test_pneumonia)}")
    
    # Check if we have any images
    if len(train_normal) == 0 or len(train_pneumonia) == 0:
        print("❌ No images found! Please check your dataset download.")
        return None
    
    # Combine all training images
    all_train_paths = train_normal + train_pneumonia
    all_train_labels = [0] * len(train_normal) + [1] * len(train_pneumonia)
    
    # Create proper train/validation split (80/20)
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        all_train_paths, 
        all_train_labels, 
        test_size=0.2, 
        random_state=42,
        stratify=all_train_labels  # Maintain class balance
    )
    
    # Prepare test set
    test_paths = test_normal + test_pneumonia
    test_labels = [0] * len(test_normal) + [1] * len(test_pneumonia)
    
    # Shuffle test set
    test_combined = list(zip(test_paths, test_labels))
    random.shuffle(test_combined)
    test_paths, test_labels = zip(*test_combined)
    
    print("\n✅ Proper dataset split:")
    print(f"   Training: {len(train_paths)} images")
    print(f"   Validation: {len(val_paths)} images")
    print(f"   Test: {len(test_paths)} images")
    print(f"   Class distribution in validation:")
    print(f"     - NORMAL: {sum(1 for label in val_labels if label == 0)}")
    print(f"     - PNEUMONIA: {sum(1 for label in val_labels if label == 1)}")
    
    return (list(train_paths), list(train_labels), 
            list(val_paths), list(val_labels), 
            list(test_paths), list(test_labels))


def main():
    print("🚀 Medical XAI Project - Corrected Validation Setup")
    print("=" * 60)
    
    # Create necessary directories
    os.makedirs(config.MODEL_SAVE_PATH, exist_ok=True)
    os.makedirs(config.RESULTS_PATH, exist_ok=True)
    
    # Load dataset with proper validation split
    dataset = load_dataset_paths_corrected()
    if dataset is None:
        return
    
    train_paths, train_labels, val_paths, val_labels, test_paths, test_labels = dataset
    
    # Initialize components
    preprocessor = MedicalDataPreprocessor(config.IMAGE_SIZE)
    
    print("\n🔄 Creating TensorFlow datasets...")
    
    # Test the preprocessor with a few samples first
    print("🧪 Testing data pipeline with sample images...")
    sample_image = preprocessor.load_and_preprocess_image(train_paths[0], augment=False)
    if sample_image is not None:
        print(f"✅ Sample image loaded: shape {sample_image.shape}")
    else:
        print("❌ Failed to load sample image!")
        return
    
    # Create datasets
    train_dataset = preprocessor.create_tf_dataset(
        train_paths, train_labels, 
        batch_size=config.BATCH_SIZE, 
        shuffle=True, augment=True
    )
    
    val_dataset = preprocessor.create_tf_dataset(
        val_paths, val_labels,
        batch_size=config.BATCH_SIZE,
        shuffle=False, augment=False
    )
    
    test_dataset = preprocessor.create_tf_dataset(
        test_paths, test_labels,
        batch_size=config.BATCH_SIZE,
        shuffle=False, augment=False
    )
    
    # Test one batch to ensure everything works
    print("🧪 Testing data batches...")
    try:
        for images, labels in train_dataset.take(1):
            print(f"✅ Batch shape: {images.shape}")
            print(f"✅ Labels shape: {labels.shape}")
            print(f"✅ Data types - Images: {images.dtype}, Labels: {labels.dtype}")
            break
    except Exception as e:
        print(f"❌ Error creating datasets: {e}")
        return
    
    # Build and train model
    print("\n🧠 Building model...")
    model_builder = MedicalModelBuilder(config)
    model = model_builder.build_model()
    model = model_builder.compile_model(model)
    
    print("📋 Model Summary:")
    model.summary()
    
    # Callbacks
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            patience=config.PATIENCE,
            restore_best_weights=True,
            monitor='val_accuracy',
            verbose=1
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(config.MODEL_SAVE_PATH, "best_model.h5"),
            save_best_only=True,
            monitor='val_accuracy',
            verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            factor=0.2,
            patience=5,
            monitor='val_loss',
            verbose=1
        ),
        tf.keras.callbacks.CSVLogger(
            os.path.join(config.RESULTS_PATH, 'training_log.csv')
        )
    ]
    
    print("\n🎯 Starting training with proper validation...")
    print(f"📈 Training for {config.EPOCHS} epochs")
    print(f"📦 Batch size: {config.BATCH_SIZE}")
    print(f"🎯 Learning rate: {config.LEARNING_RATE}")
    
    # Train the model
    try:
        history = model.fit(
            train_dataset,
            epochs=config.EPOCHS,
            validation_data=val_dataset,
            callbacks=callbacks,
            verbose=1
        )
        
           print("✅ Training completed!")
    
    # Final evaluation - FIXED VERSION
    print("\n🧪 Final evaluation on test set...")
    try:
        test_results = model.evaluate(test_dataset, verbose=1, return_dict=True)
        print("\n📊 Final Test Results:")
        for metric, value in test_results.items():
            print(f"   {metric}: {value:.4f}")
        
        test_accuracy = test_results.get('accuracy', test_results.get('sparse_categorical_accuracy', 0))
        test_loss = test_results['loss']
        
        print(f"🎯 Test Accuracy: {test_accuracy:.4f}")
        print(f"🎯 Test Loss: {test_loss:.4f}")
        
    except Exception as e:
        print(f"⚠️ Evaluation warning: {e}")
        print("💡 But model training was successful!")
        test_accuracy = 0.0
        test_loss = 0.0
    
    # Save model
    model.save(os.path.join(config.MODEL_SAVE_PATH, "final_model.h5"))
    print("💾 Model saved!")
    
    # Create XAI engine for explanations
    print("\n🔍 Setting up XAI engine...")
    xai_engine = MedicalXAIEngine(model, config.CLASS_NAMES)
    
    # Test XAI on a few sample images
    print("🧪 Testing XAI on sample images...")
    sample_test_images = test_paths[:3]  # First 3 test images
    for i, img_path in enumerate(sample_test_images):
        explanation = xai_engine.generate_explanation_report(img_path)
        if explanation:
            print(f"✅ XAI explanation {i+1} generated for: {os.path.basename(img_path)}")
        else:
            print(f"⚠️ Could not generate XAI for: {os.path.basename(img_path)}")
    
    return model, history