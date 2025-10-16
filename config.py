import os

class Config:
    # Data Configuration - UPDATED FOR PROPER VALIDATION
    DATA_PATH = "data/"
    IMAGE_SIZE = (224, 224)
    BATCH_SIZE = 32  # Increased for better performance
    NUM_CLASSES = 2  # Changed to 2 classes (NORMAL vs PNEUMONIA)
    CLASS_NAMES = ['NORMAL', 'PNEUMONIA']  # Updated class names
    
    # Model Configuration
    MODEL_NAME = "DenseNet121"
    PRETRAINED = True
    LEARNING_RATE = 1e-4
    
    # Training Configuration
    EPOCHS = 30  # Reduced since we have proper validation
    PATIENCE = 8
    
    # XAI Configuration
    XAI_METHODS = ['grad_cam', 'score_cam']
    
    # Paths
    MODEL_SAVE_PATH = "models/"
    RESULTS_PATH = "results/"

config = Config()