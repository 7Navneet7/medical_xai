import tensorflow as tf
import cv2
import numpy as np

class MedicalDataPreprocessor:
    def __init__(self, image_size=(224, 224)):
        self.image_size = image_size
    
    def load_and_preprocess_image(self, image_path, augment=False):
        """Load and preprocess medical image"""
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, self.image_size)
        image = image.astype(np.float32) / 255.0
        
        return image
    
    def create_tf_dataset(self, image_paths, labels, batch_size=16, shuffle=False, augment=False):
        """Create TensorFlow dataset from paths and labels"""
        def generator():
            for img_path, label in zip(image_paths, labels):
                try:
                    image = self.load_and_preprocess_image(img_path, augment)
                    yield image, label
                except Exception as e:
                    print(f"Warning: Could not load {img_path}: {e}")
                    continue
        
        dataset = tf.data.Dataset.from_generator(
            generator,
            output_signature=(
                tf.TensorSpec(shape=(*self.image_size, 3), dtype=tf.float32),
                tf.TensorSpec(shape=(), dtype=tf.int32)
            )
        )
        
        if shuffle:
            dataset = dataset.shuffle(buffer_size=len(image_paths))
        
        dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
        return dataset