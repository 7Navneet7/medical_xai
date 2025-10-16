import tensorflow as tf
from tensorflow.keras import layers, Model
import tensorflow.keras.applications as apps

class MedicalModelBuilder:
    def __init__(self, config):
        self.config = config
    
    def build_model(self) -> Model:
        """Build medical image classification model optimized for M4"""
        
        # Base model - choose based on performance
        if self.config.MODEL_NAME == "DenseNet121":
            base_model = apps.DenseNet121(
                weights='imagenet' if self.config.PRETRAINED else None,
                include_top=False,
                input_shape=(*self.config.IMAGE_SIZE, 3)
            )
        elif self.config.MODEL_NAME == "EfficientNetB0":
            base_model = apps.EfficientNetB0(
                weights='imagenet' if self.config.PRETRAINED else None,
                include_top=False,
                input_shape=(*self.config.IMAGE_SIZE, 3)
            )
        else:
            raise ValueError(f"Unsupported model: {self.config.MODEL_NAME}")
        
        # Freeze early layers
        base_model.trainable = True
        for layer in base_model.layers[:-30]:  # Fine-tune last 30 layers
            layer.trainable = False
        
        # Add custom head
        x = base_model.output
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dense(512, activation='relu')(x)
        x = layers.Dropout(0.3)(x)
        x = layers.Dense(256, activation='relu')(x)
        x = layers.Dropout(0.3)(x)
        outputs = layers.Dense(self.config.NUM_CLASSES, activation='softmax')(x)
        
        model = Model(inputs=base_model.input, outputs=outputs)
        
        # Store reference to last conv layer for XAI
        model.last_conv_layer = base_model.layers[-1].name
        
        return model
    
    def compile_model(self, model: Model) -> Model:
        """Compile model with appropriate settings"""
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.config.LEARNING_RATE),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy', 'sparse_categorical_accuracy']
        )
        return model