"""
Posture Classification Training Script (Local Version)
Supports: Arm Raise, Squats classification using EfficientNetB0
"""

import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import argparse
from datetime import datetime


class PostureTrainer:
    def __init__(self, config):
        self.data_dir = config['data_dir']
        self.img_size = config['img_size']
        self.batch_size = config['batch_size']
        self.epochs = config['epochs']
        self.seed = config['seed']
        self.model_save_dir = config['model_save_dir']
        
        # Create save directory if it doesn't exist
        os.makedirs(self.model_save_dir, exist_ok=True)
        
        # Data augmentation
        self.data_augmentation = keras.Sequential([
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.2),
            layers.RandomZoom(0.2),
            layers.RandomContrast(0.2),
        ], name="augmentation")
        
    def load_datasets(self):
        """Load training, validation, and test datasets"""
        print("Loading datasets...")
        
        train_dir = os.path.join(self.data_dir, "Train")
        test_dir = os.path.join(self.data_dir, "Test")
        
        # Check if directories exist
        if not os.path.exists(train_dir):
            raise FileNotFoundError(f"Training directory not found: {train_dir}")
        if not os.path.exists(test_dir):
            raise FileNotFoundError(f"Test directory not found: {test_dir}")
        
        # Load training dataset with validation split
        self.train_ds = keras.utils.image_dataset_from_directory(
            train_dir,
            image_size=self.img_size,
            batch_size=self.batch_size,
            validation_split=0.2,
            subset="training",
            seed=self.seed
        )
        
        self.val_ds = keras.utils.image_dataset_from_directory(
            train_dir,
            image_size=self.img_size,
            batch_size=self.batch_size,
            validation_split=0.2,
            subset="validation",
            seed=self.seed
        )
        
        self.test_ds = keras.utils.image_dataset_from_directory(
            test_dir,
            image_size=self.img_size,
            batch_size=self.batch_size,
            shuffle=False
        )
        
        self.class_names = self.train_ds.class_names
        print(f"Classes found: {self.class_names}")
        
    def compute_class_weights(self):
        """Compute class weights to handle imbalanced datasets"""
        print("Computing class weights...")
        y_train = np.concatenate([y for x, y in self.train_ds], axis=0)
        counts = np.bincount(y_train)
        total = np.sum(counts)
        self.class_weights = {i: total/(len(self.class_names)*c) for i, c in enumerate(counts)}
        print(f"Class weights: {self.class_weights}")
        
    def preprocess_datasets(self):
        """Apply preprocessing and augmentation to datasets"""
        print("Preprocessing datasets...")
        preprocess_input = keras.applications.efficientnet.preprocess_input
        
        def prep(ds, augment=False, shuffle=False):
            if augment:
                ds = ds.map(lambda x, y: (self.data_augmentation(x, training=True), y),
                           num_parallel_calls=tf.data.AUTOTUNE)
            ds = ds.map(lambda x, y: (preprocess_input(x), y), 
                       num_parallel_calls=tf.data.AUTOTUNE)
            if shuffle:
                ds = ds.shuffle(1000, seed=self.seed)
            return ds.prefetch(tf.data.AUTOTUNE)
        
        self.train_ds = prep(self.train_ds, augment=True, shuffle=True)
        self.val_ds = prep(self.val_ds)
        self.test_ds = prep(self.test_ds)
        
    def build_model(self):
        """Build EfficientNetB0-based model"""
        print("Building model...")
        
        # Create base model without loading weights first
        base_model = keras.applications.EfficientNetB0(
            weights=None,  # Don't load weights yet
            include_top=False,
            input_shape=(self.img_size[0], self.img_size[1], 3)
        )
        
        # Now load the weights
        base_model.load_weights(
            keras.utils.get_file(
                'efficientnetb0_notop.h5',
                'https://storage.googleapis.com/keras-applications/efficientnetb0_notop.h5',
                cache_subdir='models'
            ),
            skip_mismatch=True  # Skip mismatched layers
        )
        base_model.trainable = False  # Freeze initially
        
        # Build model
        inputs = keras.Input(shape=(self.img_size[0], self.img_size[1], 3))
        x = self.data_augmentation(inputs)
        x = keras.applications.efficientnet.preprocess_input(x)
        x = base_model(x, training=False)
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dropout(0.5)(x)
        x = layers.Dense(256, activation="relu", 
                        kernel_regularizer=keras.regularizers.l2(1e-4))(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.4)(x)
        outputs = layers.Dense(len(self.class_names), activation="softmax")(x)
        
        self.model = keras.Model(inputs, outputs)
        self.base_model = base_model
        
        print(f"Model built successfully!")
        print(f"Input shape: {self.model.input_shape}")
        print(f"Output shape: {self.model.output_shape}")
        
    def train_phase1(self):
        """Phase 1: Train with frozen base model"""
        print("\n" + "="*50)
        print("PHASE 1: Training with frozen backbone")
        print("="*50)
        
        self.model.compile(
            optimizer=keras.optimizers.Adam(1e-3),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"]
        )
        
        checkpoint_path = os.path.join(self.model_save_dir, "best_model_phase1.keras")
        callbacks = [
            keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
            keras.callbacks.ModelCheckpoint(checkpoint_path, save_best_only=True),
            keras.callbacks.ReduceLROnPlateau(factor=0.2, patience=5, min_lr=1e-6)
        ]
        
        history1 = self.model.fit(
            self.train_ds,
            validation_data=self.val_ds,
            epochs=15,
            class_weight=self.class_weights,
            callbacks=callbacks
        )
        
        return history1
        
    def train_phase2(self):
        """Phase 2: Fine-tune last layers of base model"""
        print("\n" + "="*50)
        print("PHASE 2: Fine-tuning backbone layers")
        print("="*50)
        
        # Unfreeze last 30 layers
        self.base_model.trainable = True
        for layer in self.base_model.layers[:-30]:
            layer.trainable = False
        
        self.model.compile(
            optimizer=keras.optimizers.Adam(1e-5),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"]
        )
        
        checkpoint_path = os.path.join(self.model_save_dir, "best_model_phase2.keras")
        callbacks = [
            keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
            keras.callbacks.ModelCheckpoint(checkpoint_path, save_best_only=True),
            keras.callbacks.ReduceLROnPlateau(factor=0.2, patience=5, min_lr=1e-6)
        ]
        
        history2 = self.model.fit(
            self.train_ds,
            validation_data=self.val_ds,
            epochs=self.epochs,
            class_weight=self.class_weights,
            callbacks=callbacks
        )
        
        return history2
        
    def evaluate(self):
        """Evaluate model on test set"""
        print("\n" + "="*50)
        print("EVALUATION")
        print("="*50)
        
        test_loss, test_acc = self.model.evaluate(self.test_ds)
        print(f"✅ Test Accuracy: {test_acc:.4f}")
        print(f"✅ Test Loss: {test_loss:.4f}")
        
        return test_loss, test_acc
        
    def save_model(self):
        """Save final model"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = os.path.join(self.model_save_dir, f"posture_model_final_{timestamp}.keras")
        self.model.save(save_path)
        print(f"\n✅ Final model saved at: {save_path}")
        
        # Also save class names
        class_names_path = os.path.join(self.model_save_dir, "class_names.txt")
        with open(class_names_path, 'w') as f:
            f.write('\n'.join(self.class_names))
        print(f"✅ Class names saved at: {class_names_path}")
        
        return save_path


def main():
    parser = argparse.ArgumentParser(description='Train posture classification model')
    parser.add_argument('--data_dir', type=str, default='./Datasets',
                       help='Path to dataset directory containing Train and Test folders')
    parser.add_argument('--model_save_dir', type=str, default='./models',
                       help='Directory to save trained models')
    parser.add_argument('--img_size', type=int, default=224,
                       help='Image size (default: 224)')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size (default: 32)')
    parser.add_argument('--epochs', type=int, default=40,
                       help='Number of epochs for phase 2 (default: 40)')
    parser.add_argument('--seed', type=int, default=123,
                       help='Random seed (default: 123)')
    
    args = parser.parse_args()
    
    config = {
        'data_dir': args.data_dir,
        'model_save_dir': args.model_save_dir,
        'img_size': (args.img_size, args.img_size),
        'batch_size': args.batch_size,
        'epochs': args.epochs,
        'seed': args.seed
    }
    
    print("="*50)
    print("POSTURE CLASSIFICATION TRAINING")
    print("="*50)
    print(f"TensorFlow version: {tf.__version__}")
    print(f"Configuration: {config}")
    
    # Initialize trainer
    trainer = PostureTrainer(config)
    
    # Load and prepare data
    trainer.load_datasets()
    trainer.compute_class_weights()
    trainer.preprocess_datasets()
    
    # Build model
    trainer.build_model()
    
    # Training
    history1 = trainer.train_phase1()
    history2 = trainer.train_phase2()
    
    # Evaluation
    trainer.evaluate()
    
    # Save
    model_path = trainer.save_model()
    
    print("\n" + "="*50)
    print("TRAINING COMPLETE!")
    print("="*50)
    print(f"Model saved at: {model_path}")


if __name__ == "__main__":
    main()

