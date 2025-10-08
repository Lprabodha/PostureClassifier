"""
Posture Classification Training Script
Enhanced with advanced techniques for higher accuracy
"""

import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
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
        
        os.makedirs(self.model_save_dir, exist_ok=True)
        
        # Enhanced data augmentation
        self.data_augmentation = keras.Sequential([
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.3),
            layers.RandomZoom(0.3),
            layers.RandomContrast(0.3),
            layers.RandomBrightness(0.2),
            layers.RandomTranslation(0.15, 0.15),
        ], name="augmentation")
        
    def load_datasets(self):
        """Load datasets with proper image loading - only Arm_Raise and Squats"""
        print("Loading datasets...")
        
        train_dir = os.path.join(self.data_dir, "Train")
        test_dir = os.path.join(self.data_dir, "Test")
        
        if not os.path.exists(train_dir):
            raise FileNotFoundError(f"Training directory not found: {train_dir}")
        if not os.path.exists(test_dir):
            raise FileNotFoundError(f"Test directory not found: {test_dir}")
        
        # Load all datasets first
        train_ds_full = keras.utils.image_dataset_from_directory(
            train_dir,
            image_size=self.img_size,
            batch_size=self.batch_size,
            validation_split=0.2,
            subset="training",
            seed=self.seed,
            color_mode='rgb',
            interpolation='bilinear'
        )
        
        val_ds_full = keras.utils.image_dataset_from_directory(
            train_dir,
            image_size=self.img_size,
            batch_size=self.batch_size,
            validation_split=0.2,
            subset="validation",
            seed=self.seed,
            color_mode='rgb',
            interpolation='bilinear'
        )
        
        test_ds_full = keras.utils.image_dataset_from_directory(
            test_dir,
            image_size=self.img_size,
            batch_size=self.batch_size,
            shuffle=False,
            color_mode='rgb',
            interpolation='bilinear'
        )
        
        all_class_names = train_ds_full.class_names
        print(f"All classes in dataset: {all_class_names}")
        
        # Filter to only Arm_Raise and Squats
        allowed_classes = ['Arm_Raise', 'Squats']
        keep_indices = [i for i, name in enumerate(all_class_names) if name in allowed_classes]
        
        # Create label remapping tensor
        label_map_list = [-1] * len(all_class_names)  # -1 for classes to filter out
        for new_idx, old_idx in enumerate(keep_indices):
            label_map_list[old_idx] = new_idx
        label_map_tensor = tf.constant(label_map_list, dtype=tf.int32)
        
        def remap_label(image, label):
            # Get new label from mapping
            new_label = tf.gather(label_map_tensor, label)
            return image, new_label
        
        # Apply remapping, unbatch, filter out -1 labels, then rebatch
        self.train_ds = train_ds_full.unbatch().map(remap_label).filter(
            lambda x, y: tf.not_equal(y, -1)
        ).batch(self.batch_size)
        
        self.val_ds = val_ds_full.unbatch().map(remap_label).filter(
            lambda x, y: tf.not_equal(y, -1)
        ).batch(self.batch_size)
        
        self.test_ds = test_ds_full.unbatch().map(remap_label).filter(
            lambda x, y: tf.not_equal(y, -1)
        ).batch(self.batch_size)
        
        self.class_names = [all_class_names[i] for i in keep_indices]
        print(f"Using classes: {self.class_names}")
        
        # Print dataset statistics
        print(f"\nDataset Statistics:")
        print(f"  Training batches: {tf.data.experimental.cardinality(self.train_ds).numpy()}")
        print(f"  Validation batches: {tf.data.experimental.cardinality(self.val_ds).numpy()}")
        print(f"  Test batches: {tf.data.experimental.cardinality(self.test_ds).numpy()}")
        print(f"  Note: Filtered to only use Arm_Raise and Squats poses")
        
    def compute_class_weights(self):
        """Compute class weights for imbalanced data"""
        print("\nComputing class weights...")
        y_train = np.concatenate([y for x, y in self.train_ds], axis=0)
        counts = np.bincount(y_train)
        total = np.sum(counts)
        
        # Print class distribution
        print("\nClass Distribution:")
        for i, (name, count) in enumerate(zip(self.class_names, counts)):
            percentage = (count / total) * 100
            print(f"  {name}: {count} images ({percentage:.1f}%)")
        
        self.class_weights = {i: total/(len(self.class_names)*c) for i, c in enumerate(counts)}
        print(f"\nClass weights: {self.class_weights}")
        
    def preprocess_datasets(self):
        """Enhanced preprocessing pipeline"""
        print("\nPreprocessing datasets...")
        preprocess_input = keras.applications.efficientnet.preprocess_input
        
        def prep(ds, augment=False, shuffle=False, cache=False):
            if cache:
                ds = ds.cache()
            if shuffle:
                ds = ds.shuffle(1000, seed=self.seed)
            if augment:
                ds = ds.map(lambda x, y: (self.data_augmentation(x, training=True), y),
                           num_parallel_calls=tf.data.AUTOTUNE)
            ds = ds.map(lambda x, y: (preprocess_input(x), y), 
                       num_parallel_calls=tf.data.AUTOTUNE)
            return ds.prefetch(tf.data.AUTOTUNE)
        
        self.train_ds = prep(self.train_ds, augment=True, shuffle=True, cache=False)
        self.val_ds = prep(self.val_ds, cache=True)
        self.test_ds = prep(self.test_ds, cache=True)
        
    def build_model(self):
        """Build enhanced model with better architecture"""
        print("\nBuilding model...")
        
        # Use EfficientNetB0 backbone
        base_model = keras.applications.EfficientNetB0(
            weights=None,
            include_top=False,
            input_shape=(self.img_size[0], self.img_size[1], 3)
        )
        
        # Load weights with skip_mismatch
        try:
            base_model.load_weights(
                keras.utils.get_file(
                    'efficientnetb0_notop.h5',
                    'https://storage.googleapis.com/keras-applications/efficientnetb0_notop.h5',
                    cache_subdir='models'
                ),
                skip_mismatch=True
            )
            print("Pretrained weights loaded successfully")
        except Exception as e:
            print(f"Warning: Could not load pretrained weights: {e}")
        
        base_model.trainable = False
        
        # Enhanced head architecture
        inputs = keras.Input(shape=(self.img_size[0], self.img_size[1], 3))
        
        # Apply augmentation in the model
        x = self.data_augmentation(inputs)
        x = keras.applications.efficientnet.preprocess_input(x)
        
        # Feature extraction
        x = base_model(x, training=False)
        x = layers.GlobalAveragePooling2D()(x)
        
        # Deeper classification head with regularization
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.5)(x)
        x = layers.Dense(512, activation="relu", 
                        kernel_regularizer=keras.regularizers.l2(1e-4))(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.4)(x)
        x = layers.Dense(256, activation="relu",
                        kernel_regularizer=keras.regularizers.l2(1e-4))(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.3)(x)
        
        # Output layer
        outputs = layers.Dense(len(self.class_names), activation="softmax")(x)
        
        self.model = keras.Model(inputs, outputs)
        self.base_model = base_model
        
        print("Model built successfully")
        print(f"   Input shape: {self.model.input_shape}")
        print(f"   Output shape: {self.model.output_shape}")
        print(f"   Total parameters: {self.model.count_params():,}")
        
    def train_phase1(self):
        """Phase 1: Train with frozen backbone"""
        print("\n" + "="*60)
        print("PHASE 1: Training with frozen backbone")
        print("="*60)
        
        # Use fixed learning rate with ReduceLROnPlateau
        initial_lr = 1e-3
        
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=initial_lr),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"]
        )
        
        checkpoint_path = os.path.join(self.model_save_dir, "best_model_phase1.keras")
        callbacks = [
            keras.callbacks.EarlyStopping(
                patience=12, 
                restore_best_weights=True,
                monitor='val_accuracy'
            ),
            keras.callbacks.ModelCheckpoint(
                checkpoint_path, 
                save_best_only=True,
                monitor='val_accuracy',
                mode='max'
            ),
            keras.callbacks.ReduceLROnPlateau(
                factor=0.5, 
                patience=5, 
                min_lr=1e-7,
                monitor='val_loss',
                verbose=1
            )
        ]
        
        history = self.model.fit(
            self.train_ds,
            validation_data=self.val_ds,
            epochs=20,
            class_weight=self.class_weights,
            callbacks=callbacks,
            verbose=1
        )
        
        return history
        
    def train_phase2(self):
        """Phase 2: Fine-tune with unfrozen layers"""
        print("\n" + "="*60)
        print("PHASE 2: Fine-tuning backbone layers")
        print("="*60)
        
        # Unfreeze more layers for better adaptation
        self.base_model.trainable = True
        for layer in self.base_model.layers[:-50]:
            layer.trainable = False
        
        print(f"Trainable layers: {sum([1 for layer in self.model.layers if layer.trainable])}")
        
        # Lower learning rate for fine-tuning
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=5e-6),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"]
        )
        
        checkpoint_path = os.path.join(self.model_save_dir, "best_model_phase2.keras")
        callbacks = [
            keras.callbacks.EarlyStopping(
                patience=15, 
                restore_best_weights=True,
                monitor='val_accuracy'
            ),
            keras.callbacks.ModelCheckpoint(
                checkpoint_path, 
                save_best_only=True,
                monitor='val_accuracy',
                mode='max'
            ),
            keras.callbacks.ReduceLROnPlateau(
                factor=0.5, 
                patience=6, 
                min_lr=1e-8,
                monitor='val_loss',
                verbose=1
            )
        ]
        
        history = self.model.fit(
            self.train_ds,
            validation_data=self.val_ds,
            epochs=self.epochs,
            class_weight=self.class_weights,
            callbacks=callbacks,
            verbose=1
        )
        
        return history
        
    def evaluate(self):
        """Detailed evaluation on test set"""
        print("\n" + "="*60)
        print("FINAL EVALUATION")
        print("="*60)
        
        # Overall metrics
        results = self.model.evaluate(self.test_ds, verbose=1)
        print(f"\nTest Loss: {results[0]:.4f}")
        print(f"Test Accuracy: {results[1]:.4f}")
        
        # Per-class accuracy
        print("\nPer-Class Performance:")
        y_true = []
        y_pred = []
        
        for images, labels in self.test_ds:
            predictions = self.model.predict(images, verbose=0)
            y_true.extend(labels.numpy())
            y_pred.extend(np.argmax(predictions, axis=1))
        
        from sklearn.metrics import classification_report, confusion_matrix
        
        print("\nClassification Report:")
        print(classification_report(y_true, y_pred, target_names=self.class_names))
        
        print("\nConfusion Matrix:")
        cm = confusion_matrix(y_true, y_pred)
        print(cm)
        
        # Per-class accuracy
        for i, class_name in enumerate(self.class_names):
            class_acc = cm[i, i] / cm[i].sum() if cm[i].sum() > 0 else 0
            print(f"  {class_name}: {class_acc*100:.2f}%")
        
        return results
        
    def save_model(self):
        """Save final model with metadata"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = os.path.join(self.model_save_dir, f"posture_model_{timestamp}.keras")
        self.model.save(save_path)
        print(f"\nFinal model saved at: {save_path}")
        
        # Save class names
        class_names_path = os.path.join(self.model_save_dir, "class_names.txt")
        with open(class_names_path, 'w') as f:
            f.write('\n'.join(self.class_names))
        print(f"Class names saved at: {class_names_path}")
        
        # Save model info
        info_path = os.path.join(self.model_save_dir, f"model_info_{timestamp}.txt")
        with open(info_path, 'w') as f:
            f.write(f"Model: EfficientNetB0 + Enhanced Head\n")
            f.write(f"Classes: {', '.join(self.class_names)}\n")
            f.write(f"Image Size: {self.img_size}\n")
            f.write(f"Total Parameters: {self.model.count_params():,}\n")
            f.write(f"Timestamp: {timestamp}\n")
        
        return save_path


def main():
    parser = argparse.ArgumentParser(description='Train posture classification model')
    parser.add_argument('--data_dir', type=str, default='./Datasets')
    parser.add_argument('--model_save_dir', type=str, default='./models')
    parser.add_argument('--img_size', type=int, default=224)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--seed', type=int, default=123)
    
    args = parser.parse_args()
    
    config = {
        'data_dir': args.data_dir,
        'model_save_dir': args.model_save_dir,
        'img_size': (args.img_size, args.img_size),
        'batch_size': args.batch_size,
        'epochs': args.epochs,
        'seed': args.seed
    }
    
    print("="*60)
    print("POSTURE CLASSIFICATION TRAINING")
    print("="*60)
    print(f"TensorFlow version: {tf.__version__}")
    print(f"Configuration: {config}\n")
    
    # Initialize trainer
    trainer = PostureTrainer(config)
    
    # Training pipeline
    trainer.load_datasets()
    trainer.compute_class_weights()
    trainer.preprocess_datasets()
    trainer.build_model()
    
    # Two-phase training
    history1 = trainer.train_phase1()
    history2 = trainer.train_phase2()
    
    # Evaluation
    trainer.evaluate()
    
    # Save
    model_path = trainer.save_model()
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print("="*60)
    print(f"Model saved at: {model_path}")
    print("\nTo use the model:")
    print("  python app.py")
    print("  Then visit: http://localhost:5000")


if __name__ == "__main__":
    main()
