import os
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.python.client import device_lib
import matplotlib.pyplot as plt
import kagglehub
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
print("TensorFlow version:", tf.__version__)
print("GPUs detected:", tf.config.list_physical_devices('GPU'))
print(device_lib.list_local_devices())

MODELS_DIR = "models"
if not os.path.exists(MODELS_DIR):
    os.makedirs(MODELS_DIR)

CAR_MODEL_PATH = os.path.join(MODELS_DIR, "vehicle_classifier_model.keras")

# Data augmentation pipeline
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.05),
    layers.RandomZoom(0.05),
], name="data_augmentation")

def download_data():
    """
    Downloads the vehicle images dataset from Kaggle and returns the local path.
    """
    path = kagglehub.dataset_download("lyensoetanto/vehicle-images-dataset")
    if path is None:
        print("Failed to download the dataset.")
        return None
    return path


def plot_history(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs_range = range(len(acc))

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Train Accuracy')
    plt.plot(epochs_range, val_acc, label='Val Accuracy')
    plt.legend(loc='lower right')
    plt.title('Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Train Loss')
    plt.plot(epochs_range, val_loss, label='Val Loss')
    plt.legend(loc='upper right')
    plt.title('Loss')

    plt.show()


def train_model():
    IMAGE_SIZE = (224, 224)
    BATCH_SIZE = 32

    DATASET_DIR = download_data()
    if DATASET_DIR is None:
        return

    # Load dataset
    full_dataset = tf.keras.preprocessing.image_dataset_from_directory(
        DATASET_DIR,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        label_mode='categorical',
        shuffle=True,
        seed=123
    )

    class_names = full_dataset.class_names
    num_classes = len(class_names)
    np.save(os.path.join(MODELS_DIR, "class_names.npy"), class_names)

    # Split into train/validation
    train_size = 0.8
    train_ds = full_dataset.take(int(len(full_dataset) * train_size))
    val_ds = full_dataset.skip(int(len(full_dataset) * train_size))

    # Optimize dataset performance
    train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)

    # Calculate class weights to handle class imbalance


    class_indices = np.concatenate([y for x, y in full_dataset], axis=0).argmax(axis=1)
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(class_indices),
        y=class_indices
    )
    class_weights = {i: weight for i, weight in enumerate(class_weights)}

    # Build transfer learning model
    base_model = MobileNetV2(input_shape=(224,224,3), include_top=False, weights='imagenet')
    base_model.trainable = False  # freeze backbone

    model = models.Sequential([
        data_augmentation,
        layers.Rescaling(1./255),
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.3),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation='softmax')
    ], name="vehicle_classifier_transfer")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    # Callbacks
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.8, patience=4, min_lr=1e-6)
    ]

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=50,
        callbacks=callbacks,
        class_weight=class_weights  # Add class weights here
    )

    # Save training history
    history_dir = "training_history"
    os.makedirs(history_dir, exist_ok=True)
    with open(os.path.join(history_dir, "history.txt"), 'w') as f:
        f.write(str(history.history))

    # Save and convert model
    model.save(CAR_MODEL_PATH)
    print("Model saved to:", CAR_MODEL_PATH)
    compress_model(CAR_MODEL_PATH, os.path.join(MODELS_DIR, "vehicle_classifier_model.tflite"))
    os.remove(CAR_MODEL_PATH)

    plot_history(history)


def compress_model(input_model_path, output_model_path):
    model = models.load_model(input_model_path)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_types = [tf.float32]
    tflite_model = converter.convert()
    with open(output_model_path, 'wb') as f:
        f.write(tflite_model)


if __name__ == "__main__":
    train_model()