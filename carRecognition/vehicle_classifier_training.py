import os
import tensorflow as tf
import keras
from keras import layers, models
from carRecognition.download_data import download_data

MODELS_DIR = "models"
if not os.path.exists(MODELS_DIR):
    os.makedirs(MODELS_DIR)


def train_model():
    # Set parameters
    IMAGE_SIZE = (224, 224)
    BATCH_SIZE = 32
    DATASET_DIR = download_data()
    if DATASET_DIR is None:
        print("Failed to download the dataset.")
        return

    # Load the dataset
    full_dataset = keras.preprocessing.image_dataset_from_directory(
        DATASET_DIR,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        label_mode='categorical',  # Multi-class classification
        shuffle=True,
        seed=123
    )

    # Split into training and validation
    train_size = 0.8
    train_dataset = full_dataset.take(int(len(full_dataset) * train_size))
    val_dataset = full_dataset.skip(int(len(full_dataset) * train_size))

    # Cache and prefetch for performance
    train_dataset = train_dataset.cache().shuffle(
        1000).prefetch(buffer_size=tf.data.AUTOTUNE)
    val_dataset = val_dataset.cache().prefetch(buffer_size=tf.data.AUTOTUNE)

    # Build the model
    model = models.Sequential([
        layers.Rescaling(1./255, input_shape=(224, 224, 3)),
        layers.Conv2D(32, 3, activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(128, 3, activation='relu'),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(7, activation='softmax')  # 7 classes
    ])

    # Compile the model
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    # Train the model
    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=30
    )

    # Save the training history
    history_dir = "training_history"
    if not os.path.exists(history_dir):
        os.makedirs(history_dir)
    history_file = os.path.join(history_dir, "history.txt")
    with open(history_file, 'w') as f:
        f.write(str(history.history))

    # Save the model
    model.save(os.path.join(MODELS_DIR, "vehicle_classifier_model.keras"))

    print("Model saved to:", os.path.join(
        MODELS_DIR, "vehicle_classifier_model.keras"))
    compress_model(os.path.join(MODELS_DIR, "vehicle_classifier_model.keras"),
                   os.path.join(MODELS_DIR, "vehicle_classifier_model.tflite"))

    # Delete the original model file after compression
    os.remove(os.path.join(MODELS_DIR, "vehicle_classifier_model.keras"))


def compress_model(input_model_path, output_model_path):
    # Load the model
    model = keras.models.load_model(input_model_path)

    # Convert the model to TensorFlow Lite format
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()

    # Save the compressed model
    with open(output_model_path, 'wb') as f:
        f.write(tflite_model)

