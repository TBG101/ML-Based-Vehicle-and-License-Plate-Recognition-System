# Import Libraries
import kagglehub
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import numpy as np
import os
import shutil
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
classes = ['Big Truck', 'City Car', 'Multi Purpose Vehicle',
           'Sedan', 'Sport Utility Vehicle', 'Truck', 'Van']


def download_data():
    """
    Downloads the vehicle images dataset from Kaggle and returns the local path.
    """
    path = kagglehub.dataset_download("lyensoetanto/vehicle-images-dataset")
    if path is None:
        print("Failed to download the dataset.")
        return None
    return path


def split_dataset(source_dir, base_target_dir, test_ratio=0.2, val_ratio=0.1, seed=42):
    """
    Splits dataset into train/val/test without external libraries
    """
    # Create target directories
    splits = ['train', 'val', 'test']
    for split in splits:
        os.makedirs(os.path.join(base_target_dir, split), exist_ok=True)

    # Process each class
    for class_name in os.listdir(source_dir):
        class_path = os.path.join(source_dir, class_name)
        if not os.path.isdir(class_path):
            continue

        # Get all files for this class
        files = [f for f in os.listdir(
            class_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        np.random.shuffle(files)

        # Split into temp (train+val) and test
        temp_files, test_files = train_test_split(
            files,
            test_size=test_ratio,
            random_state=seed
        )

        # Split temp into train and val
        train_files, val_files = train_test_split(
            temp_files,
            test_size=val_ratio/(1-test_ratio),  # Adjusted ratio
            random_state=seed
        )

        # Helper function to copy files
        def copy_files(file_list, split_name):
            target_dir = os.path.join(base_target_dir, split_name, class_name)
            os.makedirs(target_dir, exist_ok=True)
            for f in file_list:
                src = os.path.join(class_path, f)
                dst = os.path.join(target_dir, f)
                shutil.copyfile(src, dst)

        # Copy files to respective directories
        copy_files(train_files, 'train')
        copy_files(val_files, 'val')
        copy_files(test_files, 'test')


def compress_model(input_model_path, output_model_path):
    model = load_model(input_model_path)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_types = [tf.float32]
    tflite_model = converter.convert()
    with open(output_model_path, 'wb') as f:
        f.write(tflite_model)


def save_performance_graphs(history, output_dir='training_history'):
    """
    Saves training and validation accuracy and loss graphs.

    Parameters:
        history (dict): Training history containing accuracy and loss values.
        output_dir (str): Directory to save the graphs.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Plot accuracy
    plt.figure(figsize=(10, 6))
    plt.plot(history['accuracy'], label='Training Accuracy')
    plt.plot(history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'accuracy.png'))
    plt.close()

    # Plot loss
    plt.figure(figsize=(10, 6))
    plt.plot(history['loss'], label='Training Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'loss.png'))
    plt.close()


def train_model():
    # Usage: Split your raw dataset
    source_directory = download_data()
    target_directory = "./data/"  # Will be created

    if os.path.exists(target_directory):
        shutil.rmtree(target_directory)

    split_dataset(source_directory, target_directory)

    train_dir = os.path.join(target_directory, 'train')
    val_dir = os.path.join(target_directory, 'val')
    test_dir = os.path.join(target_directory, 'test')

    # Data augmentation for training
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True
    )

    # Validation and test generators (no augmentation)
    val_test_datagen = ImageDataGenerator(rescale=1./255)

    # Data generators
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical'
    )

    val_generator = val_test_datagen.flow_from_directory(
        val_dir,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical'
    )

    test_generator = val_test_datagen.flow_from_directory(
        test_dir,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=False  # Important for evaluation
    )

    # Build Model
    base_model = MobileNetV2(
        weights='imagenet',
        include_top=False,
        input_shape=(224, 224, 3)
    )

    # Freeze the base model
    base_model.trainable = False

    # Add custom layers
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(512, activation='relu')(x)
    predictions = Dense(train_generator.num_classes, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=predictions)

    # Compile
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    # Train the Model
    EPOCHS = 100

    callbacks = [
        EarlyStopping(monitor='val_loss', patience=3,
                      restore_best_weights=True),
        ModelCheckpoint('models/best_model.h5', save_best_only=True)
    ]

    history = model.fit(
        train_generator,
        epochs=EPOCHS,
        validation_data=val_generator,
        callbacks=callbacks
    )

    # save the history
    np.save('training_history/history.npy', history.history)

    # Save performance graphs
    save_performance_graphs(history.history)

    # Save the model
    model.save('models/best_model.h5')

    # Load best saved weights
    model.load_weights('models/best_model.h5')

    test_loss, test_acc = model.evaluate(test_generator)
    print(f"\nTest Accuracy: {test_acc*100:.2f}%")
    print(f"Test Loss: {test_loss:.4f}")

    compress_model('models/best_model.h5',
                   'models/vehicle_classifier_model.tflite')
    os.remove('models/best_model.h5')


if __name__ == "__main__":
    train_model()
