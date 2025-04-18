import numpy as np
import tensorflow as tf
from PIL import Image

CAR_MODEL_PATH = "models/vehicle_classifier_model.tflite"

classes = ['Big Truck', 'City Car', 'Multi Purpose Vehicle',
           'Sedan', 'Sport Utility Vehicle', 'Truck', 'Van']


def predict(img_path: str, debug=False):
    """Predict the class of a vehicle in an image using a TFLite model."""

    # Load TFLite model and allocate tensors
    interpreter = tf.lite.Interpreter(model_path=CAR_MODEL_PATH)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    input_shape = input_details[0]['shape']
    input_index = input_details[0]['index']
    output_index = output_details[0]['index']

    # Load and preprocess image
    img = Image.open(img_path).convert('RGB')
    # Resize to model's expected size
    img = img.resize((input_shape[1], input_shape[2]))
    img = np.array(img)

    if debug:
        print("Model loaded successfully")
        print(f"Original image shape: {img.shape}")
        print(f"Input shape expected by model: {input_shape}")

    # Optional: normalize image if model expects float inputs
    if input_details[0]['dtype'] == np.float32:
        img = img / 255.0

    # Add batch dimension
    input_data = np.expand_dims(img, axis=0).astype(input_details[0]['dtype'])

    # Set input tensor
    interpreter.set_tensor(input_index, input_data)

    # Run inference
    interpreter.invoke()

    # Get prediction
    output_data = interpreter.get_tensor(output_index)
    predicted_class = np.argmax(output_data[0])

    if debug:
        print("Raw prediction scores:", output_data[0])

    print(f"Predicted class: {predicted_class} ({classes[predicted_class]})")
    return classes[predicted_class], output_data[0][predicted_class]

