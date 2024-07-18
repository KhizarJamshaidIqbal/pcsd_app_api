import os
import numpy as np
import tensorflow as tf
from PIL import Image

# Load the TensorFlow Lite model
tflite_model_path = 'C:/Users/Khizar Jamshed Iqbal/Desktop/FYP(PCSADA)/api/VGG16_Model.tflite'

# Check if the model file exists and is accessible
if not os.path.exists(tflite_model_path):
    raise FileNotFoundError(f"TensorFlow Lite model file '{tflite_model_path}' not found or inaccessible")

# Initialize TensorFlow Lite interpreter
try:
    interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
    interpreter.allocate_tensors()

    # Get input and output details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Load class labels
    labels_path = "C:/Users/Khizar Jamshed Iqbal/Desktop/FYP(PCSADA)/api/labels.txt"
    if not os.path.exists(labels_path):
        raise FileNotFoundError(f"Labels file '{labels_path}' not found or inaccessible")
    
    with open(labels_path, "r") as file:
        class_labels = [line.strip() for line in file]

except Exception as e:
    print(f"Error initializing TensorFlow Lite interpreter: {e}")
    raise

def process_image(img: Image.Image):
    # Preprocess the image
    img = img.resize((224, 224))
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array.astype(np.float32)
    img_array = img_array / 255.0

    try:
        # Set the tensor to point to the input data to be inferred
        interpreter.set_tensor(input_details[0]['index'], img_array)

        # Run inference
        interpreter.invoke()

        # Get the results
        output_data = interpreter.get_tensor(output_details[0]['index'])
        top_3_indices = np.argsort(output_data[0])[-3:][::-1]
        top_3_labels = [(class_labels[i], float(output_data[0][i])) for i in top_3_indices]

        return top_3_labels

    except Exception as e:
        print(f"Error processing image: {e}")
        raise
