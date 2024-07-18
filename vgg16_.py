<<<<<<< HEAD
import numpy as np
import tensorflow as tf
from PIL import Image

# Load the TensorFlow Lite model
tflite_model_path = 'C:/Users/Khizar Jamshed Iqbal/Desktop/FYP(PCSADA)/api/VGG16_Model.tflite'
interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
interpreter.allocate_tensors()

# Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Load class labels
with open("C:/Users/Khizar Jamshed Iqbal/Desktop/FYP(PCSADA)/api/labels.txt", "r") as file:
    class_labels = [line.strip() for line in file]

def process_image(img: Image.Image):
    # Preprocess the image
    img = img.resize((224, 224))
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array.astype(np.float32)
    img_array = img_array / 255.0

    # Set the tensor to point to the input data to be inferred
    interpreter.set_tensor(input_details[0]['index'], img_array)

    # Run inference
    interpreter.invoke()

    # Get the results
    output_data = interpreter.get_tensor(output_details[0]['index'])
    top_3_indices = np.argsort(output_data[0])[-3:][::-1]
    top_3_labels = [(class_labels[i], float(output_data[0][i])) for i in top_3_indices]

    return top_3_labels
=======
import numpy as np
import tensorflow as tf
from PIL import Image

# Load the TensorFlow Lite model
tflite_model_path = 'C:/Users/Khizar Jamshed Iqbal/Desktop/FYP(PCSADA)/api/VGG16_Model.tflite'
interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
interpreter.allocate_tensors()

# Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Load class labels
with open("C:/Users/Khizar Jamshed Iqbal/Desktop/FYP(PCSADA)/api/labels.txt", "r") as file:
    class_labels = [line.strip() for line in file]

def process_image(img: Image.Image):
    # Preprocess the image
    img = img.resize((224, 224))
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array.astype(np.float32)
    img_array = img_array / 255.0

    # Set the tensor to point to the input data to be inferred
    interpreter.set_tensor(input_details[0]['index'], img_array)

    # Run inference
    interpreter.invoke()

    # Get the results
    output_data = interpreter.get_tensor(output_details[0]['index'])
    top_3_indices = np.argsort(output_data[0])[-3:][::-1]
    top_3_labels = [(class_labels[i], float(output_data[0][i])) for i in top_3_indices]

    return top_3_labels
>>>>>>> 57eacd3 (first commit)
