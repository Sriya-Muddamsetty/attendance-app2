import os
import cv2
import numpy as np
from flask import Flask, render_template, jsonify, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Initialize Flask app
app = Flask(__name__)

# Load the trained model
model = load_model('class_roll_number_model.h5')

# Load the class labels (roll numbers) from the saved file
class_names = np.load('classes.npy', allow_pickle=True)

# Define the image size expected by the model
img_size = (100, 100)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Check if the image is sent
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400

    # Get the image from the request
    img_file = request.files['image']
    
    # Convert the image to numpy array
    img_array = np.array(bytearray(img_file.read()), dtype=np.uint8)
    
    # Convert byte array to OpenCV image
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    if img is None:
        return jsonify({'error': 'Failed to decode image'}), 400

    # Resize the image to the model's expected input size
    img = cv2.resize(img, img_size)
    
    # Normalize the image
    img = img / 255.0

    # Prepare image for prediction (add batch dimension)
    img_array = np.expand_dims(img, axis=0)

    # Make prediction
    prediction = model.predict(img_array)
    
    # Debugging: Check prediction output
    print(f"Prediction output: {prediction}")  # This will print the raw output of the prediction
    predicted_label = np.argmax(prediction, axis=1)
    
    # Debugging: Check the predicted label type and value
    print(f"Predicted label (before conversion): {predicted_label} (Type: {type(predicted_label)})")

    # Convert numpy.int64 to a regular Python int (ensure it's serializable)
    predicted_label = int(predicted_label[0])
    
    # Debugging: After conversion
    print(f"Predicted label (after conversion): {predicted_label} (Type: {type(predicted_label)})")

    # Map the predicted label back to the roll number (class name)
    roll_number = class_names[predicted_label]

    # Ensure roll_number is a string type
    roll_number = str(roll_number)

    # Return the predicted roll number as JSON
    return jsonify({'roll_number': roll_number})

if __name__ == '__main__':
    app.run(debug=True)
