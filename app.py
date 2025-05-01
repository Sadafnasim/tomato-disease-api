from flask import Flask, request, jsonify
from flask_cors import CORS
from pyngrok import ngrok
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from PIL import Image
import os

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load the model only once
model_path = 'tomato_disease_model.h5'
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file not found: {model_path}")
model = tf.keras.models.load_model(model_path)

# Path to the dataset on Google Drive (for Colab)
dataset_path = '/content/drive/MyDrive/tomato/train'  # Update with the actual path in Google Drive

# Function to prepare the image for prediction
def prepare_image(image):
    image = image.resize((224, 224))  # Resize to match model input
    image = np.array(image) / 255.0  # Normalize the image
    if image.shape[-1] == 4:
        image = image[..., :3]  # Convert RGBA to RGB
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    try:
        # Open the uploaded image and prepare it for prediction
        image = Image.open(file.stream).convert('RGB')
        processed_image = prepare_image(image)
        
        # Predict the class
        prediction = model.predict(processed_image)
        predicted_class = np.argmax(prediction)

        class_labels = [
            'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight',
            'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite',
            'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus',
            'Tomato___healthy'
        ]

        # Return the predicted class as JSON
        return jsonify({'prediction': class_labels[predicted_class]})

    except Exception as e:
        return jsonify({'error': f'Prediction failed: {str(e)}'})

# Start ngrok tunnel and Flask app
if __name__ == '__main__':
    # Authenticate ngrok (use your own auth token here)
    from pyngrok import ngrok
    ngrok.set_auth_token("YOUR_NGROK_AUTH_TOKEN")  # Replace with your actual ngrok auth token
    
    port = 5000
    public_url = ngrok.connect(port)
    print(" * ngrok tunnel running at:", public_url)

    # Start the Flask app
    app.run(host='0.0.0.0', port=port, debug=False, threaded=True)
