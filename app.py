from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import numpy as np
import tensorflow as tf
import os
from pyngrok import ngrok

app = Flask(__name__)
CORS(app)

# Load the model
model_path = 'tomato_disease_model.h5'
model = tf.keras.models.load_model(model_path)

# Class labels
class_labels = [
    'Tomato___Bacterial_spot',
    'Tomato___Early_blight',
    'Tomato___Late_blight',
    'Tomato___Leaf_Mold',
    'Tomato___Septoria_leaf_spot',
    'Tomato___Spider_mites Two-spotted_spider_mite',
    'Tomato___Target_Spot',
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
    'Tomato___Tomato_mosaic_virus',
    'Tomato___healthy'
]

# Image preprocessing
def prepare_image(image):
    image = image.resize((224, 224))
    image = np.array(image) / 255.0
    if image.shape[-1] == 4:
        image = image[..., :3]
    image = np.expand_dims(image, axis=0)
    return image

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    try:
        image = Image.open(file.stream).convert('RGB')
        processed_image = prepare_image(image)
        prediction = model.predict(processed_image)
        predicted_label = class_labels[np.argmax(prediction)]
        return jsonify({'prediction': predicted_label})
    except Exception as e:
        return jsonify({'error': f'Prediction failed: {str(e)}'})

if __name__ == '__main__':
    port = 5000
    public_url = ngrok.connect(port)
    print(" * ngrok tunnel running at:", public_url)
    app.run(host='0.0.0.0', port=port, debug=False, threaded=True)
