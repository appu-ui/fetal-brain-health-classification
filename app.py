import os
from flask import Flask, request, render_template, jsonify
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model

# Your class names in the order your model was trained on
class_names = [
    "anold-chiari-malformation",
    "arachnoid-cyst",
    "cerebellah-hypoplasia",
    "colphocephaly",
    "encephalocele",
    "holoprosencephaly",
    "hydracenphaly",
    "intracranial-hemorrdge",
    "intracranial-tumor",
    "m-magna",
    "mild-ventriculomegaly",
    "moderate-ventriculomegaly",
    "normal",
    "polencephaly",
    "severe-ventriculomegaly",
    "vein-of-galen"
] # <-- Replace with your actual class names

# Flask app
app = Flask(__name__)

# Load your trained model
MODEL_PATH = 'models/best_model.keras'  # or 'models/model.h5'
model = load_model(MODEL_PATH)
print('Model loaded. Check http://127.0.0.1:5000/')

def model_predict(img, model):
    img = img.convert('L')  # Convert to grayscale
    img = img.resize((64, 64))
    x = np.array(img) / 255.0
    x = np.expand_dims(x, axis=-1)  # (64, 64, 1)
    x = np.expand_dims(x, axis=0)   # (1, 64, 64, 1)
    preds = model.predict(x)
    pred_idx = np.argmax(preds)
    pred_class = class_names[pred_idx]
    pred_proba = float(np.max(preds))
    return pred_class, pred_proba

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    file = request.files['file']
    img = Image.open(file.stream)
    pred_class, pred_proba = model_predict(img, model)
    return jsonify(result=pred_class, probability=round(pred_proba, 3))

if __name__ == '__main__':
    http_server = WSGIServer(('0.0.0.0', 5000), app)
    http_server.serve_forever()
