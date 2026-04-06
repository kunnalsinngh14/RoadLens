from __future__ import division, print_function
import sys
import os
import base64
import numpy as np
import tensorflow as tf
import cv2

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

from flask import Flask, redirect, url_for, request, render_template, jsonify
from werkzeug.utils import secure_filename

app = Flask(__name__)

MODEL_PATH = 'model.h5'

model = load_model(MODEL_PATH)

def grayscale(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img

def equalize(img):
    img = cv2.equalizeHist(img)
    return img

def preprocessing(img):
    img = grayscale(img)
    img = equalize(img)
    img = img / 255
    return img

def getClassName(classNo):
    classes = {
        0: 'Speed Limit 20 km/h', 1: 'Speed Limit 30 km/h',
        2: 'Speed Limit 50 km/h', 3: 'Speed Limit 60 km/h',
        4: 'Speed Limit 70 km/h', 5: 'Speed Limit 80 km/h',
        6: 'End of Speed Limit 80 km/h', 7: 'Speed Limit 100 km/h',
        8: 'Speed Limit 120 km/h', 9: 'No passing',
        10: 'No passing for vehicles over 3.5 metric tons',
        11: 'Right-of-way at the next intersection',
        12: 'Priority road', 13: 'Yield', 14: 'Stop',
        15: 'No vehicles', 16: 'Vehicles over 3.5 metric tons prohibited',
        17: 'No entry', 18: 'General caution',
        19: 'Dangerous curve to the left',
        20: 'Dangerous curve to the right', 21: 'Double curve',
        22: 'Bumpy road', 23: 'Slippery road',
        24: 'Road narrows on the right', 25: 'Road work',
        26: 'Traffic signals', 27: 'Pedestrians',
        28: 'Children crossing', 29: 'Bicycles crossing',
        30: 'Beware of ice/snow', 31: 'Wild animals crossing',
        32: 'End of all speed and passing limits',
        33: 'Turn right ahead', 34: 'Turn left ahead',
        35: 'Ahead only', 36: 'Go straight or right',
        37: 'Go straight or left', 38: 'Keep right',
        39: 'Keep left', 40: 'Roundabout mandatory',
        41: 'End of no passing',
        42: 'End of no passing by vehicles over 3.5 metric tons'
    }
    return classes.get(classNo, 'Unknown')


def predict_from_image(img_bgr):
    """Predict traffic sign from a BGR numpy array."""
    img = cv2.resize(img_bgr, (32, 32))
    img = preprocessing(img)
    img = img.reshape(1, 32, 32, 1)
    predictions = model.predict(img, verbose=0)
    classIndex = np.argmax(predictions, axis=1)[0]
    confidence = float(np.amax(predictions))
    className = getClassName(classIndex)
    return className, confidence


def model_predict(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224))
    img = np.asarray(img)
    img = cv2.resize(img, (32, 32))
    img = preprocessing(img)
    img = img.reshape(1, 32, 32, 1)
    predictions = model.predict(img, verbose=0)
    classIndex = np.argmax(predictions, axis=1)[0]
    preds = getClassName(classIndex)
    return preds


@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def upload():
    f = request.files['file']
    basepath = os.path.dirname(__file__)
    file_path = os.path.join(basepath, 'uploads', secure_filename(f.filename))
    f.save(file_path)
    preds = model_predict(file_path, model)
    return preds


@app.route('/predict_frame', methods=['POST'])
def predict_frame():
    """Accept a base64-encoded webcam frame and return prediction."""
    data = request.get_json()
    if not data or 'image' not in data:
        return jsonify({'error': 'No image data received'}), 400

    # Decode base64 image
    img_data = data['image']
    # Strip the data URL prefix if present
    if ',' in img_data:
        img_data = img_data.split(',')[1]

    img_bytes = base64.b64decode(img_data)
    np_arr = np.frombuffer(img_bytes, np.uint8)
    img_bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    if img_bgr is None:
        return jsonify({'error': 'Could not decode image'}), 400

    className, confidence = predict_from_image(img_bgr)

    return jsonify({
        'prediction': className,
        'confidence': round(confidence * 100, 1)
    })


if __name__ == '__main__':
    app.run(port=5001, debug=True)