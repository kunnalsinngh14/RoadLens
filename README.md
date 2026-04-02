# 🛣️ RoadLens: Real-Time Traffic Sign Recognition

RoadLens is an end-to-end intelligent system designed to detect and interpret traffic signs from live video or image uploads. By leveraging **Convolutional Neural Networks (CNN)** and **Computer Vision**, RoadLens provides real-time translations of road instructions, enhancing driver safety and assisting autonomous navigation systems.

## 🌟 Key Features
* **Deep Learning Classification:** Recognizes 43 distinct categories of traffic signs.
* **Image Preprocessing Pipeline:** Uses Grayscale conversion and Histogram Equalization.
* **Flask Web Interface:** A user-friendly web portal for instant predictions.
* **Robust Training:** Utilizes Data Augmentation to improve model generalization.

---

## 🏗️ Technical Architecture

### 1. The Model (CNN)
The heart of RoadLens is a Sequential CNN built with Keras/TensorFlow. 
* **Convolutional Layers:** Extracts spatial features from 32x32 images.
* **Max Pooling:** Reduces dimensionality while retaining critical features.
* **Dropout Layers:** Prevents overfitting (set at 0.5).
* **Dense Layers:** 500 neurons with ReLU activation leading to a Softmax output for 43 classes.

### 2. The Web Engine (Flask)
The `main.py` script serves as the bridge between the trained `model.h5` and the frontend. It handles:
* Secure file uploads.
* Real-time image resizing and preprocessing.
* Class mapping to return human-readable labels.

---

## 🛠️ Tech Stack
* **Languages:** Python
* **ML Frameworks:** TensorFlow, Keras
* **Computer Vision:** OpenCV
* **Web Framework:** Flask
* **Data Analysis:** NumPy, Pandas, Matplotlib, Scikit-learn

---

## 📂 Project Structure
```
* **RoadLens/**
    * **Dataset/** — Training images categorized by folder (0-42)
    * **uploads/** — Temporary storage for user-uploaded images
    * **templates/**
        * `index.html` — Web UI for the prediction service
    * `train_model.py` — CNN training and validation script
    * `main.py` — Flask Application logic
    * `model.h5` — The saved trained model
    * `labels.csv` — Reference for class names
    * `README.md`
```
---

## 🚀 Getting Started

### 1. Prerequisites
Ensure you have Python 3.8+ installed. Install the required dependencies using pip:
```bash
pip install tensorflow opencv-python flask werkzeug numpy pandas matplotlib scikit-learn
```

### 2. Training the Model
If you wish to retrain the model with your own dataset:

Place your images in the Dataset/ folder.

Run the training script:

```Bash
python train_model.py
```
### 3. Running the Web App
Start the Flask server:

```Bash
python main.py
```
Open your browser and navigate to http://127.0.0.1:5001.

Upload a traffic sign image to see the prediction!
---
📊 Model Performance
Input Size: 32x32x1 (Grayscale)

Optimizer: Adam (Learning Rate: 0.001)

Loss Function: Categorical Crossentrop

Validation: 20% of the dataset used for validation.

---
## 🤝 Contributing
Contributions, issues, and feature requests are welcome! Feel free to connect if you want to contribute to the computer vision pipeline or the UI.

---

© 2026 Kunal Singh