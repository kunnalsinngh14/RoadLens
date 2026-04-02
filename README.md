<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>RoadLens - Documentation</title>
    <style>
        body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; line-height: 1.6; color: #333; max-width: 800px; margin: 0 auto; padding: 40px 20px; }
        h1, h2, h3 { color: #1a202c; border-bottom: 1px solid #edf2f7; padding-bottom: 8px; }
        code { background: #f7fafc; padding: 2px 5px; border-radius: 4px; font-family: monospace; }
        pre { background: #1a202c; color: #fff; padding: 15px; border-radius: 8px; overflow-x: auto; font-family: monospace; }
        .feature-list { list-style: none; padding: 0; }
        .feature-list li::before { content: "🌟 "; }
        .tech-stack { display: flex; gap: 10px; flex-wrap: wrap; margin-bottom: 20px; }
        .tag { background: #ebf8ff; color: #2b6cb0; padding: 4px 12px; border-radius: 999px; font-size: 0.85rem; font-weight: 600; }
        footer { margin-top: 50px; padding-top: 20px; border-top: 2px solid #eee; text-align: center; font-style: italic; }
    </style>
</head>
<body>

    <header>
        <h1>🛣️ RoadLens: Real-Time Traffic Sign Recognition</h1>
        <p>RoadLens is an end-to-end intelligent system designed to detect and interpret traffic signs from live video or image uploads using <strong>Convolutional Neural Networks (CNN)</strong> and <strong>Computer Vision</strong>.</p>
    </header>

    <section id="features">
        <h2>Key Features</h2>
        <ul class="feature-list">
            <li><strong>Deep Learning Classification:</strong> Recognizes 43 distinct categories of traffic signs.</li>
            <li><strong>Image Preprocessing:</strong> Uses Grayscale conversion and Histogram Equalization.</li>
            <li><strong>Flask Web Interface:</strong> User-friendly portal for instant predictions.</li>
            <li><strong>Robust Training:</strong> Utilizes Data Augmentation (Rotation, Zoom, Shear).</li>
        </ul>
    </section>

    <section id="architecture">
        <h2>Technical Architecture</h2>
        <h3>1. The Model (CNN)</h3>
        <p>A Sequential CNN built with Keras/TensorFlow featuring Convolutional layers, Max Pooling, and Dropout (0.5) to prevent overfitting.</p>
        
        <h3>2. The Web Engine (Flask)</h3>
        <p>The <code>main.py</code> script handles secure file uploads and real-time image preprocessing to match the model's training input.</p>
    </section>

    <section id="stack">
        <h2>Tech Stack</h2>
        <div class="tech-stack">
            <span class="tag">Python</span>
            <span class="tag">TensorFlow</span>
            <span class="tag">OpenCV</span>
            <span class="tag">Flask</span>
            <span class="tag">NumPy</span>
            <span class="tag">Scikit-learn</span>
        </div>
    </section>

    <section id="structure">
        <h2>Project Structure</h2>
<pre>
RoadLens/
├── Dataset/             # Training images (0-42)
├── uploads/             # User-uploaded images
├── templates/
│   └── index.html       # Web UI
├── train_model.py       # CNN training script
├── main.py              # Flask Application
├── model.h5             # Saved trained model
└── labels.csv           # Class name reference
</pre>
    </section>

    <section id="getting-started">
        <h2>🚀 Getting Started</h2>
        <h3>1. Prerequisites</h3>
<pre>
pip install tensorflow opencv-python flask werkzeug numpy pandas matplotlib scikit-learn
</pre>

        <h3>2. Running the App</h3>
        <p>Start the Flask server by running:</p>
<pre>
python main.py
</pre>
        <p>Then navigate to <code>http://127.0.0.1:5001</code> in your browser.</p>
    </section>

    <footer>
        <p>Developed by Kunal Singh — B.Tech Computer Science (AI/ML) @ RV University.</p>
    </footer>

</body>
</html>