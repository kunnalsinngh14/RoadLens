import streamlit as st
import numpy as np
import cv2
import os
from PIL import Image

# ── Page Config ──
st.set_page_config(
    page_title="RoadLens — Traffic Sign AI",
    page_icon="🛣️",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ── Custom CSS ──
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&display=swap');

    .stApp {
        background: linear-gradient(135deg, #0a0f1e 0%, #0f172a 50%, #0a0f1e 100%);
        font-family: 'Inter', sans-serif;
    }

    /* Header */
    .hero-title {
        text-align: center;
        font-size: 2.75rem;
        font-weight: 800;
        background: linear-gradient(135deg, #3b82f6, #22d3ee);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 0.25rem;
        letter-spacing: -0.03em;
    }
    .hero-subtitle {
        text-align: center;
        color: #94a3b8;
        font-size: 1rem;
        margin-bottom: 2rem;
    }

    /* Glass Card */
    .glass-card {
        background: rgba(15, 23, 42, 0.75);
        backdrop-filter: blur(20px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 1.25rem;
        padding: 2rem;
        margin-bottom: 1.5rem;
    }

    /* Result Box */
    .result-box {
        background: rgba(16, 185, 129, 0.08);
        border: 1px solid rgba(16, 185, 129, 0.25);
        border-radius: 0.75rem;
        padding: 1.5rem;
        text-align: center;
        margin-top: 1rem;
    }
    .result-label {
        font-size: 0.7rem;
        text-transform: uppercase;
        letter-spacing: 0.12em;
        color: #64748b;
        font-weight: 600;
        margin-bottom: 0.3rem;
    }
    .result-value {
        font-size: 1.5rem;
        font-weight: 800;
        color: #10b981;
    }
    .result-conf {
        font-size: 0.85rem;
        color: #22d3ee;
        margin-top: 0.25rem;
    }

    /* Info Cards */
    .info-grid {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 0.75rem;
        margin-top: 1.5rem;
    }
    .info-card {
        background: rgba(30, 41, 59, 0.5);
        border: 1px solid rgba(255, 255, 255, 0.06);
        border-radius: 0.75rem;
        padding: 1rem;
        text-align: center;
    }
    .info-card .num {
        font-size: 1.5rem;
        font-weight: 800;
        color: #3b82f6;
    }
    .info-card .desc {
        font-size: 0.7rem;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-top: 0.2rem;
    }

    /* Camera badge */
    .camera-badge {
        display: inline-flex;
        align-items: center;
        gap: 0.4rem;
        background: rgba(239, 68, 68, 0.15);
        border: 1px solid rgba(239, 68, 68, 0.3);
        color: #ef4444;
        padding: 0.3rem 0.8rem;
        border-radius: 2rem;
        font-size: 0.75rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-bottom: 0.75rem;
    }

    /* Footer */
    .footer {
        text-align: center;
        color: #475569;
        font-size: 0.75rem;
        margin-top: 2rem;
        padding-bottom: 1rem;
    }

    /* Streamlit overrides */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0;
        background: rgba(30, 41, 59, 0.5);
        border-radius: 0.75rem;
        padding: 0.25rem;
        border: 1px solid rgba(255, 255, 255, 0.06);
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 0.6rem;
        color: #94a3b8;
        font-weight: 600;
        padding: 0.5rem 1.25rem;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, rgba(59, 130, 246, 0.25), rgba(34, 211, 238, 0.15)) !important;
        color: white !important;
    }
    div[data-testid="stFileUploader"] label { color: #94a3b8 !important; }
    .stButton > button {
        width: 100%;
        background: linear-gradient(135deg, #3b82f6, #2563eb);
        color: white;
        font-weight: 700;
        border: none;
        border-radius: 0.75rem;
        padding: 0.75rem;
        font-size: 0.95rem;
        transition: all 0.25s;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(59, 130, 246, 0.4);
    }
</style>
""", unsafe_allow_html=True)


# ── Load Model (cached) ──
@st.cache_resource
def load_traffic_model():
    """Load the trained model once and cache it."""
    import tensorflow as tf
    model_path = os.path.join(os.path.dirname(__file__), 'model.h5')
    model = tf.keras.models.load_model(model_path)
    return model


# ── Preprocessing ──
def preprocessing(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.equalizeHist(img)
    img = img / 255.0
    return img


def get_class_name(class_no):
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
        42: 'End of no passing by vehicles over 3.5 metric tons',
    }
    return classes.get(class_no, 'Unknown')


def predict_sign(img_array, model):
    """Run prediction on a BGR numpy image array."""
    img = cv2.resize(img_array, (32, 32))
    img = preprocessing(img)
    img = img.reshape(1, 32, 32, 1)
    predictions = model.predict(img, verbose=0)
    class_index = np.argmax(predictions, axis=1)[0]
    confidence = float(np.amax(predictions))
    class_name = get_class_name(class_index)
    return class_name, confidence


# ── UI ──
st.markdown('<h1 class="hero-title">RoadLens</h1>', unsafe_allow_html=True)
st.markdown('<p class="hero-subtitle">AI-powered traffic sign recognition — upload an image or use your camera.</p>', unsafe_allow_html=True)

# Load model
model = load_traffic_model()

# Stats
st.markdown("""
<div class="info-grid">
    <div class="info-card"><div class="num">43</div><div class="desc">Sign Classes</div></div>
    <div class="info-card"><div class="num">97.3%</div><div class="desc">Accuracy</div></div>
    <div class="info-card"><div class="num">CNN</div><div class="desc">Architecture</div></div>
</div>
""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# Tabs
tab_upload, tab_camera = st.tabs(["📤  Upload Image", "📷  Camera"])

with tab_upload:
    uploaded_file = st.file_uploader(
        "Drag and drop or click to upload a traffic sign image",
        type=["jpg", "jpeg", "png", "bmp", "webp"],
        label_visibility="collapsed",
    )

    if uploaded_file is not None:
        # Display uploaded image
        pil_image = Image.open(uploaded_file)
        st.image(pil_image, caption="Uploaded Image", use_container_width=True)

        # Predict
        if st.button("🔍  Analyze Sign", key="upload_btn"):
            with st.spinner("Analyzing..."):
                img_array = np.array(pil_image)
                # Convert RGB to BGR for OpenCV
                img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
                class_name, confidence = predict_sign(img_bgr, model)

            st.markdown(f"""
            <div class="result-box">
                <div class="result-label">Prediction</div>
                <div class="result-value">{class_name}</div>
                <div class="result-conf">Confidence: {confidence * 100:.1f}%</div>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="border: 2px dashed rgba(59,130,246,0.3); border-radius: 1rem; padding: 3rem 1rem; text-align: center;">
            <p style="color: #64748b; font-size: 0.9rem;">Upload a traffic sign image to get started</p>
            <p style="color: #475569; font-size: 0.75rem;">PNG, JPG, JPEG, BMP, WEBP</p>
        </div>
        """, unsafe_allow_html=True)

with tab_camera:
    st.markdown('<div class="camera-badge">● LIVE</div>', unsafe_allow_html=True)

    camera_image = st.camera_input(
        "Point your camera at a traffic sign",
        label_visibility="collapsed",
    )

    if camera_image is not None:
        # Decode camera frame
        pil_image = Image.open(camera_image)
        img_array = np.array(pil_image)
        img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

        class_name, confidence = predict_sign(img_bgr, model)

        st.markdown(f"""
        <div class="result-box">
            <div class="result-label">Detected Sign</div>
            <div class="result-value">{class_name}</div>
            <div class="result-conf">Confidence: {confidence * 100:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)

# Footer
st.markdown("""
<div class="footer">
    Powered by TensorFlow · Built with Streamlit · © 2026 Kunal Singh
</div>
""", unsafe_allow_html=True)
