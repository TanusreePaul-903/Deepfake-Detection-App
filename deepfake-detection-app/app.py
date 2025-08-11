import streamlit as st 
from PIL import Image
import numpy as np
import tensorflow as tf
import cv2
import time

# Page configuration------------------------
st.set_page_config(page_title="DeepFake.AI Detection", layout="wide")

# Load model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model('model/deepfake_model_detection.keras')

model = load_model()

def preprocess_image(image):
    img = np.array(image)
    if img.shape[-1] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
    img = cv2.resize(img, (299, 299))
    img = img.astype(np.float32) / 255.0
    img = np.expand_dims(img, axis=0)
    return img

# CSS----------------------------------------
st.markdown("""
<style>
/* Hide default Streamlit header and footer */
#MainMenu, footer, header {visibility: hidden;}

/* Navbar */
.navbar {
    display: flex;
    justify-content: space-between;
    align-items: center;
    background-color: #ffffff;
    padding: 20px 50px;
    position: fixed;
    top: 0; left: 0;
    width: 100%;
    z-index: 1000;
    box-shadow: 0 2px 8px rgba(0,0,0,0.05);
}
.navbar .logo {
    font-weight: bold;
    font-size: 22px;
    color: #000000;
}
.navbar a {
    text-decoration: none;
    margin: 0 15px;
    font-weight: 600;
    color: #4a00e0;
    font-size: 16px;
}
.navbar a.active {
    border: 1px solid #ccc;
    padding: 6px 14px;
    border-radius: 20px;
}

/* Hero Section */
.hero {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 0px 80px 60px; /* Adjust top padding to avoid overlap with navbar */
    background-color: #ffffff;
}

.hero-left h1 {
    font-size: 4.5rem;
    font-weight: bold;
    color: #000000;
    margin-bottom: 20px;
}
.hero-left p {
    font-size: 1.2rem;
    color: #555;
    margin-bottom: 40px;
}
.verify-btn {
    background-color: #6c00ff;
    color: #fff;
    padding: 16px 35px;
    font-size: 20px;
    font-weight: bold;
    border: none;
    border-radius: 50px;
    cursor: pointer;
    transition: 0.3s ease-in-out;
}
.verify-btn:hover {
    background-color: #4a00e0;
}
.hero-right img {
    width: 100%;
    max-width: 400px;
}

/* Result */
.result-box {
    padding: 30px;
    margin-top: 30px;
    background: white;
    border-radius: 12px;
    box-shadow: 0 4px 8px rgba(0,0,0,0.07);
    text-align: center;
}
.result-box h2 {
    font-size: 28px;
}
.result-box .real {
    color: green;
    font-weight: bold;
}
.result-box .fake {
    color: red;
    font-weight: bold;
}

/* Upload Section */
.stFileUploader {
    margin-top: 20px;
}
</style>
""", unsafe_allow_html=True)

# Navbar-----------------------------------
st.markdown("""
<div class="navbar">
    <div class="logo">DeepFake.AI</div>
    <div>
        <a href="#" class="active">Home</a>
        <a href="#">About</a>
        <a href="#">Asset</a>
        <a href="#">Contact</a>
    </div>
</div>
""", unsafe_allow_html=True)

# Hero Section------------------------------
st.markdown('<div class="hero">', unsafe_allow_html=True)
col1, col2 = st.columns(2)

with col1:
    st.markdown("""
        <div class="hero-left">
            <h1>Detect Deepfakes<br>With Confidence</h1>
            <p>Upload an image and find out whether it's REAL or FAKE</p>
            <button class="verify-btn">Verify an Image 🔍</button>
        </div>
    """, unsafe_allow_html=True)

with col2:
     st.image("images/image.jpg", width=500)

st.markdown('</div>', unsafe_allow_html=True)

# Full-width What We Do section--------------------
st.markdown("""
    <style>
        .what-we-do-full {
            width: 100%;
            background: linear-gradient(to right, #f5f5ff, #f8f8ff);
            padding: 4rem 1rem;
            box-sizing: border-box;
        }
        .what-we-do-full h2 {
            text-align: center;
            font-size: 3rem;
            margin-bottom: 3rem;
            color: #000;
        }
        .features-wrapper {
            display: flex;
            justify-content: space-evenly;
            flex-wrap: wrap;
            max-width: 1200px;
            margin: 0 auto;
        }
        .feature {
            flex: 1;
            max-width: 320px;
            margin: 1rem;
            text-align: center;
        }
        .icon-bg {
            background: white;
            width: 80px;
            height: 80px;
            margin: 0 auto 1rem;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 2rem;
            color: purple;
            box-shadow: 0 4px 10px rgba(0, 0, 0, 0.1);
        }
        .feature h3 {
            font-size: 1.25rem;
            margin-bottom: 0.5rem;
            font-weight: bold;
        }
        .feature p {
            font-size: 0.95rem;
            color: #555;
        }

        @media screen and (max-width: 768px) {
            .features-wrapper {
                flex-direction: column;
                align-items: center;
            }
        }
    </style>

    <div class="what-we-do-full ">
        <h2>What We Do</h2>
        <div class="features-wrapper">
            <div class="feature">
                <div class="icon-bg">💡</div>
                <h3>Deep-Neural Network Training</h3>
                <p>Every pixel is scanned using deep neural networks trained on thousands of real and fake images.</p>
            </div>
            <div class="feature">
                <div class="icon-bg">⚡</div>
                <h3>Fast & Accurate</h3>
                <p>Get results in seconds with over 90% accuracy using our hybrid model.</p>
            </div>
            <div class="feature">
                <div class="icon-bg">🛡️</div>
                <h3>AI-Powered Defense</h3>
                <p>From misinformation to fraud, we protect the truth—one image at a time.</p>
            </div>
        </div>
    </div>
""", unsafe_allow_html=True)
# Centered, large heading-----------------
st.markdown("""
    <style>
        .custom-heading {
            text-align: center;
            font-size: 2.8rem;
            font-weight: 700;
            margin-top: 2rem;
            margin-bottom: 1rem;
            color: #111;
        }
    </style>

    <div class="custom-heading">Upload an image for Deepfake Detection</div>
""", unsafe_allow_html=True)

# Detection Section------------------------
uploaded_file = st.file_uploader("Upload an image for Deepfake Detection", type=["jpg", "jpeg", "png"], key="predictor")

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    if st.button("Check for Real or Fake", key="checkbtn"):
        with st.spinner("Analyzing..."):
            processed = preprocess_image(image)
            start = time.time()
            pred = model.predict(processed)
            elapsed = time.time() - start
            fake_prob = pred[0][0]
            result = "FAKE" if fake_prob > 0.5 else "REAL"
            confidence = fake_prob if fake_prob > 0.5 else 1 - fake_prob

        st.markdown(f"""
        <div class="result-box">
            <h2>Prediction Result</h2>
            <p><strong>Result:</strong> <span class="{ 'fake' if result == 'FAKE' else 'real' }">{result}</span></p>
            <p><strong>Confidence:</strong> {confidence:.2%}</p>
            <p><strong>Processing Time:</strong> {elapsed:.2f} seconds</p>
        </div>
        """, unsafe_allow_html=True)

# Prediction results section--------------
st.markdown("""
    <style>
        .section-title {
            text-align: center;
            font-size: 2.5rem;
            font-weight: 700;
            margin-bottom: 0.5rem;
            color: #111;
        }
        .section-subtext {
            text-align: center;
            font-size: 1.1rem;
            margin-bottom: 2rem;
            color: #444;
        }
    </style>

    <div class="section-title">Prediction Results</div>
    <div class="section-subtext">Sample prediction results produced by our deepfake detection model</div>
""", unsafe_allow_html=True)

# Load images using PIL
img1 = Image.open("images/sundar_pichai.jpg")
img2 = Image.open("images/lady.jpg")
img3 = Image.open("images/elon_mask.jpg")

# Display in one row using columns
col1, col2, col3 = st.columns(3)
with col1:
    st.image(img1, width=500)
with col2:
    st.image(img2, width=500)
with col3:
    st.image(img3, width=500)


# Footer section-----------------------

st.markdown("""
    <style>
    .footer {
        position: fixed;
        bottom: 0;
        left: 0;
        width: 100%;
        background-color: #ffffff;
        border-top: 1px solid #eaeaea;
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 10px 40px;
        box-shadow: 0 -2px 8px rgba(0, 0, 0, 0.05);
        z-index: 1000;
    }
    .footer-left {
        font-weight: bold;
        font-size: 16px;
        color: #4a00e0;
    }
    .subscribe-btn {
        background-color: #6c00ff;
        color: white;
        padding: 8px 20px;
        border: none;
        border-radius: 25px;
        font-size: 14px;
        cursor: pointer;
        transition: background 0.3s ease;
    }
    .subscribe-btn:hover {
        background-color: #4a00e0;
    }
    </style>

    <div class="footer">
        <div class="footer-left">DeepFake.AI</div>
        <button class="subscribe-btn">Subscribe</button>
    </div>
""", unsafe_allow_html=True)

# The end--------------------------------------