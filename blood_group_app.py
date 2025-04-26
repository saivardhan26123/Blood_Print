import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.imagenet_utils import preprocess_input
from PIL import Image
import io
import os
import time

# Set page configuration with wider layout
st.set_page_config(
    page_title="Blood Group Analysis System",
    page_icon="🩸",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #E53935;
        text-align: center;
        margin-bottom: 1rem;
        font-weight: 700;
    }
    .sub-header {
        font-size: 1.5rem;
        margin-bottom: 1rem;
        color: #424242;
        text-align: center;
        font-weight: 500;
    }
    .card {
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 1rem;
        background-color: white;
    }
    .info-box {
        background-color: #F3F4F6;
        padding: 1rem;
        border-radius: 8px;
        margin-bottom: 1rem;
    }
    .stButton button {
        background-color: #E53935;
        color: white;
        border-radius: 5px;
        padding: 0.5rem 1rem;
        font-weight: 500;
    }
    .metric-card {
        background-color: #FAFAFA;
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
    }
    .footer {
        text-align: center;
        color: #9E9E9E;
        margin-top: 2rem;
        padding-top: 1rem;
        border-top: 1px solid #EEEEEE;
    }
    .blood-icon {
        font-size: 1.2rem;
        color: #E53935;
        margin-right: 0.3rem;
    }
</style>
""", unsafe_allow_html=True)

# App header
st.markdown('<h1 class="main-header">🩸 BLOOD PRINT </h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">A Deep Learning Framework for Blood Group Prediction</p>', unsafe_allow_html=True)

# Create sidebar with information
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/drop-of-blood.png", width=80)
    st.markdown("## About This System")
    st.markdown("This AI-powered system uses deep learning to identify blood groups from sample fingerprint images.")
    
    st.markdown("### Supported Blood Types")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("• A+ (Positive)")
        st.markdown("• A- (Negative)")
        st.markdown("• B+ (Positive)")
        st.markdown("• B- (Negative)")
    with col2:
        st.markdown("• AB+ (Positive)")
        st.markdown("• AB- (Negative)")
        st.markdown("• O+ (Positive)")
        st.markdown("• O- (Negative)")
    
    st.markdown("### Model Information")
    st.markdown("The system uses a ResNet-based deep learning model trained on thousands of fingerprint sample images.")
    
    st.markdown("### How To Use")
    st.markdown("1. Upload a clear image of a fingerprint sample")
    st.markdown("2. Wait for the AI to analyze the image")
    st.markdown("3. Review the predicted blood group and confidence level")

# Main content area with cards
st.markdown('<div class="card">', unsafe_allow_html=True)

# Functions
import os

@st.cache_resource
def load_prediction_model():
    try:
        with st.spinner("Loading AI model..."):
            model_path = os.path.join(os.path.dirname(__file__), 'model_blood_group_detection_resnet.h5')
            model = load_model(model_path)
            return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# Define the class labels
labels = {'A+': 0, 'A-': 1, 'AB+': 2, 'AB-': 3, 'B+': 4, 'B-': 5, 'O+': 6, 'O-': 7}
labels = dict((v, k) for k, v in labels.items())

# Display model loading status
model = load_prediction_model()
if model:
    st.success("✅ ResNet model loaded successfully")
else:
    st.error("❌ Failed to load ResNet model. Please check if the model file exists.")

# Image upload section
st.markdown("## Upload Fingerprint Sample Image")
st.markdown("For accurate results, please upload a clear, well-lit image of a fingerprint sample.")

uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png", "bmp"])

# Example images section
st.markdown('<div class="info-box">', unsafe_allow_html=True)
st.markdown("### Don't have a sample image?")
st.markdown("You can use one of our example images or take a photo of a fingerprint sample.")
example_cols = st.columns(4)
use_example = False

# Function to make prediction
def predict_blood_group(img_data):
    try:
        # Resize and preprocess the image
        img = Image.open(img_data).convert('RGB')
        img = img.resize((256, 256))
        
        # Create columns for image and results
        col1, col2 = st.columns([1, 1.5])
        
        # Display the uploaded image
        with col1:
            st.markdown("### Sample Image")
            st.image(img, use_container_width=True)
        
        # Preprocess the image for the model
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        
        # Make prediction with a progress bar
        with col2:
            st.markdown("### Analysis Results")
            with st.spinner('AI analyzing fingerprint sample...'):
                # Add a progress bar for visual effect
                progress_bar = st.progress(0)
                for i in range(100):
                    # Simulate analysis progress
                    time.sleep(0.01)
                    progress_bar.progress(i + 1)
                
                # Make the actual prediction
                result = model.predict(x)
                
            # Clear the progress bar after completion
            progress_bar.empty()
            
            # Get prediction results
            predicted_class = np.argmax(result)
            predicted_label = labels[predicted_class]
            confidence = result[0][predicted_class] * 100
            
            # Display results with custom styling
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.markdown(f"### <span class='blood-icon'>🩸</span> Blood Group: {predicted_label}", unsafe_allow_html=True)
            st.markdown(f"#### Confidence: {confidence:.2f}%")
            
            # Display compatibility information
            st.markdown("#### Compatibility Information:")
            if predicted_label in ['O-']:
                st.markdown("🟢 **Universal Donor**: Can donate to all blood types")
            elif predicted_label in ['AB+']:
                st.markdown("🟢 **Universal Recipient**: Can receive from all blood types")
            
            st.markdown("</div>", unsafe_allow_html=True)
        
        # Show prediction distribution
        st.markdown("### Detailed Analysis Results")
        fig, ax = plt.subplots(figsize=(10, 5))
        blood_groups = list(labels.values())
        probabilities = result[0] * 100
        
        # Sort by probability for better visualization
        sorted_indices = np.argsort(probabilities)[::-1]  # Reverse to show highest first
        sorted_blood_groups = [blood_groups[i] for i in sorted_indices]
        sorted_probabilities = [probabilities[i] for i in sorted_indices]
        
        # Create a more attractive bar chart
        bars = ax.barh(sorted_blood_groups, sorted_probabilities, 
                 color=['#E53935' if i == 0 else '#90CAF9' for i in range(len(sorted_blood_groups))])
        ax.set_xlabel('Probability (%)')
        ax.set_ylabel('Blood Group')
        ax.set_title('Blood Group Probability Analysis')
        ax.grid(axis='x', linestyle='--', alpha=0.7)
        
        # Add percentage labels to bars
        for i, bar in enumerate(bars):
            width = bar.get_width()
            label_position = width if width > 5 else 5
            text_color = 'white' if width > 50 else 'black'
            ax.text(label_position, bar.get_y() + bar.get_height()/2, 
                    f"{sorted_probabilities[i]:.1f}%", 
                    va='center', ha='left' if width <= 5 else 'right', 
                    color=text_color, fontweight='bold')
            
        fig.tight_layout()
        st.pyplot(fig)
        
        # Add educational information about the predicted blood group
        st.markdown("### About This Blood Type")
        
        blood_group_info = {
            'A+': "Type A+ blood contains A antigens with Rh factor. People with A+ can receive blood from A+, A-, O+, and O- donors.",
            'A-': "Type A- blood contains A antigens without Rh factor. A- individuals can receive blood from A- and O- donors.",
            'B+': "Type B+ blood contains B antigens with Rh factor. B+ patients can receive blood from B+, B-, O+, and O- donors.",
            'B-': "Type B- blood contains B antigens without Rh factor. B- individuals can receive blood from B- and O- donors.",
            'AB+': "Type AB+ blood contains both A and B antigens with Rh factor. AB+ is the universal recipient and can receive blood from all types.",
            'AB-': "Type AB- blood contains both A and B antigens without Rh factor. AB- can receive blood from A-, B-, AB-, and O- donors.",
            'O+': "Type O+ blood contains no A or B antigens but has Rh factor. O+ individuals can receive blood from O+ and O- donors.",
            'O-': "Type O- blood contains no A or B antigens and no Rh factor. O- is the universal donor but can only receive O- blood."
        }
        
        st.info(blood_group_info.get(predicted_label, "Information not available for this blood group."))
        
    except Exception as e:
        st.error(f"Error during analysis: {e}")
        st.markdown("Please try again with a different image or check if the image is valid.")

# Process the uploaded file
if uploaded_file is not None:
    if model is not None:
        predict_blood_group(uploaded_file)
    else:
        st.warning("⚠️ Model could not be loaded. Please make sure the model file is available.")

st.markdown("</div>", unsafe_allow_html=True)

# Add footer
st.markdown('<div class="footer">', unsafe_allow_html=True)
st.markdown("Blood Group Detection System • Powered by AI • © 2025")
st.markdown("</div>", unsafe_allow_html=True)