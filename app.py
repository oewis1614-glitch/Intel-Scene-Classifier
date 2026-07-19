import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# 1. Page Configuration
st.set_page_config(page_title="Intel Scene Classifier", page_icon="🌲", layout="centered")

# Custom layer loader to bypass version mismatch errors (like quantization_config)
class SafeDense(tf.keras.layers.Dense):
    def __init__(self, *args, **kwargs):
        kwargs.pop('quantization_config', None)
        super().__init__(*args, **kwargs)

# 2. Cache and Load the Full Model Safely
@st.cache_resource
def load_my_model():
    custom_objects = {'Dense': SafeDense}
    try:
        # Tries loading .h5 format first
        return tf.keras.models.load_model('intel_scene_model.h5', custom_objects=custom_objects)
    except Exception:
        # Fallback to .keras format if that's what you uploaded
        return tf.keras.models.load_model('intel_scene_model.keras', custom_objects=custom_objects)

with st.spinner("Loading CNN Model... Please wait"):
    model = load_my_model()

# Class names sorted exactly as in the dataset
CLASS_NAMES = ['buildings', 'forest', 'glacier', 'mountain', 'sea', 'street']

# 3. User Interface
st.title("🌲 Landscape Classification using CNN")
st.write("Upload any landscape image, and the model will instantly identify and classify it with high accuracy.")

# Image Upload Tool
uploaded_file = st.file_uploader("Choose an image (JPG, JPEG, PNG)...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)
    
    st.write("---")
    with st.spinner("Analyzing image and predicting class..."):
        # 4. Image Preprocessing
        img_resized = image.convert('RGB').resize((150, 150))
        img_array = np.array(img_resized) / 255.0  # Rescaling
        img_array = np.expand_dims(img_array, axis=0)  # Add Batch dimension
        
        # 5. Model Inference & Prediction
        predictions = model.predict(img_array)
        highest_class_idx = np.argmax(predictions[0])
        confidence = predictions[0][highest_class_idx] * 100
        predicted_class = CLASS_NAMES[highest_class_idx]

    # 6. Display Results Dynamically
    st.success(f"**Final Prediction:** This landscape represents **{predicted_class.upper()}**")
    st.metric(label="Confidence Level", value=f"{confidence:.2f}%")
    
    # Visualizing prediction distribution across all classes
    st.subheader("Classification Probability Distribution:")
    for name, pred in zip(CLASS_NAMES, predictions[0]):
        st.write(f"**{name.capitalize()}:**")
        st.progress(float(pred))