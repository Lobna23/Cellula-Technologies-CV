import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image


# Load the trained model
model_path = r"C:\Users\Lobna\Desktop\Teeth Cellula\densenet_model_2.h5"
model = tf.keras.models.load_model(model_path)


# Class labels (Adjust based on your dataset)
class_labels = ['CaS', 'CoS', 'Gum', 'MC', 'OC', 'OLP', 'OT']


# Function to preprocess image
def preprocess_image(img):
    img = img.resize((224, 224))  # Resize to match model input
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Normalize
    return img_array


# Streamlit UI
st.title("🦷 Teeth Disease Classifier")
st.write("Upload an image to classify the disease type.")


# Upload image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])


if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, use_container_width=True)



    # Preprocess image
    img_array = preprocess_image(img)


    # Predict
    prediction = model.predict(img_array)
    predicted_class = class_labels[np.argmax(prediction)]
    confidence = np.max(prediction) * 100


    # Display result
    st.write(f"### Predicted Class: {predicted_class}")
    



