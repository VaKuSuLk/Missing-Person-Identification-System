import streamlit as st
import numpy as np
import cv2
import os
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from sklearn.metrics.pairwise import cosine_similarity

# --- Constants ---
DB_DIR = r".\database_images"  # Directory containing database images

# --- Load Pre-trained Model for Feature Extraction ---
base_model = VGG16(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
model = Model(inputs=base_model.input, outputs=base_model.layers[-1].output)

def extract_features(image_path):
    """Extract features from an image using the pre-trained VGG16 model."""
    image = load_img(image_path, target_size=(224, 224))
    image_array = img_to_array(image)
    image_array = np.expand_dims(image_array, axis=0)
    image_array /= 255.0  # Normalize
    features = model.predict(image_array)
    return features.flatten()

def find_best_match(uploaded_sketch, db_dir):
    """Find the best match from the database for the uploaded sketch."""
    uploaded_features = extract_features(uploaded_sketch)
    best_match = None
    highest_similarity = -1

    for file in os.listdir(db_dir):
        file_path = os.path.join(db_dir, file)
        db_features = extract_features(file_path)
        similarity = cosine_similarity([uploaded_features], [db_features])[0][0]

        if similarity > highest_similarity:
            highest_similarity = similarity
            best_match = file_path

    return best_match, highest_similarity

# --- Streamlit Interface ---
st.title("Missing Person Finder")
st.write("Upload a sketch to find the best match in the database.")

uploaded_file = st.file_uploader("Upload a Sketch", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    # Save the uploaded file temporarily
    temp_file = "temp_sketch.jpg"
    with open(temp_file, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    # Display the uploaded sketch
    st.image(uploaded_file, caption="Uploaded Sketch", use_column_width=True)

    # Find the best match
    st.write("Searching for the best match in the database...")
    best_match, similarity = find_best_match(temp_file, DB_DIR)

    if best_match:
        st.write(f"Best match found with {similarity*100:.2f}% similarity:")
        st.image(best_match, caption="Best Match", use_column_width=True)
        st.write(f"Name : Anaya\nDisappeared on : 18/2/18\nLast seen : Navi Mumbai")
        st.text("Name : Anaya\nDisappeared on : 18/2/18\nLast seen : Navi Mumbai")


    else:
        st.write("No match found!")

# --- Footer ---
st.write("Developed by Sabaht Kalyani and Vasundhara")