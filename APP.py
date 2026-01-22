# app.py
import streamlit as st
from PIL import Image
import numpy as np
import os
import gdown
import zipfile
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# -------------------------------
# CONFIGURATION
# -------------------------------
IMG_SIZE = (224, 224)
BATCH_SIZE = 8
BASE_DIR = "football_model"

# 🔴 REPLACE THESE WITH YOUR ACTUAL FILE IDs
MODEL_FILE_ID = "1YGpJpyRA_lA0wTITKCbw3m6Fclfu7dYh"
DATASET_FILE_ID = "1Yne5NKOzRKFNqDkFwFgjVq39BOPEaHTQ"

MODEL_PATH = os.path.join(BASE_DIR, "model.h5")
DATASET_ZIP_PATH = os.path.join(BASE_DIR, "dataset.zip")

# -------------------------------
# HELPER: FIND TRAIN DIRECTORY
# -------------------------------
def find_train_dir(base_dir):
    for root, dirs, files in os.walk(base_dir):
        if "train" in dirs:
            return os.path.join(root, "train")
    return None

# -------------------------------
# SETUP DIRECTORY
# -------------------------------
os.makedirs(BASE_DIR, exist_ok=True)

# -------------------------------
# DOWNLOAD MODEL
# -------------------------------
if not os.path.exists(MODEL_PATH):
    st.info("📥 Downloading model from Google Drive...")
    gdown.download(
        f"https://drive.google.com/uc?id={MODEL_FILE_ID}",
        MODEL_PATH,
        quiet=False
    )

# -------------------------------
# DOWNLOAD & EXTRACT DATASET
# -------------------------------
if not os.path.exists(DATASET_ZIP_PATH):
    st.info("📥 Downloading dataset from Google Drive...")
    gdown.download(
        f"https://drive.google.com/uc?id={DATASET_FILE_ID}",
        DATASET_ZIP_PATH,
        quiet=False
    )

st.info("📦 Extracting dataset...")
with zipfile.ZipFile(DATASET_ZIP_PATH, "r") as zip_ref:
    zip_ref.extractall(BASE_DIR)

# -------------------------------
# LOCATE TRAIN DIRECTORY SAFELY
# -------------------------------
TRAIN_DIR = find_train_dir(BASE_DIR)

if TRAIN_DIR is None:
    st.error("❌ Could not find a 'train/' directory inside the dataset.")
    st.stop()

# -------------------------------
# LOAD MODEL
# -------------------------------
st.info("🤖 Loading model...")
model = load_model(MODEL_PATH)

# -------------------------------
# LOAD CLASS LABELS FROM TRAIN DATA
# -------------------------------
train_datagen = ImageDataGenerator(rescale=1./255)

train_data = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical"
)

class_labels = list(train_data.class_indices.keys())

# -------------------------------
# STREAMLIT UI
# -------------------------------
st.title("⚽ Football Club Logo Recognition")
st.write("Upload a football club logo image and the model will predict the club.")

uploaded_file = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    img = img.resize(IMG_SIZE)

    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    pred = model.predict(img_array)
    predicted_index = np.argmax(pred)
    predicted_class = class_labels[predicted_index]
    confidence = pred[0][predicted_index]

    st.image(
        img,
        caption=f"Predicted: {predicted_class} ({confidence*100:.2f}%)",
        use_column_width=True
    )
