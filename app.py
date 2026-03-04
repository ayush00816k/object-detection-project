import streamlit as st
import cv2
import numpy as np
from PIL import Image

st.title("Object Detection System")

uploaded_file = st.file_uploader("Upload an image")

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image")

    img = np.array(image)
    st.write("Object detection processing...")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    st.image(gray, caption="Processed Image")
