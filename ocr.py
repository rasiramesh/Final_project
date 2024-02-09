import streamlit as st
import cv2
import numpy as np
from PIL import Image
import pytesseract

# Set the path to the Tesseract executable
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Rest of your script follows...

# Initialize Streamlit application
st.title("Image Processing and OCR Application")

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Convert to OpenCV format
    opencv_image = np.array(image.convert('RGB')) 
    opencv_image = opencv_image[:, :, ::-1].copy() 

    # Image Pre-processing
    # Grayscale conversion
    gray_image = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2GRAY)
    st.image(gray_image, caption='Grayscale Image', use_column_width=True)

    # Thresholding
    _, thresh_image = cv2.threshold(gray_image, 150, 255, cv2.THRESH_BINARY)
    st.image(thresh_image, caption='Threshold Image', use_column_width=True)

    # Edge Detection
    edges = cv2.Canny(gray_image, 100, 200)
    st.image(edges, caption='Edge Detected Image', use_column_width=True)

    # OCR (if applicable)
    if st.checkbox("Perform OCR on Image"):
        text = pytesseract.image_to_string(image)
        st.write("Extracted Text:")
        st.text(text)
