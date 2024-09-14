import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image

modelFile = "C:/Users/shahz/OneDrive/Desktop/AI Project/dogs.h5"
model = tf.keras.models.load_model(modelFile)
inputShape = (331,331)

allLabels = np.load("C:/Users/shahz/OneDrive/Desktop/AI Project/allDogsLabels.npy")
categories = np.unique(allLabels)

def prepareImage(img):
    resized = cv2.resize(img, inputShape, interpolation=cv2.INTER_AREA)
    imgResult = np.expand_dims(resized, axis=0)
    imgResult = imgResult / 255.
    return imgResult

def main():
    st.set_page_config(
        page_title="Dog Breed Identifier",
        page_icon="🐶"
    )
    size1 = (120, 100)
    size2 = (180, 100) 
    sedlogo = "sedlogo.png"
    ssuetlogo = "ssuetlogo.png"

    col1, col2 = st.columns(2)

    with col1:
        img_left = Image.open(ssuetlogo).resize(size1)
        st.image(img_left, use_column_width=False)

    with col2:
        img_right = Image.open(sedlogo).resize(size2)
        st.image(img_right, use_column_width=False)

    st.title("Dog Breed Identifier")

    st.markdown("**Upload Your Dog's Image**", unsafe_allow_html=True)
    uploaded_image = st.file_uploader("",type=["jpg", "jpeg", "png"])

    if uploaded_image is not None:
        st.markdown("**Uploaded Image:**", unsafe_allow_html=True)
        image_bytes = np.frombuffer(uploaded_image.read(), dtype=np.uint8)
        img = cv2.imdecode(image_bytes, cv2.IMREAD_COLOR)

        st.image(img, channels="BGR", use_column_width=True)

        desired_size = (500, 500)
        img = cv2.resize(img, desired_size)

        image_for_model = prepareImage(img)

        result_array = model.predict(image_for_model, verbose=1)
        answers = np.argmax(result_array, axis=1)
        text = categories[answers[0]]

        st.subheader("Prediction:")
        st.markdown(f"<p style='font-size:24px; font-weight:bold;'>The predicted breed is: {text}</p>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
