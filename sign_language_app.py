import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

model = tf.keras.models.load_model('output_model.h5')

class_names = {
    0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I',
    9: 'K', 10: 'L', 11: 'M', 12: 'N', 13: 'O', 14: 'P', 15: 'Q', 16: 'R',
    17: 'S', 18: 'T', 19: 'U', 20: 'V', 21: 'W', 22: 'X', 23: 'Y'
}

def preprocessor_image(image):
    image = image.resize((28, 28))
    image = image.convert('L')
    image = np.array(image) / 255.0
    image = image.reshape(1, 28, 28, 1)
    return image

st.title("Sign Language Alphabet Classifier")
st.markdown("<style>body {background-color: #f0f0f0;}</style>", unsafe_allow_html=True)
st.markdown("<h2 style='color: blue;'>Upload an image of a hand gesture to predict the sign language digit:</h2>", unsafe_allow_html=True)

uploaded_file = st.file_uploader("Choose an image...", type=["jpg","jpeg","png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image,  caption='Uploaded Image', use_column_width=True)

    if st.button('Predict'):
        processed_image = preprocessor_image(image)
        prediction = model.predict(processed_image)
        predicted_class = np.argmax(prediction)

        predicted_alphabet = class_names[predicted_class]

        st.markdown(f"<h3 style='color: green;'>Predicted Alphabet: {predicted_alphabet}</h3>", unsafe_allow_html=True)

st.markdown("""
    <style>
    body {
        background-color: #f4f4f9;
    }
    h1, h2 {
        color: #4a69bd;
        text-align: center;
    }
    .stButton>button {
        background-color: #60a3bc;
        color: white;
        border-radius: 8px;
    }
    </style>
""", unsafe_allow_html=True)