import streamlit as st
import numpy as np
import os
import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer

import pickle

# Load your model
model = load_model(r'Pre-Built-Models/next_word_model.h5')

# Initialize tokenizer (Assuming you have it saved or recreate it if needed)
# Load the tokenizer
with open(r'Pre-Built-Models/token.pkl', 'rb') as handle:
    tokenizer= pickle.load(handle)

def Predict_Next_Words(model, tokenizer, text):
    sequence = tokenizer.texts_to_sequences([text])
    sequence = np.array(sequence)
    preds = np.argmax(model.predict(sequence))
    predicted_word=""
    for key, value in tokenizer.word_index.items():
        if value == preds:
            predicted_word=(key)
            break
    st.text_area("",predicted_word)

st.markdown(f"""
<style>
    /* Set the background image for the entire app */
    .stApp {{
        background-color:#C0C0C0;
        background-size: 1300px;
        background-repeat: no-repeat;
        background-attachment: fixed;
        background-position: center;
    }}
    
    </style>
""", unsafe_allow_html=True)

  
# Streamlit interface
st.title("Predict The Next Word  ")
url='Image.jpg'
st.image(url,width=700)

st.write("please the text at least 3 words")

# Text input
text= st.text_input("Message:",)

# Button to classify
if st.button("Find the next word"):
    try:
        text = text.split(" ")
        text = text[-3:]
        Predict_Next_Words(model, tokenizer, text)
    except Exception as e:
        predicted_word=("error occured : ",e)
        st.text_area("",predicted_word)
        
# Display the result
    
# To run the app, use the command below in your terminal
# streamlit run your_script.py
