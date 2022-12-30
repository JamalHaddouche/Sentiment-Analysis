import streamlit as st
import numpy as np
import pickle
import h5py
from keras_preprocessing.sequence import pad_sequences
import tensorflow as tf
import pandas as pd



num_words = 10000
max_review_len = 200

# Load the model from the file
model = tf.keras.models.load_model('model.h5')

# Chargement de la tokenization à partir du fichier "tokenization.pkl"
with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)


def sigmoid_to_label(pred):
    pred = pred.copy()
    pred[pred <= 0.5] = 0.
    pred[pred > 0.5] = 1.
    return pred



st.markdown("<h1 style='text-align: center; color:#1278a9 ;'>Sentiment Analysis</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;color:#ffa500'>Realisé par Jamal Haddouche</p>", unsafe_allow_html=True)
# = Image.open('img.jpg')
st.image('imgs.webp', use_column_width=True)
text = st.text_area(' Donne ton opinion à propos de notre service', ' ')


pred_sequences = tokenizer.texts_to_sequences([text])
text_a_predire_final = pad_sequences(pred_sequences, maxlen=max_review_len)
#text_a_predire_final.shape

single_data=np.array([text_a_predire_final[0]])

if st.button('prédire le sentiment'):
    prediction=model.predict(single_data)
    label=sigmoid_to_label(prediction)

    if(label == 0):
        st.error(f"Le sentiment est négative avec une prababilité egale à = {(1-float(prediction[0])):.2}")
    else:
        st.success(f"Le sentiment est positif avec une prababilité egale à = {float(prediction[0]):.2}")