import streamlit as st
import numpy as np
import pickle
import tensorflow as tf
import keras
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load the LSTM model
model = load_model('next_word_lst.keras')

# Load the tokenizer
with open('tokenizer.pickle', 'rb') as handle:
    token = pickle.load(handle)

# Predict function with error handling
def predict_next_word(model, tokenizer, text, max_sequence_len):
    token_list = tokenizer.texts_to_sequences([text])[0]
    if len(token_list) >= max_sequence_len:
        token_list = token_list[-(max_sequence_len-1):]  # Ensure the sequence length matches max_sequence_len-1
    token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')
    predicted = model.predict(token_list, verbose=0)
    predicted_word_index = np.argmax(predicted, axis=1)
    for word, index in tokenizer.word_index.items():
        if index == predicted_word_index:
            return word
    return None

# App Title
st.title('🧠 Next Word Prediction with LSTM and Early Stopping')

# Input box
inp_text = st.text_input("Enter a sequence of words:", "To be or not to be")

# Predict button
if st.button('Predict Next Word'):
    max_seq_len = model.input_shape[1] + 1
    next_word = predict_next_word(model, token, inp_text, max_seq_len)

    if next_word is None or next_word.startswith("[⚠️"):
        st.warning("❌ Couldn't predict a next word. Please try a simpler or different phrase.")
        st.text(f"Error info: {next_word}")  # optional for debugging
    else:
        st.success(f"✅ Predicted Next Word: **{next_word}**")
