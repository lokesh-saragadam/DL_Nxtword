import streamlit as st
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

#Load the LSTM Model
model=load_model('next_word_lstm (3).h5')

#Load the tokenizer
with open('tokenizer (3).pickle','rb') as handle:
    tokenizer=pickle.load(handle)

# Function to predict the next word

def predict_next_word(model, tokenizer, text, max_sequence_length):
    # Convert input text to token list
    token_list = tokenizer.texts_to_sequences([text])[0]

    # Ensure the token list is of the correct length
    if len(token_list) >= max_sequence_length:
        token_list = token_list[-(max_sequence_length - 1):]  # leave space for next word

    # Pad the sequence
    token_list = pad_sequences([token_list], maxlen=max_sequence_length, padding='pre')

    # Predict the next word
    predicted_probs = model.predict(token_list, verbose=0)
    predicted_word_index = np.argmax(predicted_probs, axis=1)[0]  # Get scalar from array

    # Map index back to word
    for word, index in tokenizer.word_index.items():
        if index == predicted_word_index:
            return word

    return None  # In case the word wasn't found


# streamlit app
st.title("Next Word Prediction With LSTM And Early Stopping")
input_text=st.text_input("Enter the sequence of Words")
if st.button("Predict Next Word"):
    max_sequence_len = model.input_shape[1] # Retrieve the max sequence length from the model input shape
    next_word = predict_next_word(model, tokenizer, input_text, max_sequence_len)
    st.markdown(f"**Next Word:** `{next_word}`")


