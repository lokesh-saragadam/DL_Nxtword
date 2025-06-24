import streamlit as st
import numpy as np
import pickle
import tensorflow as tf
import keras
from keras.model import load_model
from keras.preprocessing.sequences import pad_sequences


##Loading the LSTM model

model = load_model('next_word_lstm.h5')

##Load the tokenizer

with open('tokenizer.pkl','rb') as handle:
    token = pickle.load(handle)


#predict function

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

##Title

st.title('Next Word Prediction with LSTM And Early Stopping')
inp_text = st.text_input("Enter the Sequence of Words :","To be or not to be")

if st.button('Predict Next Word'):
    max_seq_len = model.input_shape[1]+1 ##maximum len of the input
    next_word = predict_next_word(model,token,inp_text,max_seq_len)
    st.write(f"Next Word: {next_word}")
    
