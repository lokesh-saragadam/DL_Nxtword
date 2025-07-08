import gradio as gr
from tensorflow.keras.models import load_model
import pickle
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences

model = load_model("next_word_lstm.h5")
with open("tokenizer.pickle", "rb") as f:
    tokenizer = pickle.load(f)

def predict_next(text):
    token_list = tokenizer.texts_to_sequences([text])[0]
    token_list = token_list[-(model.input_shape[1] - 1):]
    token_list = pad_sequences([token_list], maxlen=model.input_shape[1], padding='pre')
    predicted = model.predict(token_list, verbose=0)
    predicted_word_index = np.argmax(predicted, axis=1)[0]
    for word, index in tokenizer.word_index.items():
        if index == predicted_word_index:
            return word
    return "?"

gr.Interface(fn=predict_next, inputs="text", outputs="text", title="Next Word Predictor").launch()
