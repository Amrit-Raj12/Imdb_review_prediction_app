import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model
import streamlit as st

# -----------------------------
# Constants
# -----------------------------
VOCAB_SIZE = 10000
MAX_LEN = 500

# -----------------------------
# Load IMDB word index
# -----------------------------
word_index = imdb.get_word_index()
word_index = {k: (v + 3) for k, v in word_index.items()}  # shift indices by 3
word_index["<PAD>"] = 0
word_index["<START>"] = 1
word_index["<UNK>"] = 2
word_index["<UNUSED>"] = 3

reverse_word_index = {value: key for key, value in word_index.items()}

# -----------------------------
# Load pretrained model
# -----------------------------
model = load_model("simple_rnn_imdb.h5")

# -----------------------------
# Decode review (for debugging)
# -----------------------------
def decode_review(encoded_review):
    return " ".join([reverse_word_index.get(i, "?") for i in encoded_review])

# -----------------------------
# Preprocess input text
# -----------------------------
def preprocess_text(text):
    words = text.lower().split()
    encoded_review = [word_index.get(word, 2) for word in words]  # 2 = <UNK>
    padded_review = sequence.pad_sequences(
        [encoded_review],
        maxlen=MAX_LEN,
        padding="post",
        truncating="post"
    )
    return padded_review

# -----------------------------
# Prediction function
# -----------------------------
def predict_sentiment(review):
    preprocessed_input = preprocess_text(review)
    prediction = model.predict(preprocessed_input)
    sentiment = "Positive" if prediction[0][0] > 0.5 else "Negative"
    return sentiment, float(prediction[0][0])

# -----------------------------
# Streamlit App
# -----------------------------
st.title("🎬 IMDB Movie Review Sentiment Analysis")
st.write("Enter a movie review to classify it as Positive or Negative.")

# User input
user_input = st.text_area("Movie Review", height=200)

if st.button("Classify"):
    if user_input.strip() != "":
        sentiment, score = predict_sentiment(user_input)
        st.write(f"**Sentiment:** {sentiment}")
        st.write(f"**Prediction Score:** {score:.4f}")
    else:
        st.warning("Please enter a valid review.")
