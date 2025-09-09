import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model
import streamlit as st

# Configuration
MAX_FEATURES = 10000  # Vocabulary size (adjust this to match your model)
MAX_LEN = 500        # Maximum sequence length

# Load the IMDB dataset word index
word_index = imdb.get_word_index()

# Filter word_index to match the vocabulary size used during training
filtered_word_index = {word: idx for word, idx in word_index.items() if idx < MAX_FEATURES - 3}
reverse_word_index = {value: key for (key, value) in filtered_word_index.items()}

# Load the pretrained model
try:
    model = load_model('simple_rnn_imdb.h5')
    model_loaded = True
except FileNotFoundError:
    st.error("Model file 'simple_rnn_imdb.h5' not found!")
    model_loaded = False
except Exception as e:
    st.error(f"Error loading model: {str(e)}")
    model_loaded = False

# Function to decode the reviews
def decode_review(encoded_review):
    return ' '.join([reverse_word_index.get(i - 3, '?') for i in encoded_review])

# Function to preprocess the input review
def preprocess_text(text):
    words = text.lower().split()
    encoded_review = []
    
    for word in words:
        if word in filtered_word_index:
            # Add 3 to match IMDB preprocessing (reserved indices: 0=padding, 1=start, 2=unknown)
            idx = filtered_word_index[word] + 3
            encoded_review.append(idx)
        else:
            # Use 2 for unknown words (UNK token)
            encoded_review.append(2)
    
    # Pad the sequence to MAX_LEN
    padded_review = sequence.pad_sequences([encoded_review], maxlen=MAX_LEN)
    return padded_review

# Prediction Function 
def predict_sentiment(review):
    if not review.strip():
        return "Please enter a review", 0.0
    
    try:
        preprocessed_input = preprocess_text(review)
        prediction = model.predict(preprocessed_input, verbose=0)
        sentiment = 'Positive' if prediction[0][0] > 0.5 else 'Negative'
        return sentiment, prediction[0][0]
    except Exception as e:
        return f"Error: {str(e)}", 0.0

# Streamlit app
st.title('🎬 IMDB Movie Review Sentiment Analysis')
st.write('Enter a movie review to classify it as positive or negative.')

if not model_loaded:
    st.stop()

# Display model info
with st.expander("Model Information"):
    st.write(f"- Vocabulary size: {MAX_FEATURES:,} words")
    st.write(f"- Maximum sequence length: {MAX_LEN} words")
    st.write(f"- Available vocabulary: {len(filtered_word_index):,} words")

# User input
user_input = st.text_area('Movie Review', 
                         placeholder="Enter your movie review here...",
                         height=150)

# Add example buttons
col1, col2, col3 = st.columns(3)

with col1:
    if st.button('Positive Example'):
        user_input = "This movie was absolutely fantastic! The acting was superb, the plot was engaging, and the cinematography was breathtaking. I loved every minute of it!"
        
with col2:
    if st.button('Negative Example'):
        user_input = "This movie was painfully boring and way too long. The story was predictable, the characters had no depth, and the dialogue felt forced. I kept waiting for something interesting to happen, but it never did."

with col3:
    if st.button('Clear'):
        user_input = ""

# Prediction
if st.button('🔍 Classify Sentiment'):
    if user_input.strip():
        with st.spinner('Analyzing sentiment...'):
            sentiment, score = predict_sentiment(user_input)
        
        if "Error" not in sentiment:
            # Display results
            col1, col2 = st.columns(2)
            
            with col1:
                if sentiment == 'Positive':
                    st.success(f"**Sentiment: {sentiment}** 😊")
                else:
                    st.error(f"**Sentiment: {sentiment}** 😞")
            
            with col2:
                st.metric("Confidence Score", f"{score:.4f}")
            
            # Visual representation
            st.subheader("Confidence Visualization")
            if sentiment == 'Positive':
                st.progress(float(score))
                st.write(f"Positive confidence: {score:.1%}")
            else:
                st.progress(float(1 - score))
                st.write(f"Negative confidence: {(1-score):.1%}")
            
            # Additional info
            words = user_input.lower().split()
            known_words = sum(1 for word in words if word in filtered_word_index)
            
            st.subheader("Analysis Details")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Words", len(words))
            with col2:
                st.metric("Known Words", known_words)
            with col3:
                if len(words) > 0:
                    coverage = (known_words / len(words)) * 100
                    st.metric("Vocabulary Coverage", f"{coverage:.1f}%")
        else:
            st.error(sentiment)
    else:
        st.warning('Please enter a movie review.')

# Add some helpful information
st.markdown("---")
st.subheader("How it works:")
st.write("""
1. **Text Preprocessing**: Your review is converted to lowercase and split into words
2. **Word Encoding**: Each word is mapped to a number using the IMDB vocabulary
3. **Sequence Padding**: The sequence is padded/truncated to exactly 500 words
4. **Prediction**: The RNN model outputs a probability score between 0 and 1
5. **Classification**: Scores > 0.5 are classified as Positive, ≤ 0.5 as Negative
""")

st.subheader("Tips for better results:")
st.write("""
- Use complete sentences rather than single words
- Include emotional language and descriptive adjectives
- The model works best with movie review-style text
- Very short reviews might not provide enough context
""")
