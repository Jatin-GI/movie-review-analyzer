import streamlit as st
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
import pickle

# ---------------------------
# Load tokenizer and model
# ---------------------------

with open('tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)

model = load_model('Sentiment_analyser_v1.h5')

MAX_SEQ_LEN = len(tokenizer.word_index)+1  # adjust as per your training

# ---------------------------
# Prediction Function
# ---------------------------

def predict_sentiment(text):
    token_text = tokenizer.texts_to_sequences([text])
    pad_text = pad_sequences(token_text, maxlen=MAX_SEQ_LEN - 1, padding='pre')
    pred_prob = model.predict(pad_text)[0][0]
    label = 'Negative' if pred_prob > 0.5 else 'Positive'
    return label, pred_prob

# ---------------------------
# Streamlit UI
# ---------------------------

st.set_page_config(page_title="🎬 Sentiment Analyzer", page_icon="💬", layout="centered")

# Title + Instructions
st.title("🎬 Movie Review Sentiment Analyzer")
st.caption("Enter any movie review below, and the model will detect whether it's positive or negative.")

# Text Input
user_input = st.text_area("📝 Write your movie review here:", 
                         height=180, 
                         placeholder="Example: The movie was absolutely amazing, I loved every part of it!")

# Analyze Button
if st.button("🔍 Analyze"):
    if user_input.strip() == "":
        st.warning("⚠️ Please enter a review before analyzing.")
    else:
        label, prob = predict_sentiment(user_input)
        
        # Calculate confidence and convert to Python float
        if label == "Negative":
            confidence = float(prob)  # Convert to Python float
            st.error(f"😠 **Prediction:** {label}")
        else:
            confidence = float(1 - prob)  # Convert to Python float
            st.success(f"😄 **Prediction:** {label}")
        
        # Show confidence
        st.markdown(f"#### 🔧 Model Confidence: `{confidence * 100:.2f}%`")
        st.progress(confidence)

# Footer
st.markdown("---")
st.markdown(
    "🚀 Built with ❤️ using **TensorFlow** and **Streamlit**",
    unsafe_allow_html=True
)