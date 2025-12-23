import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import re
import numpy as np
import json

# Page configuration
st.set_page_config(
    page_title="Financial Sentiment Analyzer",
    page_icon="📈",
    layout="centered"
)

# Constants
MAX_SEQUENCE_LENGTH = 100

# Cache the model loading
@st.cache_resource
def load_sentiment_model():
    # Load model with custom handling for version compatibility
    from tensorflow import keras
    from tensorflow.keras import regularizers
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import (Dense, Dropout, Embedding, GRU, 
                                         Bidirectional, BatchNormalization)
    
    try:
        # Try loading the newer .keras format first
        model = keras.models.load_model('financial_sentiment_model.keras')
    except:
        try:
            # Try the .h5 format with compile=False
            model = keras.models.load_model('financial_sentiment_model.h5', compile=False)
            model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        except:
            # Recreate model and load weights
            model = Sequential([
                Embedding(input_dim=5000, output_dim=128, name='embedding_layer'),
                BatchNormalization(name='batch_norm_1'),
                Bidirectional(GRU(64, return_sequences=True,
                                 kernel_regularizer=regularizers.l2(0.01),
                                 recurrent_regularizer=regularizers.l2(0.01)),
                             name='bidirectional_gru_1'),
                Dropout(0.3, name='dropout_1'),
                BatchNormalization(name='batch_norm_2'),
                Bidirectional(GRU(32,
                                 kernel_regularizer=regularizers.l1(0.01),
                                 recurrent_regularizer=regularizers.l1(0.01)),
                             name='bidirectional_gru_2'),
                Dropout(0.3, name='dropout_2'),
                BatchNormalization(name='batch_norm_3'),
                Dense(64, activation='relu',
                      kernel_regularizer=regularizers.l1_l2(l1=0.01, l2=0.01),
                      name='dense_layer_1'),
                Dropout(0.4, name='dropout_3'),
                Dense(3, activation='softmax', name='output_layer')
            ])
            
            # Build model with input shape
            model.build(input_shape=(None, 100))
            model.load_weights('financial_sentiment_weights.h5')
            model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model

@st.cache_resource
def load_tokenizer():
    import json
    with open('tokenizer.json', 'r') as f:
        tokenizer_config = json.load(f)
    return tokenizer_config

@st.cache_resource
def load_label_encoder():
    import json
    with open('label_encoder.json', 'r') as f:
        label_classes = json.load(f)
    return label_classes

def clean_text(text):
    """Clean and preprocess text data"""
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = ' '.join(text.split())
    return text

def texts_to_sequences(text, word_index, num_words, oov_token):
    """Convert text to sequence using word_index"""
    words = text.lower().split()
    sequence = []
    oov_index = word_index.get(oov_token, 1)
    for word in words:
        idx = word_index.get(word, oov_index)
        if num_words is None or idx < num_words:
            sequence.append(idx)
        else:
            sequence.append(oov_index)
    return sequence

def predict_sentiment(text, model, tokenizer_config, label_classes):
    """Predict sentiment for given text"""
    # Clean the text
    cleaned = clean_text(text)
    
    # Tokenize and pad using saved word_index
    sequence = texts_to_sequences(
        cleaned, 
        tokenizer_config['word_index'], 
        tokenizer_config['num_words'],
        tokenizer_config['oov_token']
    )
    padded = pad_sequences([sequence], maxlen=MAX_SEQUENCE_LENGTH, padding='post')
    
    # Predict
    prediction = model.predict(padded, verbose=0)
    predicted_class = np.argmax(prediction, axis=1)[0]
    confidence = prediction[0][predicted_class] * 100
    
    sentiment = label_classes[predicted_class]
    probabilities = {
        label_classes[i]: float(prediction[0][i] * 100) 
        for i in range(len(label_classes))
    }
    
    return sentiment, confidence, probabilities
    
    return sentiment, confidence, probabilities

# Main app
def main():
    # Header
    st.title("📈 Financial Sentiment Analyzer")
    st.markdown("---")
    st.markdown("""
    Analyze the sentiment of financial news and statements using a deep learning model 
    trained on the Financial PhraseBank dataset.
    """)
    
    # Load model and tokenizer
    try:
        model = load_sentiment_model()
        tokenizer = load_tokenizer()
        label_encoder = load_label_encoder()
        st.success("✅ Model loaded successfully!")
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        st.info("Please make sure to run the notebook first to save the model files.")
        return
    
    st.markdown("---")
    
    # Input section
    st.subheader("📝 Enter Financial Text")
    
    # Text input
    user_input = st.text_area(
        "Type or paste your financial text here:",
        height=150,
        placeholder="e.g., The company reported strong quarterly earnings, exceeding analyst expectations..."
    )
    
    # Example texts
    st.markdown("**Or try these examples:**")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📈 Positive Example"):
            user_input = "The company's revenue increased by 25% exceeding all expectations."
            st.session_state.example_text = user_input
    
    with col2:
        if st.button("📉 Negative Example"):
            user_input = "The firm reported significant losses and is considering layoffs."
            st.session_state.example_text = user_input
    
    with col3:
        if st.button("➖ Neutral Example"):
            user_input = "The quarterly report was released on schedule as planned."
            st.session_state.example_text = user_input
    
    # Use example text if button was clicked
    if 'example_text' in st.session_state:
        user_input = st.session_state.example_text
        del st.session_state.example_text
    
    # Analyze button
    st.markdown("---")
    if st.button("🔍 Analyze Sentiment", type="primary", use_container_width=True):
        if user_input.strip():
            with st.spinner("Analyzing..."):
                sentiment, confidence, probabilities = predict_sentiment(
                    user_input, model, tokenizer, label_encoder
                )
            
            # Display results
            st.markdown("---")
            st.subheader("📊 Analysis Results")
            
            # Sentiment with color coding
            if sentiment == "positive":
                st.success(f"**Sentiment: {sentiment.upper()}** 📈")
                color = "#28a745"
            elif sentiment == "negative":
                st.error(f"**Sentiment: {sentiment.upper()}** 📉")
                color = "#dc3545"
            else:
                st.info(f"**Sentiment: {sentiment.upper()}** ➖")
                color = "#6c757d"
            
            # Confidence
            st.metric("Confidence", f"{confidence:.1f}%")
            
            # Probability breakdown
            st.markdown("**Probability Distribution:**")
            for label, prob in sorted(probabilities.items()):
                if label == "positive":
                    st.progress(prob / 100, text=f"Positive: {prob:.1f}%")
                elif label == "negative":
                    st.progress(prob / 100, text=f"Negative: {prob:.1f}%")
                else:
                    st.progress(prob / 100, text=f"Neutral: {prob:.1f}%")
        else:
            st.warning("⚠️ Please enter some text to analyze.")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: gray;'>
        <p>Built with ❤️ using TensorFlow and Streamlit</p>
        <p>Financial Phrase Bank Sentiment Analysis Project</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
