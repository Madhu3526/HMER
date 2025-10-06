import streamlit as st
import tensorflow as tf
import numpy as np
import os
import re
from PIL import Image
from sympy import sympify, SympifyError
from sympy.parsing.latex import parse_latex
from tensorflow.keras.applications import ResNet50V2
from tensorflow.keras.applications.resnet_v2 import preprocess_input as resnet_preprocess_input

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Handwritten Math Solver",
    page_icon="🧮",
    layout="wide"
)

# --- 2. CONFIGURATION (Must match your training script) ---
EMBEDDING_DIM = 256
UNITS = 512
IMG_SIZE = (224, 224)
VOCAB_FILE = 'vocab.txt'
CHECKPOINT_DIR = './checkpoints'

# --- 3. VOCABULARY LOADING ---
@st.cache_data
def load_vocab(vocab_file):
    """Loads a vocabulary from a text file."""
    with open(vocab_file, 'r', encoding='utf-8') as f:
        vocab = [line.strip() for line in f]
    return vocab

try:
    full_vocabulary = load_vocab(VOCAB_FILE)
except FileNotFoundError:
    st.error(f"Error: Vocabulary file '{VOCAB_FILE}' not found. Please place it in the same folder as this script.")
    st.stop()

VOCAB_SIZE = len(full_vocabulary)
MAX_LENGTH = 14 # Set from your training output
clean_vocabulary = full_vocabulary[2:]
index_to_word = tf.keras.layers.StringLookup(
    vocabulary=clean_vocabulary, invert=True, oov_token='[UNK]', mask_token=''
)
start_token_index = full_vocabulary.index('<start>')


# --- 4. MODEL DEFINITIONS (Must be an exact match to your training script) ---
class CNN_Encoder(tf.keras.Model):
    def __init__(self, trainable=False):
        super(CNN_Encoder, self).__init__()
        self.image_model = ResNet50V2(include_top=False, weights='imagenet', input_shape=(*IMG_SIZE, 3))
        self.image_model.trainable = trainable
        feature_output_layer = self.image_model.get_layer('conv5_block3_out').output
        self.reshape = tf.keras.layers.Reshape((-1, feature_output_layer.shape[-1]))
        self.feature_extractor = tf.keras.Model(inputs=self.image_model.input, outputs=self.reshape(feature_output_layer))
    def call(self, x, training=False):
        return self.feature_extractor(x, training=training)

class BahdanauAttention(tf.keras.layers.Layer):
    def __init__(self, units):
        super(BahdanauAttention, self).__init__()
        self.W1, self.W2, self.V = tf.keras.layers.Dense(units), tf.keras.layers.Dense(units), tf.keras.layers.Dense(1)
    def call(self, features, hidden):
        hidden_with_time_axis = tf.expand_dims(hidden, 1)
        score = tf.nn.tanh(self.W1(features) + self.W2(hidden_with_time_axis))
        attention_weights = tf.nn.softmax(self.V(score), axis=1)
        context_vector = tf.reduce_sum(attention_weights * features, axis=1)
        return context_vector, attention_weights

class RNN_Decoder(tf.keras.Model):
    def __init__(self, embedding_dim, units, vocab_size):
        super(RNN_Decoder, self).__init__()
        self.units = units
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.lstm = tf.keras.layers.LSTM(self.units, return_sequences=True, return_state=True, recurrent_initializer='glorot_uniform')
        self.fc1 = tf.keras.layers.Dense(self.units)
        self.dropout = tf.keras.layers.Dropout(0.5)
        self.fc2 = tf.keras.layers.Dense(vocab_size)
        self.attention = BahdanauAttention(self.units)
    def call(self, x, features, hidden, training=False):
        context_vector, _ = self.attention(features, hidden[0])
        x = self.embedding(x)
        x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)
        output, state_h, state_c = self.lstm(x, initial_state=hidden)
        x = self.fc1(output)
        x = tf.reshape(x, (-1, x.shape[2]))
        x = self.dropout(x, training=training)
        x = self.fc2(x)
        return x, [state_h, state_c], None
    def reset_state(self, batch_size):
        return [tf.zeros((batch_size, self.units)) for _ in range(2)]

# --- 5. CACHED MODEL LOADING ---
@st.cache_resource
def load_model():
    """Loads and builds the trained model, then restores from a checkpoint."""
    encoder = CNN_Encoder(trainable=False)
    decoder = RNN_Decoder(EMBEDDING_DIM, UNITS, VOCAB_SIZE)
    
    # Build the models by running a dummy forward pass
    dummy_img = tf.random.normal((1, *IMG_SIZE, 3)) # 3 channels for ResNet
    dummy_features = encoder(dummy_img, training=False)
    dummy_hidden = decoder.reset_state(1)
    decoder(tf.random.uniform((1, 1), maxval=10, dtype=tf.int32), dummy_features, dummy_hidden, training=False)
    
    # Restore from the latest checkpoint
    checkpoint = tf.train.Checkpoint(encoder=encoder, decoder=decoder)
    latest_checkpoint = tf.train.latest_checkpoint(CHECKPOINT_DIR)
    
    if latest_checkpoint:
        checkpoint.restore(latest_checkpoint).expect_partial()
        print(f"Model restored from {latest_checkpoint}")
    else:
        # This will show an error in the app if checkpoints are missing
        return None, None
        
    return encoder, decoder

# --- 6. HELPER & PREDICTION FUNCTIONS ---
def load_and_preprocess_image(image_bytes):
    """Loads an image from bytes and preprocesses it for ResNet."""
    img = tf.io.decode_image(image_bytes, channels=3) # 3 channels for ResNet
    img = tf.image.resize(img, IMG_SIZE)
    img = resnet_preprocess_input(img)
    return img

def evaluate(image_bytes, encoder, decoder):
    """Generates a LaTeX prediction for a single image."""
    img_tensor = load_and_preprocess_image(image_bytes)
    features = encoder(tf.expand_dims(img_tensor, 0), training=False)
    hidden = decoder.reset_state(batch_size=1)
    dec_input = tf.expand_dims([start_token_index], 0)
    result = []
    for i in range(MAX_LENGTH):
        predictions, hidden, _ = decoder(dec_input, features, hidden, training=False)
        predicted_id = tf.argmax(predictions[0]).numpy()
        predicted_word = index_to_word(tf.constant([predicted_id])).numpy()[0].decode('utf-8')
        if predicted_word == '<end>': break
        if predicted_word not in ['<start>', '[UNK]', '']:
            result.append(predicted_word)
        dec_input = tf.expand_dims([predicted_id], 0)
    return " ".join(result)

def find_variables(latex_string):
    """Finds single-letter variables in a LaTeX string."""
    variables = re.findall(r'(?<!\\)[a-zA-Z]', latex_string)
    return sorted(list(set(variables)))

# --- 7. STREAMLIT APP LAYOUT ---
st.title("🧮 Handwritten Math Solver")
st.write("Upload an image of a handwritten expression, get the formula, and calculate the result.")

encoder, decoder = load_model()

if encoder is None or decoder is None:
    st.error("⚠️ Model checkpoints not found. Please make sure the 'training_checkpoints' folder is in the same directory as the app.")
    st.stop()

uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    col1, col2 = st.columns([1, 2])
    
    with col1:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)
        
        if st.button('🧠 Predict Formula'):
            image_bytes = uploaded_file.getvalue()
            with st.spinner('The model is thinking...'):
                prediction = evaluate(image_bytes, encoder, decoder)
                st.session_state.prediction = prediction
    
    with col2:
        if 'prediction' in st.session_state and st.session_state.prediction:
            st.subheader("Predicted Formula")
            st.latex(st.session_state.prediction)
            
            variables = find_variables(st.session_state.prediction)
            
            if variables:
                st.subheader("Enter Values for Variables")
                variable_values = {}
                # Create columns for a cleaner layout of variable inputs
                var_cols = st.columns(len(variables))
                for i, var in enumerate(variables):
                    with var_cols[i]:
                        variable_values[var] = st.number_input(f"Value for {var}:", value=1.0, format="%.2f", key=var)
                
                if st.button('Calculate Result'):
                    try:
                        expr = parse_latex(st.session_state.prediction)
                        
                        for var, value in variable_values.items():
                            expr = expr.subs(var, value)
                        
                        result = expr.evalf()
                        
                        st.success(f"**Result: {result}**")
                        
                    except (SympifyError, TypeError, Exception) as e:
                        st.error(f"Could not calculate the result. The formula might be too complex or malformed. Error: {e}")
            else:
                st.info("No variables found to calculate in the predicted formula.")