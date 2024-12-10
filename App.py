import streamlit as st
import numpy as np
import librosa
import tensorflow as tf
import gdown  # Import gdown for downloading files from Google Drive
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO

# Set Streamlit page configuration
st.set_page_config(page_title="Bird Song Classifier", page_icon="🦜", layout="centered")

# Function to download model from Google Drive using gdown
def download_model_from_google_drive(url, output_path):
    gdown.download(url, output_path, quiet=False)

# Google Drive model URLs
melspec_model_url = 'https://drive.google.com/uc?id=192VGvINbZKOyjhGioyBhjfd2alGe6ATM'  # Mel Spectrogram model URL
mfcc_model_url = 'https://drive.google.com/uc?id=1aRBAt6bHVMW3t6QwbLHzCPn3fQuqd71h'  # MFCC model URL

# Path to save the downloaded models
melspec_model_save_path = 'melspec_model.h5'
mfcc_model_save_path = 'mfcc_model.h5'

# Download the models from Google Drive
download_model_from_google_drive(melspec_model_url, melspec_model_save_path)
download_model_from_google_drive(mfcc_model_url, mfcc_model_save_path)

# Load the pre-trained models
melspec_model = tf.keras.models.load_model(melspec_model_save_path)
mfcc_model = tf.keras.models.load_model(mfcc_model_save_path)

# Function to extract MFCC features from an audio file
def extract_mfcc(file_path, sr=22050, n_mfcc=64, n_fft=2048, hop_length=512):
    # Load the audio file
    y, _ = librosa.load(file_path, sr=sr)
    
    # Extract MFCC features
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length)
    
    # Transpose to get shape (n_mfcc, time)
    mfccs = mfccs.T  # Shape: (time, n_mfcc)
    
    # Ensure the shape is (64, n_mfcc) for model input
    if mfccs.shape[0] < 64:
        mfccs = np.pad(mfccs, ((0, 64 - mfccs.shape[0]), (0, 0)), mode='constant')
    else:
        mfccs = mfccs[:64, :]
    
    # Duplicate the single-channel MFCCs to create a 3-channel input
    mfccs = np.stack([mfccs] * 3, axis=-1)  # Shape: (64, 64, 3)
    
    return mfccs

# Function to extract Mel Spectrogram features from an audio file
def extract_melspec(file_path, sr=22050, n_mels=64, n_fft=2048, hop_length=512):
    # Load the audio file
    y, _ = librosa.load(file_path, sr=sr)
    
    # Extract Mel Spectrogram features
    melspec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, n_fft=n_fft, hop_length=hop_length)
    
    # Convert to logarithmic scale (log-mel spectrogram)
    melspec = librosa.power_to_db(melspec, ref=np.max)
    
    # Transpose to get shape (n_mels, time)
    melspec = melspec.T  # Shape: (time, n_mels)
    
    # Ensure the shape is (64, n_mels) for model input
    if melspec.shape[0] < 64:
        melspec = np.pad(melspec, ((0, 64 - melspec.shape[0]), (0, 0)), mode='constant')
    else:
        melspec = melspec[:64, :]
    
    # Duplicate the single-channel Mel Spectrogram to create a 3-channel input
    melspec = np.stack([melspec] * 3, axis=-1)  # Shape: (64, 64, 3)
    
    return melspec

# Function to classify an audio file using both models (MFCC and Melspec)
def classify_audio(file_path):
    # Extract features for MFCC and Melspec
    mfcc_features = extract_mfcc(file_path)
    melspec_features = extract_melspec(file_path)
    
    # Add batch dimension for both
    mfcc_features = np.expand_dims(mfcc_features, axis=0)  # Shape: (1, 64, 64, 3)
    melspec_features = np.expand_dims(melspec_features, axis=0)  # Shape: (1, 64, 64, 3)
    
    # Make predictions using both models
    mfcc_predictions = mfcc_model.predict(mfcc_features)
    melspec_predictions = melspec_model.predict(melspec_features)
    
    # Get the predicted classes for both models
    mfcc_pred_class = np.argmax(mfcc_predictions, axis=1)
    melspec_pred_class = np.argmax(melspec_predictions, axis=1)
    
    # Get the accuracies (max prediction probabilities)
    mfcc_accuracy = np.max(mfcc_predictions)
    melspec_accuracy = np.max(melspec_predictions)
    
    return mfcc_pred_class[0], mfcc_accuracy, melspec_pred_class[0], melspec_accuracy

# Title of the app
st.title("Deep Learning in Audio: Klasifikasi Suara Burung di Indonesia Bagian Barat 🦜")

# Introduction
st.markdown("""
     **Selamat datang di aplikasi klasifikasi suara burung menggunakan Deep Learning!**
    Aplikasi ini mengimplementasikan teknik ekstraksi fitur audio menggunakan MFCC dan Melspectrogram, 
    serta menggunakan dua model CNN yang berbeda untuk mengklasifikasikan suara burung yang ada di Indonesia Bagian Barat.
    Unggah file audio dalam format MP3 atau WAV, dan model akan memberikan prediksi kelas burung beserta akurasi dari kedua model.
""")

# File upload section
st.header("Unggah File Audio Suara Burung")

uploaded_audio = st.file_uploader("Pilih file audio (MP3/WAV) untuk diuji", type=["mp3", "wav"])

if uploaded_audio is not None:
    # Display file details
    st.audio(uploaded_audio, format="audio/mp3")
    
    # Process the audio file
    audio_bytes = uploaded_audio.read()
    with BytesIO(audio_bytes) as audio_buffer:
        # Save the uploaded audio to a temporary file
        temp_file_path = 'temp_audio.wav'  # You can use any temporary file name
        with open(temp_file_path, 'wb') as f:
            f.write(audio_buffer.getbuffer())

    # Predict using both models
    if st.button('Prediksi Kelas Burung'):
        with st.spinner("Memproses..."):
            mfcc_class, mfcc_acc, melspec_class, melspec_acc = classify_audio(temp_file_path)
            
            # Display both predictions and accuracies
            st.subheader("Hasil Prediksi:")
            st.write(f"**Prediksi Kelas (Model MFCC):** {mfcc_class}")
            st.write(f"**Akurasi (Model MFCC):** {mfcc_acc * 100:.2f}%")
            
            st.write(f"**Prediksi Kelas (Model Melspec):** {melspec_class}")
            st.write(f"**Akurasi (Model Melspec):** {melspec_acc * 100:.2f}%")

# Footer
st.markdown("""
    <hr>
    <p style="text-align:center; font-size:14px; color:#888; margin-top: 10px; margin-bottom: 10px;">
        Aplikasi Klasifikasi Suara Burung menggunakan Deep Learning | Dibuat oleh Kelompok 11
    </p>
""", unsafe_allow_html=True)
