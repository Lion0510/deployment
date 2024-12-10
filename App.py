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
st.write("Mengunduh model Melspec dari Google Drive...")
download_model_from_google_drive(melspec_model_url, melspec_model_save_path)

st.write("Mengunduh model MFCC dari Google Drive...")
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

# Function to classify an audio file using the selected model
def classify_audio(file_path, model_type="mfcc"):
    if model_type == "mfcc":
        features = extract_mfcc(file_path)
        model = mfcc_model
    else:  # melspec
        features = extract_melspec(file_path)
        model = melspec_model
    
    # Add batch dimension
    features = np.expand_dims(features, axis=0)  # Shape: (1, 64, 64, 3)
    
    # Make predictions
    predictions = model.predict(features)
    
    # Get the predicted class
    predicted_class = np.argmax(predictions, axis=1)
    return predicted_class

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

# Model selection section
model_option = st.radio("Pilih model untuk klasifikasi:", ("mfcc", "melspec"))

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

    # Predict using the classify_audio function
    if st.button('Prediksi Kelas Burung'):
        with st.spinner("Memproses..."):
            predicted_class = classify_audio(temp_file_path, model_type=model_option)
            st.subheader("Hasil Prediksi:")
            st.write(f"**Prediksi Kelas:** {predicted_class[0]}")

# Footer
st.markdown("""
    <hr>
    <p style="text-align:center; font-size:12px; color:#888;">Aplikasi Klasifikasi Suara Burung menggunakan Deep Learning</p>
""")
