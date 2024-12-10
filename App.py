import streamlit as st
import numpy as np
import librosa
import tensorflow as tf
import gdown
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from io import BytesIO

# Download model dari Google Drive
melspec_model_url = 'https://drive.google.com/uc?id=192VGvINbZKOyjhGioyBhjfd2alGe6ATM'
mfcc_model_url = 'https://drive.google.com/uc?id=1aRBAt6bHVMW3t6QwbLHzCPn3fQuqd71h'

# Mendownload model Melspec dan MFCC
melspec_model_path = 'melspec_model.h5'
mfcc_model_path = 'mfcc_model.h5'

gdown.download(melspec_model_url, melspec_model_path, quiet=False)
gdown.download(mfcc_model_url, mfcc_model_path, quiet=False)

# Load kedua model
melspec_model = load_model(melspec_model_path)
mfcc_model = load_model(mfcc_model_path)

# Fungsi untuk mengekstrak MFCC dari file audio
def extract_mfcc(file_path, sr=22050, n_mfcc=64, n_fft=2048, hop_length=512):
    y, _ = librosa.load(file_path, sr=sr)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length)
    mfccs = mfccs.T  # Shape: (time, n_mfcc)
    
    # Pastikan panjangnya 64
    if mfccs.shape[0] < 64:
        mfccs = np.pad(mfccs, ((0, 64 - mfccs.shape[0]), (0, 0)), mode='constant')
    else:
        mfccs = mfccs[:64, :]
    
    # Duplikasi menjadi 3 channel
    mfccs = np.stack([mfccs] * 3, axis=-1)  # Shape: (64, 64, 3)
    
    return mfccs

# Fungsi untuk mengekstrak Melspectrogram dari file audio
def extract_melspectrogram(file_path, sr=22050, n_mels=64, n_fft=2048, hop_length=512):
    y, _ = librosa.load(file_path, sr=sr)
    mel_spectrogram = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, n_fft=n_fft, hop_length=hop_length)
    mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)  # Konversi ke dB
    
    # Pastikan panjangnya 64
    if mel_spectrogram.shape[1] < 64:
        mel_spectrogram = np.pad(mel_spectrogram, ((0, 0), (0, 64 - mel_spectrogram.shape[1])), mode='constant')
    else:
        mel_spectrogram = mel_spectrogram[:, :64]
    
    # Duplikasi menjadi 3 channel
    mel_spectrogram = np.stack([mel_spectrogram] * 3, axis=-1)  # Shape: (64, 64, 3)
    
    return mel_spectrogram

# Fungsi untuk mengklasifikasikan file audio menggunakan model
def classify_audio(file_path):
    # Ekstraksi fitur MFCC dan Melspec
    mfcc_features = extract_mfcc(file_path)
    mel_features = extract_melspectrogram(file_path)
    
    # Tambahkan batch dimension
    mfcc_features = np.expand_dims(mfcc_features, axis=0)  # Shape: (1, 64, 64, 3)
    mel_features = np.expand_dims(mel_features, axis=0)  # Shape: (1, 64, 64, 3)
    
    # Prediksi dengan model MFCC
    mfcc_predictions = mfcc_model.predict(mfcc_features)
    mfcc_predicted_class = np.argmax(mfcc_predictions, axis=1)
    mfcc_accuracy = np.max(mfcc_predictions)  # Prediksi dengan confidence tertinggi
    
    # Prediksi dengan model Melspec
    mel_predictions = melspec_model.predict(mel_features)
    mel_predicted_class = np.argmax(mel_predictions, axis=1)
    mel_accuracy = np.max(mel_predictions)  # Prediksi dengan confidence tertinggi
    
    return mfcc_predicted_class[0], mfcc_accuracy, mel_predicted_class[0], mel_accuracy

# Set Streamlit page configuration
st.set_page_config(page_title="Bird Song Classifier", page_icon="🦜", layout="centered")

# Title of the app
st.title("West Indonesia Birds Audio Classifier 🦜")

# Introduction
st.markdown("""
    **Selamat datang di aplikasi Klasifikasi Suara Burung!**
    Aplikasi ini akan mengklasifikasikan suara burung berdasarkan file audio yang diunggah.
    Cukup unggah file audio dalam format MP3 atau WAV, dan model akan memberikan prediksi kelas burung!
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

    # Predict using the classify_audio function
    if st.button('Prediksi Kelas Burung'):
        with st.spinner("Memproses..."):
            mfcc_predicted_class, mfcc_accuracy, mel_predicted_class, mel_accuracy = classify_audio(temp_file_path)
            
            st.subheader("Hasil Prediksi Model MFCC:")
            st.write(f"**Prediksi Kelas:** {mfcc_predicted_class}")
            st.write(f"**Akurasi Model:** {mfcc_accuracy:.2f}")
            
            st.subheader("Hasil Prediksi Model Melspec:")
            st.write(f"**Prediksi Kelas:** {mel_predicted_class}")
            st.write(f"**Akurasi Model:** {mel_accuracy:.2f}")

# Footer
st.markdown("""
    <hr>
    <p style="text-align:center; font-size:14px; color:#888; margin-top: 10px; margin-bottom: 10px;">
        Aplikasi Klasifikasi Suara Burung menggunakan Deep Learning | Dibuat oleh [Nama Anda]
    </p>
""", unsafe_allow_html=True)
