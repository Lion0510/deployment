import streamlit as st
import numpy as np
import tensorflow as tf
import gdown  # Import gdown for downloading files from Google Drive
import os

# Set Streamlit page configuration
st.set_page_config(page_title="Bird Song Classifier", page_icon="🦜", layout="centered")

# Function to download model from Google Drive using gdown
def download_model_from_google_drive(url, output_path):
    try:
        gdown.download(url, output_path, quiet=False)
        st.write(f"Model berhasil diunduh ke: {output_path}")
    except Exception as e:
        st.error(f"Error saat mengunduh model: {str(e)}")

# Google Drive model URLs
melspec_model_url = 'https://drive.google.com/uc?id=1--BTVqDAoyy83_3GEqL93SveSNYdncB_'
mfcc_model_url = 'https://drive.google.com/uc?id=1rHo_GkTxFp5lDsNcFphEVV__dEHBciZa'


# Path to save the downloaded models
melspec_model_save_path = 'melspec_model.h5'
mfcc_model_save_path = 'mfcc_model.h5'

# Download the models from Google Drive
st.write("Mengunduh model Melspec dari Google Drive...")
download_model_from_google_drive(melspec_model_url, melspec_model_save_path)

st.write("Mengunduh model MFCC dari Google Drive...")
download_model_from_google_drive(mfcc_model_url, mfcc_model_save_path)

# Check if the models exist and their file size
if os.path.exists(melspec_model_save_path):
    st.write(f"Model Melspec ditemukan. Ukuran file: {os.path.getsize(melspec_model_save_path)} bytes")
else:
    st.error("Model Melspec tidak ditemukan. Pastikan file berhasil diunduh.")

if os.path.exists(mfcc_model_save_path):
    st.write(f"Model MFCC ditemukan. Ukuran file: {os.path.getsize(mfcc_model_save_path)} bytes")
else:
    st.error("Model MFCC tidak ditemukan. Pastikan file berhasil diunduh.")

# Load the pre-trained models
try:
    melspec_model = tf.keras.models.load_model(melspec_model_save_path)
    st.write("Model Melspec berhasil dimuat.")
except Exception as e:
    st.error(f"Error saat memuat model Melspec: {str(e)}")

try:
    mfcc_model = tf.keras.models.load_model(mfcc_model_save_path)
    st.write("Model MFCC berhasil dimuat.")
except Exception as e:
    st.error(f"Error saat memuat model MFCC: {str(e)}")

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

# Dummy Prediction Function (replace with actual implementation later)
def dummy_predict(file_path):
    # This function is for testing purposes. Replace this with your actual predict function.
    return {"mfcc": {"class": 2, "accuracy": 0.85}, "melspec": {"class": 3, "accuracy": 0.90}}

if uploaded_audio is not None:
    # Display file details
    st.audio(uploaded_audio, format="audio/mp3")
    
    # Save the uploaded audio to a temporary file
    temp_file_path = "temp_audio.wav"
    with open(temp_file_path, "wb") as f:
        f.write(uploaded_audio.read())

    # Predict using the models
    if st.button("Prediksi Kelas Burung"):
        with st.spinner("Memproses..."):
            try:
                # Perform predictions (replace this with actual model inference)
                results = dummy_predict(temp_file_path)
                mfcc_result = results["mfcc"]
                melspec_result = results["melspec"]

                st.subheader("Hasil Prediksi:")
                st.write(f"**Model MFCC:** Prediksi kelas {mfcc_result['class']} dengan akurasi {mfcc_result['accuracy'] * 100:.2f}%")
                st.write(f"**Model Melspec:** Prediksi kelas {melspec_result['class']} dengan akurasi {melspec_result['accuracy'] * 100:.2f}%")
            except Exception as e:
                st.error(f"Error saat melakukan prediksi: {str(e)}")

# Footer
st.markdown("""
    <hr>
    <p style="text-align:center; font-size:14px; color:#888; margin-top: 10px; margin-bottom: 10px;">
        Aplikasi Klasifikasi Suara Burung menggunakan Deep Learning | Dibuat oleh Kelompok 11
    </p>
""", unsafe_allow_html=True)
