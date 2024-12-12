import streamlit as st
import gdown
import os
import tensorflow as tf
import librosa
import numpy as np

# Set Streamlit page configuration
st.set_page_config(page_title="Bird Song Classifier", page_icon="🦜", layout="centered")

# Fungsi untuk mengunduh model dari Google Drive
def download_model_from_google_drive(url, output_path):
    try:
        gdown.download(url, output_path, quiet=True)  # quiet=True untuk menyembunyikan output

# Google Drive model URLs (ganti dengan URL model Anda)
melspec_model_url = 'gdown https://drive.google.com/uc?id=1ebsCcP4GxY_X6VZTZhMIsvvaVKWKhSPq'  # Ganti dengan URL model Anda
mfcc_model_url = 'gdown https://drive.google.com/uc?id=1hBPcwqyEFIvx1-2nHNKrpFTC2DmiuF4L'  # Ganti dengan URL model Anda

# Path untuk menyimpan model yang diunduh
melspec_model_save_path = 'melspec_model.h5'
mfcc_model_save_path = 'mfcc_model.h5'

# Download model dari Google Drive
download_model_from_google_drive(melspec_model_url, melspec_model_save_path)
download_model_from_google_drive(mfcc_model_url, mfcc_model_save_path)

# Check jika model berhasil diunduh dan memuatnya
if os.path.exists(melspec_model_save_path):
    try:
        melspec_model = tf.keras.models.load_model(melspec_model_save_path)

if os.path.exists(mfcc_model_save_path):
    try:
        mfcc_model = tf.keras.models.load_model(mfcc_model_save_path)

# Title aplikasi Streamlit
st.title("Deep Learning in Audio: Klasifikasi Suara Burung di Indonesia Bagian Barat 🦜")

# Deskripsi aplikasi
st.markdown("""
    **Selamat datang di aplikasi klasifikasi suara burung menggunakan Deep Learning!**
    Aplikasi ini mengimplementasikan teknik ekstraksi fitur audio menggunakan MFCC dan Melspectrogram,
    serta menggunakan dua model CNN yang berbeda untuk mengklasifikasikan suara burung yang ada di Indonesia Bagian Barat.
    Unggah file audio dalam format MP3 atau WAV, dan model akan memberikan prediksi kelas burung beserta akurasi dari kedua model.
""")

# Upload file audio
st.header("Unggah File Audio Suara Burung")
uploaded_audio = st.file_uploader("Pilih file audio (MP3/WAV) untuk diuji", type=["mp3", "wav"])

# Fungsi dummy prediksi (ganti dengan fungsi prediksi yang sebenarnya)
def dummy_predict(file_path):
    # Fungsi dummy untuk prediksi
    # Ganti dengan implementasi yang sesuai untuk model Anda
    return {"mfcc": {"class": 2, "accuracy": 0.85}, "melspec": {"class": 3, "accuracy": 0.90}}

# Proses audio yang diunggah
if uploaded_audio is not None:
    st.audio(uploaded_audio, format="audio/mp3")
    
    # Simpan file audio sementara
    temp_file_path = "temp_audio.wav"
    with open(temp_file_path, "wb") as f:
        f.write(uploaded_audio.read())

    # Prediksi dengan model jika tombol ditekan
    if st.button("Prediksi Kelas Burung"):
        with st.spinner("Memproses..."):
            try:
                # Lakukan prediksi (ganti dengan implementasi prediksi asli menggunakan model)
                results = dummy_predict(temp_file_path)
                mfcc_result = results["mfcc"]
                melspec_result = results["melspec"]

                # Tampilkan hasil prediksi
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
