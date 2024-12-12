import os
import json
import streamlit as st
from kaggle.api.kaggle_api_extended import KaggleApi
import tensorflow as tf
import librosa
import numpy as np

# Set Streamlit page configuration
st.set_page_config(page_title="Bird Song Classifier", page_icon="🦜", layout="centered")

# Fungsi untuk mengunduh model dari Kaggle API
def download_model_from_kaggle(kernel_name, output_files, dest_folder):
    try:
        # Mengakses API key dari Streamlit Secrets
        kaggle_username = st.secrets["kaggle"]["KAGGLE_USERNAME"]
        kaggle_key = st.secrets["kaggle"]["KAGGLE_KEY"]

        # Path file kaggle.json
        kaggle_json_path = os.path.expanduser("~/.kaggle/kaggle.json")

        # Membuat folder ~/.kaggle jika belum ada
        os.makedirs(os.path.dirname(kaggle_json_path), exist_ok=True)

        # Menyimpan kredensial API Kaggle ke file kaggle.json
        with open(kaggle_json_path, 'w') as f:
            json.dump({"username": kaggle_username, "key": kaggle_key}, f)

        # Menampilkan status ke pengguna
        st.success("API Key Kaggle berhasil disalin ke ~/.kaggle/kaggle.json")

        # Autentikasi ke Kaggle API
        api = KaggleApi()
        api.authenticate()

        # Membuat folder tujuan untuk menyimpan model yang diunduh
        os.makedirs(dest_folder, exist_ok=True)

        # Mengunduh output dari kernel
        for output_file in output_files:
            st.text(f"Mengunduh {output_file} dari kernel {kernel_name}...")
            api.kernels_output(kernel_name, path=dest_folder, force=True)
            st.success(f"{output_file} berhasil diunduh ke {dest_folder}")
            
    except Exception as e:
        st.error(f"Terjadi kesalahan saat mengunduh model: {str(e)}")

# URL kernel dan file output
kernel_name = "evanaryaputra28/dl-tb"
output_files = ["cnn_melspec.h5", "cnn_mfcc.h5"]
dest_folder = "./models/"

# Download model dari Kaggle
download_model_from_kaggle(kernel_name, output_files, dest_folder)

# Cek jika model berhasil diunduh dan memuatnya
melspec_model_save_path = os.path.join(dest_folder, 'cnn_melspec.h5')
mfcc_model_save_path = os.path.join(dest_folder, 'cnn_mfcc.h5')

# Memuat model jika file ada
if os.path.exists(melspec_model_save_path):
    try:
        melspec_model = tf.keras.models.load_model(melspec_model_save_path)
        st.success("Model Melspec berhasil dimuat!")
    except Exception as e:
        st.error(f"Gagal memuat model Melspec: {str(e)}")

if os.path.exists(mfcc_model_save_path):
    try:
        mfcc_model = tf.keras.models.load_model(mfcc_model_save_path)
        st.success("Model MFCC berhasil dimuat!")
    except Exception as e:
        st.error(f"Gagal memuat model MFCC: {str(e)}")

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

# Fungsi untuk ekstraksi fitur MFCC dan Melspectrogram
def extract_features(audio_path):
    # Load audio file
    y, sr = librosa.load(audio_path, sr=None)
    
    # Ekstraksi MFCC
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc = np.mean(mfcc.T, axis=0)  # Ambil rata-rata dari MFCC untuk fitur

    # Ekstraksi Melspectrogram
    melspec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    melspec = np.mean(melspec.T, axis=0)  # Ambil rata-rata dari Melspectrogram untuk fitur

    return mfcc, melspec

# Prediksi menggunakan model yang sudah dimuat
def predict_bird_class(model, features):
    prediction = model.predict(np.expand_dims(features, axis=0))  # Tambahkan dimensi batch
    predicted_class = np.argmax(prediction, axis=1)[0]  # Ambil kelas dengan nilai tertinggi
    accuracy = np.max(prediction)  # Akurasi model
    return predicted_class, accuracy

# Proses audio yang diunggah
if uploaded_audio is not None:
    st.audio(uploaded_audio, format="audio/mp3")
    
    # Simpan file audio sementara
    temp_file_path = "temp_audio.wav"
    with open(temp_file_path, "wb") as f:
        f.write(uploaded_audio.read())

    # Ekstraksi fitur audio
    mfcc_features, melspec_features = extract_features(temp_file_path)

    # Prediksi dengan model jika tombol ditekan
    if st.button("Prediksi Kelas Burung"):
        with st.spinner("Memproses..."):
            try:
                # Prediksi dengan model MFCC
                mfcc_class, mfcc_accuracy = predict_bird_class(mfcc_model, mfcc_features)
                # Prediksi dengan model Melspec
                melspec_class, melspec_accuracy = predict_bird_class(melspec_model, melspec_features)

                # Tampilkan hasil prediksi
                st.subheader("Hasil Prediksi:")
                st.write(f"**Model MFCC:** Prediksi kelas {mfcc_class} dengan akurasi {mfcc_accuracy * 100:.2f}%")
                st.write(f"**Model Melspec:** Prediksi kelas {melspec_class} dengan akurasi {melspec_accuracy * 100:.2f}%")
            except Exception as e:
                st.error(f"Error saat melakukan prediksi: {str(e)}")

# Footer
st.markdown("""
    <hr>
    <p style="text-align:center; font-size:14px; color:#888; margin-top: 10px; margin-bottom: 10px;">
        Aplikasi Klasifikasi Suara Burung menggunakan Deep Learning | Dibuat oleh Kelompok 11
    </p>
""", unsafe_allow_html=True)
