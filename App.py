import streamlit as st
import tensorflow as tf
import librosa
import numpy as np
from PIL import Image
import os

# Set Streamlit page configuration
st.set_page_config(page_title="Bird Song Classifier", page_icon="🦜", layout="centered")

# Fungsi untuk mengunduh dan memuat model
def load_model(model_path):
    try:
        return tf.keras.models.load_model(model_path)
    except Exception as e:
        st.error(f"Error saat memuat model: {str(e)}")
        return None

# Path model yang sudah disimpan (sesuaikan dengan lokasi model Anda)
melspec_model_path = 'cnn_melspec.h5'
mfcc_model_path = 'cnn_mfcc.h5'

# Memuat model
melspec_model = load_model(melspec_model_path)
mfcc_model = load_model(mfcc_model_path)

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

# Fungsi untuk ekstraksi MFCC dan resize gambar menggunakan Pillow
def audio_to_mfcc_image(file_path):
    y, sr = librosa.load(file_path, sr=None)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)  # Ekstraksi MFCC (13 koefisien)

    # Resize MFCC menjadi (64, 64) menggunakan Pillow
    mfcc_resized = Image.fromarray(mfcc)
    mfcc_resized = mfcc_resized.resize((64, 64))  # Ubah ukuran menjadi 64x64
    mfcc_resized = np.array(mfcc_resized)  # Convert kembali ke numpy array

    # Tambahkan dimensi channel (grayscale ke RGB)
    mfcc_resized = np.expand_dims(mfcc_resized, axis=-1)  # Menambah dimensi channel (grayscale)
    mfcc_resized = np.repeat(mfcc_resized, 3, axis=-1)  # Ubah menjadi 64x64x3 (RGB)

    return mfcc_resized

# Fungsi untuk ekstraksi Melspectrogram dan resize gambar menggunakan Pillow
def audio_to_melspec_image(file_path):
    y, sr = librosa.load(file_path, sr=None)
    melspec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=64)  # Ekstraksi Mel Spectrogram

    # Resize Melspectrogram menjadi (64, 64) menggunakan Pillow
    melspec_resized = Image.fromarray(melspec)
    melspec_resized = melspec_resized.resize((64, 64))  # Ubah ukuran menjadi 64x64
    melspec_resized = np.array(melspec_resized)  # Convert kembali ke numpy array

    # Tambahkan dimensi channel (grayscale ke RGB)
    melspec_resized = np.expand_dims(melspec_resized, axis=-1)  # Menambah dimensi channel (grayscale)
    melspec_resized = np.repeat(melspec_resized, 3, axis=-1)  # Ubah menjadi 64x64x3 (RGB)

    return melspec_resized

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
                # Ekstraksi fitur untuk MFCC dan Melspectrogram
                mfcc_image = audio_to_mfcc_image(temp_file_path)
                melspec_image = audio_to_melspec_image(temp_file_path)

                # Lakukan prediksi dengan kedua model
                mfcc_result = mfcc_model.predict(np.expand_dims(mfcc_image, axis=0))  # Prediksi model MFCC
                melspec_result = melspec_model.predict(np.expand_dims(melspec_image, axis=0))  # Prediksi model Melspec

                # Ambil prediksi kelas dan akurasi
                mfcc_pred_class = np.argmax(mfcc_result, axis=1)[0]
                melspec_pred_class = np.argmax(melspec_result, axis=1)[0]
                mfcc_accuracy = np.max(mfcc_result)
                melspec_accuracy = np.max(melspec_result)

                # Tampilkan hasil prediksi
                st.subheader("Hasil Prediksi:")
                st.write(f"**Model MFCC:** Prediksi kelas {mfcc_pred_class} dengan akurasi {mfcc_accuracy * 100:.2f}%")
                st.write(f"**Model Melspec:** Prediksi kelas {melspec_pred_class} dengan akurasi {melspec_accuracy * 100:.2f}%")
            except Exception as e:
                st.error(f"Error saat melakukan prediksi: {str(e)}")

# Footer
st.markdown("""
    <hr>
    <p style="text-align:center; font-size:14px; color:#888; margin-top: 10px; margin-bottom: 10px;">
        Aplikasi Klasifikasi Suara Burung menggunakan Deep Learning | Dibuat oleh Kelompok 11
    </p>
""", unsafe_allow_html=True)
