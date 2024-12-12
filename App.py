import streamlit as st
import tensorflow as tf
import librosa
import numpy as np
import os
import cv2  # Untuk penyesuaian ukuran gambar
from sklearn.preprocessing import LabelEncoder
from keras.preprocessing import image

# Set Streamlit page configuration
st.set_page_config(page_title="Bird Song Classifier", page_icon="🦜", layout="centered")

# Fungsi untuk memuat model MFCC
def load_mfcc_model():
    model_path = '/kaggle/working/cnn_mfcc.h5'  # Ganti dengan path model MFCC Anda
    if os.path.exists(model_path):
        return tf.keras.models.load_model(model_path)
    else:
        st.error("Model MFCC tidak ditemukan.")
        return None

# Load model MFCC
model_mfcc = load_mfcc_model()

# Fungsi untuk mengonversi audio ke MFCC dan kemudian ke gambar 2D
def audio_to_mfcc_image(file_path):
    # Menggunakan librosa untuk memuat dan mengekstrak MFCC
    y, sr = librosa.load(file_path, sr=None)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)  # Extract MFCC (13 coefficients)
    
    # Reshape MFCC ke (64, 64) dan tambahkan dimensi channel (reshape untuk gambar 2D)
    mfcc_resized = cv2.resize(mfcc, (64, 64))  # Resize MFCC menjadi 64x64
    mfcc_resized = np.expand_dims(mfcc_resized, axis=-1)  # Menambah dimensi channel (grayscale)
    
    # Jika model mengharapkan 3 channel, kita bisa duplikasi channel menjadi RGB
    mfcc_resized = np.repeat(mfcc_resized, 3, axis=-1)  # Menjadi 64x64x3
    
    return mfcc_resized

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

# Fungsi untuk melakukan prediksi
def predict_bird_species(file_path):
    # Mengonversi audio ke gambar MFCC
    mfcc_image = audio_to_mfcc_image(file_path)
    
    # Melakukan prediksi menggunakan model MFCC
    mfcc_image = np.expand_dims(mfcc_image, axis=0)  # Menambahkan batch dimension
    prediction = model_mfcc.predict(mfcc_image)
    predicted_class = np.argmax(prediction, axis=1)
    accuracy = np.max(prediction)
    
    return predicted_class[0], accuracy

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
                # Lakukan prediksi
                predicted_class, accuracy = predict_bird_species(temp_file_path)
                
                # Tampilkan hasil prediksi
                st.subheader("Hasil Prediksi:")
                st.write(f"**Prediksi kelas burung:** {predicted_class} dengan akurasi {accuracy * 100:.2f}%")
                
            except Exception as e:
                st.error(f"Error saat melakukan prediksi: {str(e)}")

# Footer
st.markdown("""
    <hr>
    <p style="text-align:center; font-size:14px; color:#888; margin-top: 10px; margin-bottom: 10px;">
        Aplikasi Klasifikasi Suara Burung menggunakan Deep Learning | Dibuat oleh Kelompok 11
    </p>
""", unsafe_allow_html=True)
