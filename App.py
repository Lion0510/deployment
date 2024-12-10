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

gdown.download(melspec_model_url, melspec_model_path, quiet=True)
gdown.download(mfcc_model_url, mfcc_model_path, quiet=True)

# Load kedua model
melspec_model = load_model(melspec_model_path)
mfcc_model = load_model(mfcc_model_path)

# Fungsi untuk mengekstrak MFCC dari file audio
def extract_mfcc(file_path, sr=22050, n_mfcc=64, n_fft=2048, hop_length=512):
    y, _ = librosa.load(file_path, sr=sr)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length)
    mfccs = mfccs.T  # Transpose untuk mendapatkan shape (time, n_mfcc)
    if mfccs.shape[0] < 64:
        mfccs = np.pad(mfccs, ((0, 64 - mfccs.shape[0]), (0, 0)), mode='constant')
    else:
        mfccs = mfccs[:64, :]
    mfccs = np.stack([mfccs] * 3, axis=-1)  # Mengubah menjadi 3 channel (64, 64, 3)
    return mfccs

# Fungsi untuk mengekstrak Melspectrogram dari file audio
def extract_melspec(file_path, sr=22050, n_mels=128, n_fft=2048, hop_length=512):
    y, _ = librosa.load(file_path, sr=sr)
    melspec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, n_fft=n_fft, hop_length=hop_length)
    melspec = librosa.power_to_db(melspec, ref=np.max)  # Konversi ke dB
    melspec = melspec.T  # Transpose agar shape menjadi (time, n_mels)
    
    # Pastikan shape adalah (64, 64) sebelum padding
    if melspec.shape[0] < 64:
        melspec = np.pad(melspec, ((0, 64 - melspec.shape[0]), (0, 0)), mode='constant')
    else:
        melspec = melspec[:64, :]
    
    # Pastikan shape melspec adalah (64, 64, 3) untuk input model
    melspec = np.stack([melspec] * 3, axis=-1)  # Membuat 3 channel (64, 64, 3)
    
    return melspec


# Fungsi untuk klasifikasi menggunakan model
def classify_audio(file_path):
    mfcc_features = extract_mfcc(file_path)
    melspec_features = extract_melspec(file_path)
    
    mfcc_features = np.expand_dims(mfcc_features, axis=0)  # Menambahkan dimensi batch
    melspec_features = np.expand_dims(melspec_features, axis=0)  # Menambahkan dimensi batch
    
    # Prediksi menggunakan kedua model
    mfcc_pred = mfcc_model.predict(mfcc_features)
    melspec_pred = melspec_model.predict(melspec_features)
    
    # Mendapatkan hasil prediksi dan akurasi
    mfcc_pred_class = np.argmax(mfcc_pred, axis=1)
    melspec_pred_class = np.argmax(melspec_pred, axis=1)
    
    mfcc_accuracy = np.max(mfcc_pred)  # Prediksi untuk model MFCC
    melspec_accuracy = np.max(melspec_pred)  # Prediksi untuk model Melspec
    
    return mfcc_pred_class[0], mfcc_accuracy, melspec_pred_class[0], melspec_accuracy

# Set konfigurasi halaman Streamlit
st.set_page_config(page_title="Deep Learning in Audio - Klasifikasi Suara Burung", page_icon="🦜", layout="centered")

# Judul Aplikasi
st.title("Deep Learning in Audio: Klasifikasi Suara Burung di Indonesia Bagian Barat 🦜")

# Deskripsi
st.markdown("""
    **Selamat datang di aplikasi klasifikasi suara burung menggunakan Deep Learning!**
    Aplikasi ini mengimplementasikan teknik ekstraksi fitur audio menggunakan MFCC dan Melspectrogram, 
    serta menggunakan dua model CNN yang berbeda untuk mengklasifikasikan suara burung yang ada di Indonesia Bagian Barat.
    Unggah file audio dalam format MP3 atau WAV, dan model akan memberikan prediksi kelas burung beserta akurasi dari kedua model.
""")

# Upload file audio
st.header("Unggah File Audio Suara Burung")

uploaded_audio = st.file_uploader("Pilih file audio (MP3/WAV) untuk diuji", type=["mp3", "wav"])

if uploaded_audio is not None:
    # Menampilkan file audio
    st.audio(uploaded_audio, format="audio/mp3")
    
    # Menyimpan file audio yang diunggah ke file sementara
    audio_bytes = uploaded_audio.read()
    with BytesIO(audio_bytes) as audio_buffer:
        temp_file_path = 'temp_audio.wav'
        with open(temp_file_path, 'wb') as f:
            f.write(audio_buffer.getbuffer())
    
    # Button untuk prediksi
    if st.button('Prediksi Kelas Burung'):
        with st.spinner("Memproses..."):
            # Mengklasifikasikan menggunakan kedua model
            mfcc_class, mfcc_acc, melspec_class, melspec_acc = classify_audio(temp_file_path)
            
            # Menampilkan hasil prediksi
            st.subheader("Hasil Prediksi:")
            
            st.write(f"**Prediksi Kelas (Model MFCC):** {mfcc_class} dengan Akurasi: {mfcc_acc*100:.2f}%")
            st.write(f"**Prediksi Kelas (Model Melspec):** {melspec_class} dengan Akurasi: {melspec_acc*100:.2f}%")
            
            # Menampilkan visualisasi akurasi
            st.subheader("Visualisasi Akurasi:")
            fig, ax = plt.subplots()
            ax.bar(["MFCC Model", "Melspec Model"], [mfcc_acc, melspec_acc], color=["blue", "orange"])
            ax.set_ylim(0, 1)
            ax.set_ylabel("Akurasi")
            ax.set_title("Perbandingan Akurasi antara Model MFCC dan Melspec")
            st.pyplot(fig)

# Footer
st.markdown("""
    <hr>
    <p style="text-align:center; font-size:12px; color:#888;">Aplikasi Klasifikasi Suara Burung menggunakan Deep Learning - Implementasi Variasi Teknik Ekstraksi Fitur</p>
""", unsafe_allow_html=True)
