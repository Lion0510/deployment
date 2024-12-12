import os
import json
import streamlit as st
from kaggle.api.kaggle_api_extended import KaggleApi
import tensorflow as tf

# Fungsi untuk mengunduh model dari Kaggle API
def download_model_from_kaggle(kernel_name, output_files, dest_folder):
    try:
        # Periksa apakah model sudah ada di folder tujuan
        model_files_exist = True
        for file in output_files:
            if not os.path.exists(os.path.join(dest_folder, file)):
                model_files_exist = False
                break
        
        if model_files_exist:
            st.success("Model sudah ada, tidak perlu mengunduh ulang.")
            return
        
        # Mengakses API key dari Streamlit Secrets
        kaggle_username = st.secrets["kaggle"]["KAGGLE_USERNAME"]
        kaggle_key = st.secrets["kaggle"]["KAGGLE_KEY"]

        # Menyimpan kredensial API Kaggle ke file kaggle.json
        kaggle_json_path = os.path.expanduser("~/.kaggle/kaggle.json")
        os.makedirs(os.path.dirname(kaggle_json_path), exist_ok=True)

        with open(kaggle_json_path, 'w') as f:
            json.dump({"username": kaggle_username, "key": kaggle_key}, f)

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
kernel_name = "evanaryaputra28/dl-tb"  # Ganti dengan kernel Anda
output_files = ["cnn_melspec.h5", "cnn_mfcc.h5"]
dest_folder = "./models/"

# Cek apakah model sudah ada, jika belum, unduh model
download_model_from_kaggle(kernel_name, output_files, dest_folder)

# Cek jika model berhasil diunduh dan memuatnya
melspec_model_save_path = os.path.join(dest_folder, 'cnn_melspec.h5')
mfcc_model_save_path = os.path.join(dest_folder, 'cnn_mfcc.h5')

# Memuat model Melspec jika file ada
if os.path.exists(melspec_model_save_path):
    try:
        melspec_model = tf.keras.models.load_model(melspec_model_save_path)
        st.success("Model Melspec berhasil dimuat!")
    except Exception as e:
        st.error(f"Gagal memuat model Melspec: {str(e)}")

# Memuat model MFCC jika file ada
if os.path.exists(mfcc_model_save_path):
    try:
        mfcc_model = tf.keras.models.load_model(mfcc_model_save_path)
        st.success("Model MFCC berhasil dimuat!")
    except Exception as e:
        st.error(f"Gagal memuat model MFCC: {str(e)}")

# Fungsi untuk mengonversi audio ke MFCC dan kemudian ke gambar 2D
def audio_to_mfcc_image(file_path):
    import librosa
    import numpy as np
    import cv2  # Untuk penyesuaian ukuran gambar

    # Menggunakan librosa untuk memuat dan mengekstrak MFCC
    y, sr = librosa.load(file_path, sr=None)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)  # Extract MFCC (13 coefficients)
    
    # Reshape MFCC ke (64, 64) dan tambahkan dimensi channel (reshape untuk gambar 2D)
    mfcc_resized = cv2.resize(mfcc, (64, 64))  # Resize MFCC menjadi 64x64
    mfcc_resized = np.expand_dims(mfcc_resized, axis=-1)  # Menambah dimensi channel (grayscale)
    
    # Jika model mengharapkan 3 channel, kita bisa duplikasi channel menjadi RGB
    mfcc_resized = np.repeat(mfcc_resized, 3, axis=-1)  # Menjadi 64x64x3
    
    return mfcc_resized

# Fungsi untuk ekstraksi Melspectrogram dan resize gambar
def audio_to_melspec_image(file_path):
    import librosa
    import numpy as np
    import cv2  # Untuk penyesuaian ukuran gambar

    # Menggunakan librosa untuk memuat dan mengekstrak Melspectrogram
    y, sr = librosa.load(file_path, sr=None)
    melspec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=64)  # Ekstraksi Mel Spectrogram

    # Resize Melspectrogram menjadi (64, 64)
    melspec_resized = cv2.resize(melspec, (64, 64))  # Resize Melspectrogram menjadi 64x64
    melspec_resized = np.expand_dims(melspec_resized, axis=-1)  # Menambah dimensi channel (grayscale)
    melspec_resized = np.repeat(melspec_resized, 3, axis=-1)  # Menjadi 64x64x3
    
    return melspec_resized

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
