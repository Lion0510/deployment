import os
import json
import streamlit as st
from kaggle.api.kaggle_api_extended import KaggleApi
import tensorflow as tf
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt

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
            # Model sudah ada, tidak perlu mengunduh ulang
            return False  # Tidak ada tindakan unduh
        
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
            api.kernels_output(kernel_name, path=dest_folder, force=True)
        
        return True  # Model berhasil diunduh
        
    except Exception as e:
        st.error(f"Terjadi kesalahan saat mengunduh model: {str(e)}")
        return None  # Terjadi kesalahan

# URL kernel dan file output
kernel_name = "evanaryaputra28/dl-tb"  # Ganti dengan kernel Anda
output_files = ["cnn_melspec.h5", "cnn_mfcc.h5"]
dest_folder = "./models/"

# Periksa dan unduh model hanya jika tidak ada
download_status = download_model_from_kaggle(kernel_name, output_files, dest_folder)

# Tampilkan pesan hanya jika model benar-benar diunduh
if download_status is True:
    st.success("Model berhasil diunduh.")
elif download_status is None:
    st.error("Kesalahan terjadi saat mengunduh model.")

# Cek jika model berhasil diunduh dan memuatnya
melspec_model_save_path = os.path.join(dest_folder, 'cnn_melspec.h5')
mfcc_model_save_path = os.path.join(dest_folder, 'cnn_mfcc.h5')

# Memuat model Melspec jika file ada
if os.path.exists(melspec_model_save_path):
    try:
        melspec_model = tf.keras.models.load_model(melspec_model_save_path)
    except Exception as e:
        st.error(f"Gagal memuat model Melspec: {str(e)}")

# Memuat model MFCC jika file ada
if os.path.exists(mfcc_model_save_path):
    try:
        mfcc_model = tf.keras.models.load_model(mfcc_model_save_path)
    except Exception as e:
        st.error(f"Gagal memuat model MFCC: {str(e)}")

# Fungsi untuk menampilkan gambar spektrum
def plot_spectrogram(data, sr, title, y_axis, x_axis):
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(data, sr=sr, x_axis=x_axis, y_axis=y_axis, cmap='magma')
    plt.colorbar(format='%+2.0f dB')
    plt.title(title)
    plt.tight_layout()
    st.pyplot(plt)  # Menampilkan plot di Streamlit
    plt.close()

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
                # Ekstraksi MFCC dan Melspectrogram
                y, sr = librosa.load(temp_file_path, sr=None)
                mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
                melspec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=64)

                # Plot MFCC sebagai spektrum
                st.subheader("Spektrum MFCC")
                plot_spectrogram(mfcc, sr, "MFCC", y_axis='mel', x_axis='time')

                # Plot Melspectrogram
                st.subheader("Spektrum Melspectrogram")
                melspec_db = librosa.power_to_db(melspec, ref=np.max)
                plot_spectrogram(melspec_db, sr, "Melspectrogram", y_axis='mel', x_axis='time')

                # Lakukan prediksi dengan kedua model
                mfcc_image = np.expand_dims(mfcc, axis=0)  # Dummy preprocessing
                melspec_image = np.expand_dims(melspec, axis=0)  # Dummy preprocessing

                mfcc_result = mfcc_model.predict(mfcc_image)  # Prediksi model MFCC
                melspec_result = melspec_model.predict(melspec_image)  # Prediksi model Melspec

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
