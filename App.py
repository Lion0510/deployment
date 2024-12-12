import os
import json
import streamlit as st
from kaggle.api.kaggle_api_extended import KaggleApi
import tensorflow as tf

# Fungsi untuk mengunduh model dari Kaggle API
def download_model_from_kaggle(kernel_name, output_files, dest_folder):
    try:
        # Cek jika model sudah ada di folder tujuan
        if all(os.path.exists(os.path.join(dest_folder, file)) for file in output_files):
            st.success("Model sudah tersedia. Tidak perlu mengunduh lagi.")
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

# Periksa dan unduh model hanya jika tidak ada
download_model_from_kaggle(kernel_name, output_files, dest_folder)

# Cek jika model berhasil diunduh dan memuatnya
melspec_model_save_path = os.path.join(dest_folder, 'cnn_melspec.h5')
mfcc_model_save_path = os.path.join(dest_folder, 'cnn_mfcc.h5')

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
