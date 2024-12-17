import os
import json
import streamlit as st
import tensorflow as tf
import librosa
import librosa.display
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from kaggle.api.kaggle_api_extended import KaggleApi

# Menyembunyikan log TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# CSS untuk styling halaman
st.markdown("""
    <style>
        .stApp {
            background-image: url('https://raw.githubusercontent.com/Lion0510/deployment/main/images/bg.jpg');
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
        }
        .header-content {
            display: flex;
            justify-content: center;
            align-items: center;
            gap: 20px;
            margin-top: 10px;
        }
        .logo {
            background-color: white;
            padding: 10px;
            border-radius: 10px;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
            display: inline-block;
            width: 100px;
            height: auto;
        }
        .header-box {
            background-color: rgba(0, 0, 0, 0.6);
            color: white;
            padding: 15px 20px;
            border-radius: 10px;
            text-align: center;
            margin: 20px auto;
            display: inline-block;
            box-shadow: 0px 2px 5px rgba(0, 0, 0, 0.3);
        }
        .content-section {
            background-color: rgba(0, 0, 0, 0.6);
            color: white;
            padding: 20px;
            border-radius: 10px;
            margin: 20px auto;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.3);
        }
        .footer {
            background-color: rgba(0, 0, 0, 0.8);
            color: white;
            text-align: center;
            padding: 10px;
            border-radius: 10px;
            margin-top: 20px;
        }
    </style>
""", unsafe_allow_html=True)

# Fungsi mengunduh model dari Kaggle API
def download_model_from_kaggle(kernel_name, output_files, dest_folder):
    try:
        # Cek apakah model sudah ada
        os.makedirs(dest_folder, exist_ok=True)
        model_files_exist = all([os.path.exists(os.path.join(dest_folder, file)) for file in output_files])
        if model_files_exist:
            st.info("Model sudah tersedia. Tidak perlu mengunduh ulang.")
            return

        # Inisialisasi Kaggle API
        kaggle_username = st.secrets["kaggle"]["KAGGLE_USERNAME"]
        kaggle_key = st.secrets["kaggle"]["KAGGLE_KEY"]
        
        kaggle_json_path = os.path.expanduser("~/.kaggle/kaggle.json")
        os.makedirs(os.path.dirname(kaggle_json_path), exist_ok=True)

        with open(kaggle_json_path, 'w') as f:
            json.dump({"username": kaggle_username, "key": kaggle_key}, f)
        os.chmod(kaggle_json_path, 600)

        api = KaggleApi()
        api.authenticate()

        # Unduh model dari output kernel
        for output_file in output_files:
            st.info(f"Mengunduh model {output_file}...")
            api.kernels_output(kernel_name, path=dest_folder, force=True)
        st.success("Model berhasil diunduh!")

    except Exception as e:
        st.error(f"Kesalahan saat mengunduh model: {str(e)}")

# Unduh model jika belum tersedia
models_dir = "./models/"
output_files = ["cnn_mfcc.h5", "cnn_melspec.h5"]
download_model_from_kaggle("evanaryaputra28/tubes-dll", output_files, models_dir)

# Memuat model
mfcc_model_path = os.path.join(models_dir, "cnn_mfcc.h5")
melspec_model_path = os.path.join(models_dir, "cnn_melspec.h5")

mfcc_model = tf.keras.models.load_model(mfcc_model_path)
melspec_model = tf.keras.models.load_model(melspec_model_path)

# Fungsi pemrosesan MFCC
def preprocess_mfcc(mfcc):
    mfcc_image = Image.fromarray(mfcc)
    mfcc_image = mfcc_image.resize((64, 64))
    mfcc_resized = np.array(mfcc_image)
    mfcc_resized = np.expand_dims(mfcc_resized, axis=-1)
    mfcc_resized = np.repeat(mfcc_resized, 3, axis=-1)
    return mfcc_resized

# Fungsi pemrosesan MelSpectrogram
def preprocess_melspec(melspec):
    melspec_db = librosa.power_to_db(melspec, ref=np.max)
    melspec_image = Image.fromarray(melspec_db)
    melspec_image = melspec_image.resize((64, 64))
    melspec_resized = np.array(melspec_image)
    melspec_resized = np.expand_dims(melspec_resized, axis=-1)
    melspec_resized = np.repeat(melspec_resized, 3, axis=-1)
    return melspec_resized

# Header
st.markdown("""
<div class="header-content">
    <img src="https://raw.githubusercontent.com/Lion0510/deployment/main/images/Logo2.jpg" alt="Logo Fakultas Sains" class="logo">
    <img src="https://blogger.googleusercontent.com/img/b/R29vZ2xl/AVvXsEhpSH0B8r5lSPmWBfANSG_LjlIEx2q0rEMXqQLxzr5Ggr7dSi7jfn7ALTDRPGrbUVkhgevNViaXgZokaU0_wwNme660o667wS7T_l4SzhKbQi50g2gLlVXsUNJBSbgOQ7nXi_hzfTDkv0yX/s320/logo+itera+oke.png" alt="Logo ITERA" class="logo">
    <img src="https://pbs.twimg.com/profile_images/1272461269136576512/Uw9AShxq_400x400.jpg" alt="Logo Fakultas Teknologi" class="logo">
</div>
<div class="header-box">
    <h1>Klasifikasi Suara Burung Sumatera 🦜</h1>
    <p>Identifikasi Burung Berdasarkan Suara Secara Otomatis</p>
</div>
""", unsafe_allow_html=True)

# Upload audio
uploaded_audio = st.file_uploader("Pilih file audio (MP3/WAV)", type=["mp3", "wav"])
if uploaded_audio is not None:
    st.audio(uploaded_audio)
    temp_file_path = "temp_audio.wav"
    with open(temp_file_path, "wb") as f:
        f.write(uploaded_audio.read())

    with st.spinner("Memproses audio..."):
        y, sr = librosa.load(temp_file_path, sr=None)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        melspec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=64)

        mfcc_image = preprocess_mfcc(mfcc)
        melspec_image = preprocess_melspec(melspec)

        mfcc_result = mfcc_model.predict(np.expand_dims(mfcc_image, axis=0))
        melspec_result = melspec_model.predict(np.expand_dims(melspec_image, axis=0))

        mfcc_pred_class = np.argmax(mfcc_result)
        melspec_pred_class = np.argmax(melspec_result)

        st.write(f"**Model MFCC:** Prediksi kelas {mfcc_pred_class}")
        st.write(f"**Model Melspec:** Prediksi kelas {melspec_pred_class}")

# Footer
st.markdown("""
<div class="footer">
    <p>&copy; 2024 Klasifikasi Suara Burung Sumatera | Kelompok 11 Deep Learning</p>
</div>
""", unsafe_allow_html=True)
