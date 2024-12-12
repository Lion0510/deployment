import os
import json
import streamlit as st
from kaggle.api.kaggle_api_extended import KaggleApi
import tensorflow as tf
import librosa
import librosa.display
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Menambahkan CSS untuk meniru gaya HTML yang diberikan
def add_custom_css():
    st.markdown("""
    <style>
    body {
        font-family: 'Arial', sans-serif;
        background-color: #f0f0f0;
        margin: 0;
        padding: 0;
    }
    .main-header {
        text-align: center;
        padding: 20px;
        background-color: #1E1E1E;
        color: #ffffff;
    }
    .main-header .header-content {
        display: flex;
        justify-content: center;
        align-items: center;
        gap: 20px;
    }
    .main-header .logo {
        width: 80px;
        height: auto;
    }
    .main-header .header-title h1 {
        font-size: 2.5rem;
        margin: 10px 0;
    }
    .main-header .header-title p {
        font-size: 1.2rem;
        margin: 0;
    }
    nav {
        background-color: #fff;
        box-shadow: 0 2px 5px rgba(0, 0, 0, 0.2);
        padding: 10px 0;
    }
    nav ul {
        list-style: none;
        display: flex;
        justify-content: center;
        padding: 0;
        margin: 0;
    }
    nav ul li {
        margin: 0 15px;
    }
    nav ul li a {
        text-decoration: none;
        font-weight: bold;
        color: #333;
        transition: color 0.3s;
    }
    nav ul li a:hover {
        color: #2196F3;
    }
    .content-section {
        max-width: 800px;
        margin: 20px auto;
        padding: 20px;
        background: #fff;
        border-radius: 10px;
        box-shadow: 0 2px 5px rgba(0, 0, 0, 0.2);
    }
    .content-section h2 {
        font-size: 2rem;
        margin-bottom: 10px;
    }
    footer {
        text-align: center;
        padding: 20px;
        background-color: #1E1E1E;
        color: #cccccc;
    }
    </style>
    """, unsafe_allow_html=True)

# Tambahkan CSS ke aplikasi
add_custom_css()

# Header
st.markdown("""
<header class="main-header">
    <div class="header-content">
        <img src="https://fs.itera.ac.id/wp-content/uploads/2020/03/Logo-FSains.png" alt="Logo Fakultas Sains" class="logo">
        <img src="https://blogger.googleusercontent.com/img/b/R29vZ2xl/AVvXsEhpSH0B8r5lSPmWBfANSG_LjlIEx2q0rEMXqQLxzr5Ggr7dSi7jfn7ALTDRPGrbUVkhgevNViaXgZokaU0_wwNme660o667wS7T_l4SzhKbQi50g2gLlVXsUNJBSbgOQ7nXi_hzfTDkv0yX/s320/logo+itera+oke.png" alt="Logo ITERA" class="logo">
        <img src="https://pbs.twimg.com/profile_images/1272461269136576512/Uw9AShxq_400x400.jpg" alt="Logo Fakultas Teknologi" class="logo">
    </div>
    <div class="header-title">
        <h1>Klasifikasi Suara Burung Sumatera</h1>
        <p>Identifikasi Burung Berdasarkan Suara Secara Otomatis</p>
    </div>
</header>
""", unsafe_allow_html=True)

# Fungsi untuk mengunduh model dari Kaggle API
def download_model_from_kaggle(kernel_name, output_files, dest_folder):
    try:
        model_files_exist = all([os.path.exists(os.path.join(dest_folder, file)) for file in output_files])
        if model_files_exist:
            return False  # Model sudah ada, tidak perlu mengunduh ulang

        kaggle_username = st.secrets["kaggle"]["KAGGLE_USERNAME"]
        kaggle_key = st.secrets["kaggle"]["KAGGLE_KEY"]

        kaggle_json_path = os.path.expanduser("/home/appuser/.kaggle/kaggle.json")
        os.makedirs(os.path.dirname(kaggle_json_path), exist_ok=True)

        with open(kaggle_json_path, 'w') as f:
            json.dump({"username": kaggle_username, "key": kaggle_key}, f)

        api = KaggleApi()
        api.authenticate()

        os.makedirs(dest_folder, exist_ok=True)
        for output_file in output_files:
            api.kernels_output(kernel_name, path=dest_folder, force=True)
        return True
    except Exception as e:
        st.error(f"Terjadi kesalahan saat mengunduh model: {str(e)}")
        return None

kernel_name = "evanaryaputra28/tubes-dll"
output_files = ["cnn_melspec.h5", "cnn_mfcc.h5"]
dest_folder = "./models/"

download_status = download_model_from_kaggle(kernel_name, output_files, dest_folder)

melspec_model_save_path = os.path.join(dest_folder, 'cnn_melspec.h5')
mfcc_model_save_path = os.path.join(dest_folder, 'cnn_mfcc.h5')

if os.path.exists(melspec_model_save_path):
    try:
        melspec_model = tf.keras.models.load_model(melspec_model_save_path)
    except Exception as e:
        st.error(f"Gagal memuat model Melspec: {str(e)}")

if os.path.exists(mfcc_model_save_path):
    try:
        mfcc_model = tf.keras.models.load_model(mfcc_model_save_path)
    except Exception as e:
        st.error(f"Gagal memuat model MFCC: {str(e)}")
        
# Fungsi untuk memproses MFCC menjadi gambar 64x64x3
def preprocess_mfcc(mfcc):
    mfcc_image = Image.fromarray(mfcc)
    mfcc_image = mfcc_image.resize((64, 64))
    mfcc_resized = np.array(mfcc_image)
    mfcc_resized = np.expand_dims(mfcc_resized, axis=-1)
    mfcc_resized = np.repeat(mfcc_resized, 3, axis=-1)
    return mfcc_resized

# Fungsi untuk memproses Melspectrogram menjadi gambar 64x64x3
def preprocess_melspec(melspec):
    melspec_db = librosa.power_to_db(melspec, ref=np.max)
    melspec_image = Image.fromarray(melspec_db)
    melspec_image = melspec_image.resize((64, 64))
    melspec_resized = np.array(melspec_image)
    melspec_resized = np.expand_dims(melspec_resized, axis=-1)
    melspec_resized = np.repeat(melspec_resized, 3, axis=-1)
    return melspec_resized

# Fungsi untuk menampilkan spektrum
def plot_spectrogram(data, sr, title, y_axis, x_axis):
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(data, sr=sr, x_axis=x_axis, y_axis=y_axis, cmap='magma')
    plt.colorbar(format='%+2.0f dB')
    plt.title(title)
    plt.tight_layout()
    st.pyplot(plt)
    plt.close()


# Navigasi
st.markdown("""
<nav>
    <ul>
        <li><a href="#home">Beranda</a></li>
        <li><a href="#upload">Unggah Suara</a></li>
        <li><a href="#results">Hasil</a></li>
        <li><a href="#about">Tentang</a></li>
    </ul>
</nav>
""", unsafe_allow_html=True)

# Konten
st.markdown("""
<section id="home" class="content-section">
    <h2>Selamat Datang</h2>
    <p>Aplikasi ini dapat membantu mengidentifikasi burung Sumatera melalui suara. Unggah file suara untuk melakukan identifikasi!</p>
</section>

<section id="upload" class="content-section">
    <h2>Unggah Suara</h2>
    <form>
        <label for="audio-upload">Pilih file suara (format .wav):</label>
        <input type="file" id="audio-upload" accept=".wav" required>
        <button type="submit">Klasifikasi</button>
    </form>
</section>

<section id="results" class="content-section">
    <h2>Hasil Klasifikasi</h2>
    <div>
        <p>Hasil klasifikasi akan muncul di sini setelah Anda mengunggah suara.</p>
    </div>
</section>

<section id="about" class="content-section">
    <h2>Tentang Kami</h2>
    <p>Aplikasi ini dirancang oleh Kelompok 11 prodi Sains Data ITERA untuk mendukung konservasi burung di Sumatera.</p>
</section>
""", unsafe_allow_html=True)

# Footer
st.markdown("""
<footer>
    <p>&copy; 2024 Klasifikasi Suara Burung Sumatera | Kelompok 11 Deep Learning Sains Data</p>
</footer>
""", unsafe_allow_html=True)
