import os
import streamlit as st
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Menyembunyikan log TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Menambahkan CSS untuk styling
def add_custom_css():
    st.markdown("""
    <style>
        body {
            font-family: 'Arial', sans-serif;
            background-color: #121212;
            color: #ffffff;
            margin: 0;
            padding: 0;
        }
        .main-header {
            text-align: center;
            padding: 20px 0;
            border-bottom: 3px solid #333;
        }
        .main-header .header-images {
            display: flex;
            justify-content: center;
            align-items: center;
            gap: 40px;
            margin-bottom: 20px;
        }
        .main-header img {
            width: 100px;
            height: auto;
        }
        .main-header h1 {
            font-size: 2.5em;
            color: #ffffff;
            margin: 10px 0;
        }
        .main-header p {
            color: #cccccc;
            font-size: 1.2em;
            margin-bottom: 30px;
        }
        .navigation-container {
            display: flex;
            justify-content: center;
            align-items: center;
            background-color: #1E1E1E;
            padding: 15px 0;
            border-bottom: 2px solid #333;
            gap: 15px;
            margin-bottom: 30px;
        }
        .navigation-container button {
            font-size: 16px;
            font-weight: bold;
            padding: 10px 20px;
            background-color: #333;
            border: 2px solid #ffffff;
            color: #ffffff;
            border-radius: 25px;
            cursor: pointer;
            transition: all 0.3s ease;
        }
        .navigation-container button:hover {
            background-color: #ffffff;
            color: #333;
        }
        .content-section {
            padding: 40px;
            margin: 30px auto;
            max-width: 800px;
            text-align: center;
            background-color: #1E1E1E;
            border-radius: 10px;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.5);
        }
        footer {
            text-align: center;
            padding: 20px;
            background-color: #1E1E1E;
            border-top: 2px solid #333;
            margin-top: 30px;
        }
        footer p {
            margin: 0;
            color: #cccccc;
        }
    </style>
    """, unsafe_allow_html=True)

# Tambahkan CSS ke aplikasi
add_custom_css()

# State untuk navigasi
if "page" not in st.session_state:
    st.session_state.page = "home"

# Fungsi navigasi
def navigate(page):
    st.session_state.page = page

# Header
st.markdown("""
<header class="main-header">
    <div class="header-images">
        <img src="https://fs.itera.ac.id/wp-content/uploads/2020/03/Logo-FSains.png" alt="Logo Fakultas Sains">
        <img src="https://blogger.googleusercontent.com/img/b/R29vZ2xl/AVvXsEhpSH0B8r5lSPmWBfANSG_LjlIEx2q0rEMXqQLxzr5Ggr7dSi7jfn7ALTDRPGrbUVkhgevNViaXgZokaU0_wwNme660o667wS7T_l4SzhKbQi50g2gLlVXsUNJBSbgOQ7nXi_hzfTDkv0yX/s320/logo+itera+oke.png" alt="Logo ITERA">
        <img src="https://pbs.twimg.com/profile_images/1272461269136576512/Uw9AShxq_400x400.jpg" alt="Logo Fakultas Teknologi">
    </div>
    <h1>Klasifikasi Suara Burung Sumatera</h1>
    <p>Identifikasi Burung Berdasarkan Suara Secara Otomatis</p>
</header>
""", unsafe_allow_html=True)

# Navigasi dengan tombol Streamlit
with st.container():
    st.markdown('<div class="navigation-container">', unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        if st.button("Beranda"):
            navigate("home")
    with col2:
        if st.button("Unggah Suara dan Hasil"):
            navigate("upload_results")
    with col3:
        if st.button("Tentang Kami"):
            navigate("about")
    st.markdown('</div>', unsafe_allow_html=True)

# Konten berdasarkan navigasi
if st.session_state.page == "home":
    st.markdown("""
    <div class="content-section">
        <h2>Selamat Datang</h2>
        <p>Aplikasi ini dapat membantu mengidentifikasi burung Sumatera melalui suara. Unggah file suara untuk melakukan identifikasi!</p>
    </div>
    """, unsafe_allow_html=True)

elif st.session_state.page == "upload_results":
    st.markdown("""
    <div class="content-section">
        <h2>Unggah Suara dan Hasil</h2>
    </div>
    """, unsafe_allow_html=True)
    uploaded_audio = st.file_uploader("Pilih file audio (MP3/WAV) untuk diuji", type=["mp3", "wav"])
    if uploaded_audio:
        st.audio(uploaded_audio, format="audio/mp3")
        st.markdown("<p style='color: lightgray;'>File berhasil diunggah. Analisis akan segera dimulai!</p>", unsafe_allow_html=True)

elif st.session_state.page == "about":
    st.markdown("""
    <div class="content-section">
        <h2>Tentang Kami</h2>
        <p>Aplikasi ini dirancang oleh Kelompok 11 prodi Sains Data ITERA untuk mendukung konservasi burung di Sumatera.</p>
    </div>
    """, unsafe_allow_html=True)

# Footer
st.markdown("""
<footer>
    <p>&copy; 2024 Klasifikasi Suara Burung Sumatera | Kelompok 11 Deep Learning Sains Data</p>
</footer>
""", unsafe_allow_html=True)
