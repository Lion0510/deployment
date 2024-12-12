import streamlit as st
import tensorflow as tf
import json
import os
import librosa
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

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

# State untuk navigasi
if "page" not in st.session_state:
    st.session_state.page = "home"

def navigate(page):
    st.session_state.page = page

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

# Navigasi Horizontal
st.markdown("""
<nav>
    <ul>
        <li><a href="#" onclick="window.location='/home'">Beranda</a></li>
        <li><a href="#" onclick="window.location='/klasifikasi-suara'">Klasifikasi Suara</a></li>
        <li><a href="#" onclick="window.location='/tentang'">Tentang</a></li>
    </ul>
</nav>
""", unsafe_allow_html=True)

# Konten Berdasarkan Halaman
if st.session_state.page == "home":
    st.markdown("""
    <section class="content-section">
        <h2>Selamat Datang</h2>
        <p>Halaman Beranda: Temukan informasi tentang klasifikasi suara burung di sini.</p>
    </section>
    """, unsafe_allow_html=True)
elif st.session_state.page == "klasifikasi-suara":
    st.markdown("""
    <section class="content-section">
        <h2>Klasifikasi Suara</h2>
    </section>
    """, unsafe_allow_html=True)
    uploaded_audio = st.file_uploader("Pilih file audio (MP3/WAV) untuk diuji", type=["mp3", "wav"])
    if uploaded_audio is not None:
        st.audio(uploaded_audio, format="audio/mp3")
        st.write("Hasil klasifikasi akan muncul di sini setelah proses selesai.")
elif st.session_state.page == "tentang":
    st.markdown("""
    <section class="content-section">
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
