import os
import streamlit as st

# Menambahkan CSS untuk gaya navigasi horizontal
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
            background-color: #1E1E1E;
            border-bottom: 3px solid #333;
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
            justify-content: space-evenly;
            align-items: center;
            background-color: rgba(0, 0, 0, 0.7); /* Transparansi dengan latar belakang gelap */
            padding: 15px;
            border-radius: 10px;
            margin: 20px auto;
            max-width: 800px;
        }
        .navigation-button {
            font-size: 16px;
            font-weight: bold;
            padding: 10px 20px;
            color: #ffffff;
            text-transform: uppercase;
            text-decoration: none;
            background: none;
            border: none; /* Tidak ada border */
            cursor: pointer;
            transition: all 0.3s ease;
        }
        .navigation-button:hover {
            background-color: #ffffff;
            color: #333;
            border-radius: 5px; /* Opsional untuk memberikan efek rounded saat hover */
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
    <h1>Klasifikasi Suara Burung Sumatera</h1>
    <p>Identifikasi burung Sumatera secara otomatis melalui suara</p>
</header>
""", unsafe_allow_html=True)

# Navigasi Horizontal
st.markdown('<div class="navigation-container">', unsafe_allow_html=True)
col1, col2, col3 = st.columns(3)
with col1:
    if st.button("Beranda", key="home_button"):
        navigate("home")
with col2:
    if st.button("Unggah Audio", key="upload_results_button"):
        navigate("upload_results")
with col3:
    if st.button("Tentang", key="about_button"):
        navigate("about")
st.markdown('</div>', unsafe_allow_html=True)

# Konten Berdasarkan Halaman
if st.session_state.page == "home":
    st.markdown("<h2>Selamat Datang</h2><p>Halaman Beranda</p>", unsafe_allow_html=True)
elif st.session_state.page == "upload_results":
    st.markdown("<h2>Unggah Suara dan Hasil</h2><p>Unggah file suara untuk identifikasi dan lihat hasilnya.</p>", unsafe_allow_html=True)
    uploaded_audio = st.file_uploader("Pilih file audio (MP3/WAV) untuk diuji", type=["mp3", "wav"])
    if uploaded_audio:
        st.audio(uploaded_audio, format="audio/mp3")
        st.markdown("<p style='color: lightgray;'>File berhasil diunggah. Analisis akan segera dimulai!</p>", unsafe_allow_html=True)
        # Tambahkan logika hasil di sini jika ada model prediksi
        st.markdown("<h3>Hasil Identifikasi:</h3><p>[Hasil akan ditampilkan di sini]</p>", unsafe_allow_html=True)
elif st.session_state.page == "about":
    st.markdown("<h2>Tentang</h2><p>Informasi tentang aplikasi ini.</p>", unsafe_allow_html=True)

# Footer
st.markdown("""
<footer style="text-align: center; margin-top: 50px; color: #cccccc;">
    <p>&copy; 2024 Klasifikasi Suara Burung Sumatera | Kelompok 11 Deep Learning Sains Data</p>
</footer>
""", unsafe_allow_html=True)
