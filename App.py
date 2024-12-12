import os
import streamlit as st

def add_custom_css():
    st.markdown("""
    <style>
    /* Gaya dasar */
    body {
        font-family: Arial, sans-serif;
        background-color: #f0f0f0;
        margin: 0;
        padding: 0;
    }

    /* Container navigasi */
    .navigation-container {
        display: flex;
        justify-content: center;
        align-items: center;
        background-color: #fff;
        box-shadow: 0 2px 5px rgba(0, 0, 0, 0.3);
        padding: 10px;
        border-radius: 5px;
        margin: 20px auto;
        max-width: 800px;
    }

    /* Tombol navigasi */
    .navigation-button {
        padding: 15px 25px;
        background-color: transparent;
        border: none;
        color: #333;
        text-decoration: none;
        font-weight: bold;
        transition: all 0.3s ease;
    }

    .navigation-button:hover {
        background-color: #eee;
        border-radius: 5px;
    }

    /* Tombol aktif */
    .active {
        background-color: #eee;
        border-radius: 5px;
    }

    /* Responsivitas (opsional) */
    @media (max-width: 768px) {
        .navigation-container {
            flex-direction: column;
            align-items: flex-start;
        }
        .navigation-button {
            margin-bottom: 10px;
        }
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
st.title("Klasifikasi Suara Burung Sumatera")
st.write("Identifikasi burung Sumatera secara otomatis melalui suara")

# Navigasi Horizontal
with st.container():
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("Beranda", key="home_button", class_="navigation-button" + (" active" if st.session_state.page == "home" else "")):
            navigate("home")
    with col2:
        if st.button("Unggah Audio", key="upload_results_button", class_="navigation-button" + (" active" if st.session_state.page == "upload_results" else "")):
            navigate("upload_results")
    with col3:
        if st.button("Tentang", key="about_button", class_="navigation-button" + (" active" if st.session_state.page == "about" else "")):
            navigate("about")

# Konten Berdasarkan Halaman
if st.session_state.page == "home":
    st.write("Halaman Beranda")
elif st.session_state.page == "upload_results":
    # ... (kode unggah audio dan hasil)
elif st.session_state.page == "about":
    st.write("Halaman Tentang")

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
