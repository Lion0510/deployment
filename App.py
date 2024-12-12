import os
import json
import streamlit as st
import tensorflow as tf
import librosa
import librosa.display
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Menyembunyikan log TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Tambahkan CSS untuk styling agar fit ke layar dan tombol lebih rapi
st.markdown("""
    <style>
        body {
            font-family: Arial, sans-serif;
            background-color: #121212;
            color: #ffffff;
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        .main-header {
            background-color: #1E1E1E;
            padding: 20px;
            text-align: center;
            border-bottom: 2px solid #333;
            width: 100%;
        }
        .header-content {
            display: flex;
            justify-content: center;
            align-items: center;
        }
        .header-content img {
            width: 80px;
            height: auto;
            margin: 0 10px;
        }
        .header-title h1 {
            font-size: 3em;
            color: #ffffff;
            margin: 10px 0;
        }
        .header-title p {
            font-size: 1.2em;
            color: #cccccc;
            margin: 0;
        }
        nav ul {
            list-style-type: none;
            display: flex;
            justify-content: center;
            margin: 0;
            padding: 10px 0;
            background-color: #1E1E1E;
            width: 100%;
        }
        nav ul li {
            margin: 0 10px;
        }
        nav ul li button {
            background: none;
            border: 2px solid #ffffff;
            color: #ffffff;
            padding: 10px 20px;
            font-weight: bold;
            font-size: 14px;
            cursor: pointer;
            border-radius: 25px;
            transition: all 0.3s ease;
        }
        nav ul li button:hover {
            background-color: #ffffff;
            color: #000000;
            transform: scale(1.1);
        }
        .content-section {
            padding: 20px;
            margin: 20px auto;
            max-width: 1200px;
            text-align: center;
            background-color: #1E1E1E;
            border-radius: 10px;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.5);
        }
        footer {
            text-align: center;
            padding: 10px;
            background-color: #1E1E1E;
            border-top: 2px solid #333;
            width: 100%;
        }
        footer p {
            margin: 0;
            color: #cccccc;
        }
    </style>
""", unsafe_allow_html=True)

# Kamus deskripsi kelas burung
BIRD_CLASSES = {
    0: {
        "name": "Pitta sordida",
        "description": "Burung ini terkenal dengan bulu-bulunya yang warna-warni, seperti hijau, biru, dan kuning. Pitta Sayap Hitam hidup di hutan-hutan tropis dan suka mencari makan di tanah, biasanya berupa serangga kecil dan cacing.",
        "image": "images/Pitta_sordida.jpg"
    },
    1: {
        "name": "Dryocopus javensis",
        "description": "Burung pelatuk ini memiliki bulu hitam dengan warna merah mencolok di kepalanya. Ia menggunakan paruhnya yang kuat untuk mematuk batang pohon, mencari serangga, atau membuat sarang.",
        "image": "images/Dryocopus_javensis.jpg"
    },
    2: {
        "name": "Caprimulgus macrurus",
        "description": "Penjelasan: Burung ini aktif di malam hari dan memiliki bulu yang menyerupai warna kulit kayu, sehingga mudah berkamuflase. Kangkok Malam Besar memakan serangga dan sering ditemukan di area terbuka dekat hutan.",
        "image": "images/Caprimulgus_macrurus.jpg"
    },
    3: {
        "name": "Pnoepyga pusilla",
        "description": "Burung kecil ini hampir tidak memiliki ekor dan sering bersembunyi di semak-semak. Suaranya sangat nyaring meskipun ukurannya kecil. Mereka makan serangga kecil dan hidup di daerah pegunungan.",
        "image": "images/Pnoepyga_pusilla.jpg"
    },
    4: {
        "name": "Anthipes solitaris",
        "description": "Penjelasan: Kacer Soliter adalah burung kecil yang suka berada di dekat aliran sungai. Bulunya berwarna abu-abu dan putih dengan suara kicauan yang lembut. Ia sering makan serangga kecil.",
        "image": "images/Anthipes_solitaris.jpg"
    },
    5: {
        "name": "Buceros rhinoceros",
        "description": "Enggang Badak adalah burung besar dengan paruh besar yang melengkung dan tanduk di atasnya. Burung ini adalah simbol keberagaman hutan tropis dan sering ditemukan di Kalimantan dan Sumatra. Mereka memakan buah-buahan, serangga, dan bahkan hewan kecil.",
        "image": "images/Buceros_rhinoceros.jpg"
    }
}

# Fungsi untuk mendapatkan informasi kelas berdasarkan prediksi
def get_bird_info(pred_class):
    if pred_class in BIRD_CLASSES:
        return BIRD_CLASSES[pred_class]
    else:
        return {"name": "Unknown", "description": "Deskripsi tidak tersedia.", "image": None}

# State untuk navigasi
if "page" not in st.session_state:
    st.session_state.page = "home"

# Fungsi navigasi
def navigate(page):
    st.session_state.page = page

# Header
st.markdown("""
    <header class="main-header">
        <div class="header-content">
            <img src="https://fs.itera.ac.id/wp-content/uploads/2020/03/Logo-FSains.png" alt="Logo Fakultas Sains">
            <img src="https://blogger.googleusercontent.com/img/b/R29vZ2xl/AVvXsEhpSH0B8r5lSPmWBfANSG_LjlIEx2q0rEMXqQLxzr5Ggr7dSi7jfn7ALTDRPGrbUVkhgevNViaXgZokaU0_wwNme660o667wS7T_l4SzhKbQi50g2gLlVXsUNJBSbgOQ7nXi_hzfTDkv0yX/s320/logo+itera+oke.png" alt="Logo ITERA">
            <img src="https://pbs.twimg.com/profile_images/1272461269136576512/Uw9AShxq_400x400.jpg" alt="Logo Fakultas Teknologi">
        </div>
        <div class="header-title">
            <h1>Klasifikasi Suara Burung Sumatera</h1>
            <p>Identifikasi Burung Berdasarkan Suara Secara Otomatis</p>
        </div>
    </header>
""", unsafe_allow_html=True)

# Menu Navigasi dengan Streamlit
col1, col2, col3 = st.columns(3)

with col1:
    if st.button("Beranda"):
        navigate("home")

with col2:
    if st.button("Unggah Suara dan Hasil"):
        navigate("upload_results")

with col3:
    if st.button("Tentang"):
        navigate("about")

# Konten berdasarkan navigasi
if st.session_state.page == "home":
    st.markdown("""
        <section id="home" class="content-section">
            <h2>Selamat Datang</h2>
            <p>Aplikasi ini dapat membantu mengidentifikasi burung Sumatera melalui suara. Unggah file suara untuk melakukan identifikasi!</p>
        </section>
    """, unsafe_allow_html=True)

elif st.session_state.page == "upload_results":
    st.markdown("""
        <section id="upload" class="content-section">
            <h2>Unggah Suara dan Hasil</h2>
    """, unsafe_allow_html=True)
    uploaded_audio = st.file_uploader("Pilih file audio (MP3/WAV) untuk diuji", type=["mp3", "wav"])
    if uploaded_audio:
        st.audio(uploaded_audio, format="audio/mp3")
        temp_file_path = "temp_audio.wav"
        with open(temp_file_path, "wb") as f:
            f.write(uploaded_audio.read())

        with st.spinner("Memproses..."):
            try:
                y, sr = librosa.load(temp_file_path, sr=None)
                mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
                melspec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=64)

                st.subheader("Spektrum MFCC")
                fig, ax = plt.subplots(figsize=(12, 6))
                librosa.display.specshow(mfcc, sr=sr, x_axis='time', y_axis='mel', ax=ax, cmap='magma')
                plt.colorbar(format='%+2.0f dB')
                st.pyplot(fig)

                st.subheader("Spektrum Melspectrogram")
                fig, ax = plt.subplots(figsize=(12, 6))
                melspec_db = librosa.power_to_db(melspec, ref=np.max)
                librosa.display.specshow(melspec_db, sr=sr, x_axis='time', y_axis='mel', ax=ax, cmap='magma')
                plt.colorbar(format='%+2.0f dB')
                st.pyplot(fig)

                mfcc_image = np.expand_dims(mfcc, axis=(0, -1))
                melspec_image = np.expand_dims(melspec, axis=(0, -1))

                mfcc_result = mfcc_model.predict(mfcc_image)
                melspec_result = melspec_model.predict(melspec_image)

                mfcc_pred_class = np.argmax(mfcc_result, axis=1)[0]
                melspec_pred_class = np.argmax(melspec_result, axis=1)[0]
                mfcc_accuracy = np.max(mfcc_result)
                melspec_accuracy = np.max(melspec_result)

                mfcc_bird_info = get_bird_info(mfcc_pred_class)
                melspec_bird_info = get_bird_info(melspec_pred_class)

                st.subheader("Hasil Prediksi:")
                st.write(f"**Model MFCC:** Prediksi kelas {mfcc_pred_class} dengan akurasi {mfcc_accuracy * 100:.2f}%")
                st.write(f"Nama: {mfcc_bird_info['name']}")
                st.write(f"Deskripsi: {mfcc_bird_info['description']}")
                if mfcc_bird_info['image']:
                    st.image(mfcc_bird_info['image'], caption=f"{mfcc_bird_info['name']} (Model MFCC)")

                st.write("---")
                st.write(f"**Model Melspec:** Prediksi kelas {melspec_pred_class} dengan akurasi {melspec_accuracy * 100:.2f}%")
                st.write(f"Nama: {melspec_bird_info['name']}")
                st.write(f"Deskripsi: {melspec_bird_info['description']}")
                if melspec_bird_info['image']:
                    st.image(melspec_bird_info['image'], caption=f"{melspec_bird_info['name']} (Model Melspec)")

            except Exception as e:
                st.error(f"Error saat melakukan prediksi: {str(e)}")

elif st.session_state.page == "about":
    st.markdown("""
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
