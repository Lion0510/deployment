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

# Menyembunyikan log TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Gaya CSS untuk navigasi dan halaman dengan background gambar
st.markdown("""
<style>
body {
    font-family: 'Montserrat', sans-serif;
    background-image: url('https://jenis.net/wp-content/uploads/2020/06/jenis-nyamuk-e1591437296119-768x456.jpg'); /* Ganti dengan path gambar Anda */
    background-size: cover;  /* Agar gambar memenuhi layar */
    background-position: center center;  /* Memposisikan gambar di tengah */
    background-attachment: fixed;  /* Membuat gambar tetap saat scroll */
    margin: 0;
    padding: 0;
    color: #fff;
}

.navigation-container {
    display: flex;
    justify-content: center;
    align-items: center;
    background-color: rgba(255, 255, 255, 0.8);  /* Warna latar belakang transparan untuk navigasi */
    box-shadow: 0 2px 5px rgba(0, 0, 0, 0.3);
    padding: 10px;
    border-radius: 5px;
    margin: 20px auto;
    max-width: 800px;
}

.navigation-button {
    padding: 15px 25px;
    background-color: #fff;
    color: #333;
    border: 1px solid #ccc;
    text-decoration: none;
    font-weight: bold;
    transition: all 0.3s ease;
    cursor: pointer;
}

.navigation-button:hover {
    background-color: #e0e0e0;
    box-shadow: 0px 2px 5px rgba(0, 0, 0, 0.2);
}

.active {
    background-color: #2196F3;
    color: #fff;
    border-color: #2196F3;
}

.header-content {
    display: flex;
    justify-content: center;
    align-items: center;
    gap: 20px; /* Memberikan jarak antara logo */
}

.logo {
    width: 100px;  /* Atur ukuran logo */
    height: auto;
    display: inline-block;
}

.header-title h1 {
    color: #fff;
    text-align: center;
    margin-top: 20px;
    font-size: 2em;
}

.header-title p {
    text-align: center;
    margin-top: 5px;
    font-size: 1.2em;
    color: #ccc;
}

.content-section {
    text-align: center;  /* Menengahkan teks */
    background-color: rgba(0, 0, 0, 0.6);  /* Latar belakang transparan untuk teks */
    border-radius: 10px;
    padding: 20px;
    margin: 20px auto;
    max-width: 800px;
    color: white;
    box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);  /* Memberikan bayangan pada container */
}

.footer {
    text-align: center;
    margin-top: 20px;
    background-color: rgba(0, 0, 0, 0.8);
    color: white;
    padding: 10px;
    border-radius: 10px;
    box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
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

# Fungsi untuk mengunduh model dari Kaggle API
def download_model_from_kaggle(kernel_name, output_files, dest_folder):
    try:
        model_files_exist = all([os.path.exists(os.path.join(dest_folder, file)) for file in output_files])
        if model_files_exist:
            return False  # Model sudah ada, tidak perlu mengunduh ulang

        kaggle_username = st.secrets["kaggle"]["KAGGLE_USERNAME"]
        kaggle_key = st.secrets["kaggle"]["KEY"]

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

# Tambahkan CSS ke aplikasi
#add_custom_css()

# Header
st.markdown("""
<header class="main-header">
    <div class="header-content">
        <img src="images/Logo2.png" alt="Logo Fakultas Sains" class="logo">
        <img src="https://blogger.googleusercontent.com/img/b/R29vZ2xl/AVvXsEhpSH0B8r5lSPmWBfANSG_LjlIEx2q0rEMXqQLxzr5Ggr7dSi7jfn7ALTDRPGrbUVkhgevNViaXgZokaU0_wwNme660o667wS7T_l4SzhKbQi50g2gLlVXsUNJBSbgOQ7nXi_hzfTDkv0yX/s320/logo+itera+oke.png" alt="Logo ITERA" class="logo">
        <img src="https://pbs.twimg.com/profile_images/1272461269136576512/Uw9AShxq_400x400.jpg" alt="Logo Fakultas Teknologi" class="logo">
    </div>
    <div class="header-title">
        <h1>Klasifikasi Suara Burung Sumatera 🦜</h1>
        <p>Identifikasi Burung Berdasarkan Suara Secara Otomatis</p>
    </div>
</header>
""", unsafe_allow_html=True)

# Konten Aplikasi
st.markdown("""
<section class="content-section">
    <h2>Klasifikasi Suara</h2>
    <p>Burung yang termasuk dalam klasifikasi ini adalah:</p>
    <ul>
        <li>Pitta sordida</li>
        <li>Dryocopus javensis</li>
        <li>Caprimulgus macrurus</li>
        <li>Pnoepyga pusilla</li>
        <li>Anthipes solitaris</li>
        <li>Buceros rhinoceros</li>
    </ul>
""", unsafe_allow_html=True)

uploaded_audio = st.file_uploader("Pilih file audio (MP3/WAV) untuk diuji", type=["mp3", "wav"])
if uploaded_audio is not None:
    st.audio(uploaded_audio, format="audio/mp3")
    st.write("Hasil klasifikasi akan muncul di sini setelah proses selesai.")

st.markdown("""
</section>
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
