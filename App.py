import os
import json
import gdown
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

# Menambahkan CSS untuk styling halaman
st.markdown("""
    <style>
        /* Background halaman */
        .stApp {
            background-image: url('https://raw.githubusercontent.com/Lion0510/deployment/main/images/bg.jpg');
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
        }

        /* Styling Header */
        .header-content {
            display: flex;
            justify-content: center;
            align-items: center;
            gap: 20px; /* Jarak antar logo */
            margin-top: 10px;
        }

        .logo {
            background-color: white; /* Background putih untuk logo */
            padding: 10px; /* Padding di dalam logo */
            border-radius: 10px; /* Membuat sudut membulat */
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2); /* Bayangan lembut */
            display: inline-block; /* Sesuai ukuran konten */
            width: 100px; /* Ukuran logo */
            height: auto;
        }

        /* Judul Header */
        .header-box {
            background-color: rgba(0, 0, 0, 0.6); /* Latar hitam transparan */
            color: white; /* Warna teks putih */
            padding: 15px 20px;
            border-radius: 10px;
            text-align: center;
            margin: 20px auto;
            display: inline-block;
            box-shadow: 0px 2px 5px rgba(0, 0, 0, 0.3);
        }

        /* Konten Utama */
        .content-section {
            background-color: rgba(0, 0, 0, 0.6);
            color: white;
            padding: 20px;
            border-radius: 10px;
            margin: 20px auto;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.3);
        }

        /* Footer */
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

# Konten Aplikasi
st.markdown("""
<section class="content-section">
    <h2>Selamat Datang di Aplikasi Klasifikasi Suara Burung</h2>
    <p>Aplikasi ini dirancang untuk mengidentifikasi jenis burung berdasarkan suara yang direkam.</p>
    <ul>
        <li>Pitta sordida</li>
        <li>Dryocopus javensis</li>
        <li>Caprimulgus macrurus</li>
        <li>Pnoepyga pusilla</li>
        <li>Anthipes solitaris</li>
        <li>Buceros rhinoceros</li>
    </ul>
</section>
""", unsafe_allow_html=True)

uploaded_audio = st.file_uploader("Pilih file audio (MP3/WAV) untuk diuji", type=["mp3", "wav"])

if uploaded_audio is not None:
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
            plot_spectrogram(mfcc, sr, "MFCC", y_axis="mel", x_axis="time")

            st.subheader("Spektrum Melspectrogram")
            melspec_db = librosa.power_to_db(melspec, ref=np.max)
            plot_spectrogram(melspec_db, sr, "Melspectrogram", y_axis="mel", x_axis="time")

            mfcc_image = preprocess_mfcc(mfcc)
            melspec_image = preprocess_melspec(melspec)

            mfcc_image = np.expand_dims(mfcc_image, axis=0)
            melspec_image = np.expand_dims(melspec_image, axis=0)

            mfcc_result = mfcc_model.predict(mfcc_image)
            melspec_result = melspec_model.predict(melspec_image)

            mfcc_pred_class = np.argmax(mfcc_result, axis=1)[0]
            melspec_pred_class = np.argmax(melspec_result, axis=1)[0]
            mfcc_accuracy = np.max(mfcc_result)
            melspec_accuracy = np.max(melspec_result)

            mfcc_bird_info = get_bird_info(mfcc_pred_class)
            melspec_bird_info = get_bird_info(melspec_pred_class)

            st.subheader("Hasil Prediksi:")
            st.write(f"*Model MFCC:* Prediksi kelas {mfcc_pred_class} dengan akurasi {mfcc_accuracy * 100:.2f}%")
            st.write(f"Nama: {mfcc_bird_info['name']}")
            st.write(f"Deskripsi: {mfcc_bird_info['description']}")
            if mfcc_bird_info['image']:
                st.image(mfcc_bird_info['image'], caption=f"{mfcc_bird_info['name']} (Model MFCC)")

            st.write("---")
            st.write(f"*Model Melspec:* Prediksi kelas {melspec_pred_class} dengan akurasi {melspec_accuracy * 100:.2f}%")
            st.write(f"Nama: {melspec_bird_info['name']}")
            st.write(f"Deskripsi: {melspec_bird_info['description']}")
            if melspec_bird_info['image']:
                st.image(melspec_bird_info['image'], caption=f"{melspec_bird_info['name']} (Model Melspec)")

        except Exception as e:
            st.error(f"Error saat melakukan prediksi: {str(e)}")

st.markdown("""
</section>
<section class="content-section">
    <h2>Tentang Kami</h2>
    <p>Aplikasi ini dirancang oleh Kelompok 11 prodi Sains Data ITERA untuk mendukung konservasi burung di Sumatera.</p>
</section>
""", unsafe_allow_html=True)

# Footer
st.markdown("""
<div class="footer">
    <p>&copy; 2024 Klasifikasi Suara Burung Sumatera | Kelompok 11 Deep Learning ITERA</p>
</div>
""", unsafe_allow_html=True)
