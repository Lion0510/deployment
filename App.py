import os
import streamlit as st
import tensorflow as tf
from kaggle.api.kaggle_api_extended import KaggleApi
import zipfile
import shutil

# Set Streamlit page configuration
st.set_page_config(page_title="Bird Song Classifier", page_icon="🦜", layout="centered")

# Fungsi untuk mengunduh dataset atau model dari Kaggle
def download_dataset_from_kaggle(dataset_name, dest_path):
    # Setup Kaggle API
    api = KaggleApi()
    api.authenticate()

    try:
        # Mengunduh dataset dari Kaggle
        st.info(f"Mengunduh dataset {dataset_name}...")
        api.dataset_download_files(dataset_name, path=dest_path, unzip=True)
        st.success(f"Dataset berhasil diunduh ke {dest_path}")
    except Exception as e:
        st.error(f"Terjadi error saat mengunduh dataset: {str(e)}")

# Mengambil kredensial API Kaggle dari Streamlit Secrets
os.environ['KAGGLE_USERNAME'] = st.secrets["kaggle.json"]['username']
os.environ['KAGGLE_KEY'] = st.secrets["kaggle.json"]['key']

# Nama dataset atau model di Kaggle
dataset_name = 'evanaryaputra28/dl-tb'  # Gantilah dengan nama dataset atau model yang sesuai
output_dest_path = 'kaggle_output/'

# Pastikan folder untuk menyimpan output ada
if not os.path.exists(output_dest_path):
    os.makedirs(output_dest_path)

# Unduh dataset atau model dari Kaggle
download_dataset_from_kaggle(dataset_name, output_dest_path)

# Tentukan path model yang sudah diunduh
melspec_model_path = os.path.join(output_dest_path, 'melspec_model.h5')  # Sesuaikan nama model Anda
mfcc_model_path = os.path.join(output_dest_path, 'mfcc_model.h5')  # Sesuaikan nama model Anda

# Pastikan model sudah diunduh
if os.path.exists(melspec_model_path) and os.path.exists(mfcc_model_path):
    try:
        melspec_model = tf.keras.models.load_model(melspec_model_path)
        mfcc_model = tf.keras.models.load_model(mfcc_model_path)
        st.success("Model berhasil dimuat!")
    except Exception as e:
        st.error(f"Terjadi error saat memuat model: {str(e)}")
else:
    st.error("Model tidak ditemukan setelah diunduh!")

# Title aplikasi Streamlit
st.title("Deep Learning in Audio: Klasifikasi Suara Burung di Indonesia Bagian Barat 🦜")

# Deskripsi aplikasi
st.markdown("""
    **Selamat datang di aplikasi klasifikasi suara burung menggunakan Deep Learning!**
    Aplikasi ini mengimplementasikan teknik ekstraksi fitur audio menggunakan MFCC dan Melspectrogram, 
    serta menggunakan dua model CNN yang berbeda untuk mengklasifikasikan suara burung yang ada di Indonesia Bagian Barat.
    Unggah file audio dalam format MP3 atau WAV, dan model akan memberikan prediksi kelas burung beserta akurasi dari kedua model.
""")

# Upload file audio
st.header("Unggah File Audio Suara Burung")
uploaded_audio = st.file_uploader("Pilih file audio (MP3/WAV) untuk diuji", type=["mp3", "wav"])

# Fungsi dummy prediksi (ganti dengan fungsi prediksi yang sebenarnya)
def dummy_predict(file_path):
    # Fungsi dummy untuk prediksi
    # Ganti dengan implementasi yang sesuai untuk model Anda
    return {"mfcc": {"class": 2, "accuracy": 0.85}, "melspec": {"class": 3, "accuracy": 0.90}}

# Proses audio yang diunggah
if uploaded_audio is not None:
    st.audio(uploaded_audio, format="audio/mp3")
    
    # Simpan file audio sementara
    temp_file_path = "temp_audio.wav"
    with open(temp_file_path, "wb") as f:
        f.write(uploaded_audio.read())

    # Prediksi dengan model jika tombol ditekan
    if st.button("Prediksi Kelas Burung"):
        with st.spinner("Memproses..."):
            try:
                # Lakukan prediksi (ganti dengan implementasi prediksi asli menggunakan model)
                results = dummy_predict(temp_file_path)
                mfcc_result = results["mfcc"]
                melspec_result = results["melspec"]

                # Tampilkan hasil prediksi
                st.subheader("Hasil Prediksi:")
                st.write(f"**Model MFCC:** Prediksi kelas {mfcc_result['class']} dengan akurasi {mfcc_result['accuracy'] * 100:.2f}%")
                st.write(f"**Model Melspec:** Prediksi kelas {melspec_result['class']} dengan akurasi {melspec_result['accuracy'] * 100:.2f}%")
            except Exception as e:
                st.error(f"Error saat melakukan prediksi: {str(e)}")

# Footer
st.markdown("""
    <hr>
    <p style="text-align:center; font-size:14px; color:#888; margin-top: 10px; margin-bottom: 10px;">
        Aplikasi Klasifikasi Suara Burung menggunakan Deep Learning | Dibuat oleh Kelompok 11
    </p>
""", unsafe_allow_html=True)
