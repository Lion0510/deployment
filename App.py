import streamlit as st
import numpy as np
import librosa
import tensorflow as tf
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
import gdown 

# Google Drive file IDs
melspec_model_url = 'https://drive.google.com/uc?id=192VGvINbZKOyjhGioyBhjfd2alGe6ATM'
mfcc_model_url = 'https://drive.google.com/uc?id=1aRBAt6bHVMW3t6QwbLHzCPn3fQuqd71h'

# Set Streamlit page configuration
st.set_page_config(page_title="Bird Song Classifier", page_icon="🦜", layout="centered")

# Set the background color and theme
st.markdown("""
    <style>
        body {
            background-color: #F0F8FF; /* Soft light blue background */
            color: #333333;  /* Dark grey text */
        }
        .stButton>button {
            background-color: #1E90FF;
            color: white;
            border-radius: 12px;
            font-size: 18px;
        }
        .stButton>button:hover {
            background-color: #4682B4;
        }
        .stTextInput input {
            background-color: #FFFAF0;
            border: 2px solid #1E90FF;
        }
        h1 {
            color: #1E90FF;
            font-family: 'Arial', sans-serif;
        }
        .stFileUploader {
            background-color: #FFFAF0;
            border: 2px dashed #1E90FF;
        }
        .stProgress {
            background-color: #1E90FF;
        }
    </style>
""", unsafe_allow_html=True)

# Title of the app
st.title("West Indonesia Birds Audio Classifier 🦜")

# Introduction
st.markdown("""
    **Selamat datang di aplikasi Klasifikasi Suara Burung!**
    Aplikasi ini akan mengklasifikasikan suara burung berdasarkan file audio yang diunggah.
    Cukup unggah file audio dalam format MP3 atau WAV, dan model akan memberikan prediksi kelas burung!
""")

# File upload section
st.header("Unggah File Audio Suara Burung")

uploaded_audio = st.file_uploader("Pilih file audio (MP3/WAV) untuk diuji", type=["mp3", "wav"])

if uploaded_audio is not None:
    # Display file details
    st.audio(uploaded_audio, format="audio/mp3")
    
    # Process the audio file
    # Read audio file using librosa
    audio_bytes = uploaded_audio.read()
    with BytesIO(audio_bytes) as audio_buffer:
        # Load audio using librosa
        y, sr = librosa.load(audio_buffer, sr=None)  # Keep original sample rate

    # Extract Mel-spectrogram and MFCC features
    st.subheader("Mel-spectrogram dan MFCC Extracted Features")

    # Mel-Spectrogram
    mel_spectrogram = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
    mel_db = librosa.power_to_db(mel_spectrogram, ref=np.max)

    # Cek shape dari mel_db sebelum fix_length
    st.write("Original Mel Spectrogram Shape:", mel_db.shape)

    # Resize Mel-spectrogram to a fixed width (e.g., 500)
    mel_db_resized = librosa.util.fix_length(mel_db, size=500, axis=-1)

    # Cek shape setelah fix_length
    st.write("Resized Mel Spectrogram Shape:", mel_db_resized.shape)

    # Plot Mel-spectrogram
    fig, ax = plt.subplots(figsize=(10, 4))
    librosa.display.specshow(mel_db_resized, x_axis='time', y_axis='mel', ax=ax)
    ax.set_title('Mel-spectrogram')
    st.pyplot(fig)

    # MFCC
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    
    # Plot MFCC
    fig, ax = plt.subplots(figsize=(10, 4))
    librosa.display.specshow(mfcc, x_axis='time', ax=ax)
    ax.set_title('MFCC')
    st.pyplot(fig)

    # Prepare features for prediction
    mel_spectrogram = mel_db_resized[..., np.newaxis]  # Add channel dimension
    mfcc = mfcc.T  # Transpose MFCC to match input shape

    # Normalize features
    mel_spectrogram = mel_spectrogram / np.max(mel_spectrogram)
    mfcc = mfcc / np.max(mfcc)

    # Reshape Mel-spectrogram to have 3 channels (required by the model)
    mel_spectrogram = np.repeat(mel_spectrogram, 3, axis=-1)  # Repeat channels 3 times to match the model input

    # Predict using the models
    if st.button('Prediksi Kelas Burung'):
        with st.spinner("Memproses..."):
            # Reshape for model input
            mel_spectrogram = np.expand_dims(mel_spectrogram, axis=0)  # Add batch dimension
            mfcc = np.expand_dims(mfcc, axis=0)  # Add batch dimension
            
            # Predict using the models (Melspec and MFCC models)
            melspec_pred = melspec_model.predict(mel_spectrogram)
            mfcc_pred = mfcc_model.predict(mfcc)

            # Decode predictions
            melspec_pred_class = np.argmax(melspec_pred, axis=1)[0]
            mfcc_pred_class = np.argmax(mfcc_pred, axis=1)[0]

            # Display results
            st.subheader("Hasil Prediksi:")
            st.write(f"**Melspec Model Prediksi:** Kelas {melspec_pred_class}")
            st.write(f"**MFCC Model Prediksi:** Kelas {mfcc_pred_class}")
            
            # Show a confusion matrix (dummy data for the example)
            fig, ax = plt.subplots(figsize=(8, 6))
            cm = np.array([[182, 16, 21, 2, 1, 0],
                           [10, 167, 18, 1, 1, 1],
                           [73, 16, 182, 3, 11, 5],
                           [2, 17, 4, 246, 8, 0],
                           [2, 0, 2, 7, 63, 0],
                           [20, 13, 50, 1, 1, 23]])  # Confusion matrix for demo
            
            sns.heatmap(cm, annot=True, fmt='g', cmap='Blues', ax=ax)
            plt.xlabel("Predicted")
            plt.ylabel("True")
            st.pyplot(fig)

            st.success("Prediksi selesai!")

# Footer
st.markdown("""
    <hr>
    <p style="text-align:center; font-size:12px; color:#555;">Aplikasi ini dibangun menggunakan Streamlit dan TensorFlow. Dataset burung Indonesia diambil dari Kaggle.</p>
    <p style="text-align:center; font-size:12px; color:#555;">Desain oleh <strong>AI Model</strong>.</p>
""", unsafe_allow_html=True)

