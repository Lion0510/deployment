import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# Nama file CSV untuk menyimpan data
DATA_FILE = "data_penjualan.csv"

# Fungsi untuk membaca data
def load_data():
    try:
        return pd.read_csv(DATA_FILE)
    except FileNotFoundError:
        return pd.DataFrame(columns=["Tanggal", "Varian", "Produksi", "Harga Jual", "Terjual", "Pendapatan"])

# Fungsi untuk menyimpan data
def save_data(df):
    df.to_csv(DATA_FILE, index=False)

# Header aplikasi
st.title("📊 Sistem Pendataan & Visualisasi Penjualan Rice Cake")

# Form input data
with st.form("data_input"):
    tanggal = st.date_input("📅 Tanggal Produksi")
    varian = st.selectbox("🍰 Varian Rice Cake", ["Coklat", "Pandan", "Matcha", "Vanilla"])
    produksi = st.number_input("🏭 Jumlah Produksi", min_value=0, step=1)
    harga = st.number_input("💰 Harga Jual per Unit", min_value=0, step=500)
    terjual = st.number_input("🛒 Jumlah Terjual", min_value=0, step=1)
    submit = st.form_submit_button("Simpan Data")

# Memuat data yang sudah ada
df = load_data()

# Simpan data baru jika ada input
if submit:
    pendapatan = harga * terjual
    new_data = pd.DataFrame([[tanggal, varian, produksi, harga, terjual, pendapatan]],
                            columns=df.columns)
    df = pd.concat([df, new_data], ignore_index=True)
    save_data(df)
    st.success("✅ Data berhasil disimpan!")

# Menampilkan data dalam tabel
st.subheader("📋 Data Penjualan")
st.dataframe(df)

# Visualisasi Data
if not df.empty:
    st.subheader("📊 Grafik Penjualan per Varian")
    fig, ax = plt.subplots()
    df.groupby("Varian")["Terjual"].sum().plot(kind="bar", ax=ax, color=["brown", "green", "red", "blue"])
    ax.set_ylabel("Jumlah Terjual")
    st.pyplot(fig)

    st.subheader("📈 Tren Penjualan")
    fig, ax = plt.subplots()
    df.groupby("Tanggal")["Terjual"].sum().plot(kind="line", marker="o", ax=ax)
    ax.set_ylabel("Jumlah Terjual")
    st.pyplot(fig)
