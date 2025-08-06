
import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt

# Load model dan fitur (ubah path jika diperlukan)
with open("model_risiko_random_forest.pkl", "rb") as f:
    model = pickle.load(f)

with open("fitur_model.pkl", "rb") as f:
    fitur_model = pickle.load(f)

# Judul
st.title("📊 Prediksi Risiko Kredit Nasabah – PT PNM Cabang Padang")

# Layout 2 kolom
col1, col2 = st.columns([2, 3])

with col1:
    st.subheader("📝 Input Data Nasabah")
    nama = st.text_input("Nama")
    product_id = st.text_input("Product ID")
    disbursed_amount = st.number_input("Disbursed Amount", min_value=0.0)
    total_tunggakan = st.number_input("Total Tunggakan", min_value=0.0)
    mopt = st.number_input("MoPT (Masa Pemakaian Tunai)", min_value=0.0)

    if st.button("🔍 Prediksi"):
        # Buat dataframe input
        input_data = pd.DataFrame([{
            "ProductID": product_id,
            "DisbursedAmount": disbursed_amount,
            "TotalTunggakan": total_tunggakan,
            "MoPT": mopt
        }])

        # Pastikan kolom sesuai urutan fitur model
        input_data = input_data[fitur_model]

        # Prediksi
        prediksi = model.predict(input_data)[0]
        hasil_prediksi = "Risiko Tinggi" if prediksi == 1 else "Risiko Rendah"

        st.success(f"Hasil Prediksi: {hasil_prediksi}")

with col2:
    st.subheader("📈 Visualisasi Data Historis")

    # Contoh visualisasi dummy - bisa diganti dengan data asli
    st.markdown("### Distribusi Nasabah per Kategori Risiko")
    distribusi_data = pd.DataFrame({
        "Kategori": ["Rendah", "Tinggi"],
        "Jumlah": [180, 70]
    })
    fig1, ax1 = plt.subplots()
    ax1.bar(distribusi_data["Kategori"], distribusi_data["Jumlah"])
    ax1.set_ylabel("Jumlah Nasabah")
    st.pyplot(fig1)

    st.markdown("### Tren Penyaluran Pinjaman")
    tren_data = pd.DataFrame({
        "Bulan": pd.date_range(start="2024-01", periods=6, freq="M"),
        "Jumlah": [120, 135, 150, 160, 170, 165]
    })
    tren_data.set_index("Bulan", inplace=True)
    st.line_chart(tren_data)
