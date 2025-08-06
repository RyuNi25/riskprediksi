# app.py
import streamlit as st
import pickle
import pandas as pd
import matplotlib.pyplot as plt

# Load model dan fitur
with open('model_risiko_random_forest.pkl', 'rb') as f:
    model = pickle.load(f)

with open('fitur_model.pkl', 'rb') as f:
    fitur = pickle.load(f)

# Title dan deskripsi
st.title("🎯 Prediksi Risiko Kredit Nasabah - PT PNM")
st.markdown("Masukkan data nasabah untuk memprediksi risiko kredit. Model ini menggunakan algoritma **Random Forest** yang dilatih dari data historis nasabah.")

st.markdown("---")

# Input data interaktif
data_input = {}
for kolom in fitur:
    if "kategori" in kolom.lower() or "risiko" in kolom.lower():
        data_input[kolom] = st.selectbox(f"{kolom}", [0, 1])
    else:
        data_input[kolom] = st.number_input(f"{kolom}", min_value=0.0)

# Buat DataFrame dari input
input_df = pd.DataFrame([data_input])

# Prediksi dan tampilan hasil
if st.button("🔍 Prediksi"):
    prediksi = model.predict(input_df)[0]
    proba = model.predict_proba(input_df)[0]

    # Tampilkan hasil
    st.success(f"Hasil Prediksi: **{prediksi.upper()}**")

    # Penjelasan tambahan
    if prediksi.lower() == "rendah":
        st.info("✅ Nasabah ini termasuk ke dalam kategori risiko rendah. Dapat dipertimbangkan untuk diberikan pinjaman.")
    else:
        st.warning("⚠️ Nasabah ini memiliki risiko tinggi. Perlu dilakukan analisa lebih lanjut sebelum pemberian pinjaman.")

    # Visualisasi probabilitas prediksi
    st.markdown("### Probabilitas Klasifikasi:")
    fig, ax = plt.subplots()
    ax.bar(model.classes_, proba, color=["green", "red"])
    ax.set_ylabel("Probabilitas")
    ax.set_ylim(0, 1)
    st.pyplot(fig)

    # Simpan riwayat prediksi ke CSV
    input_df["prediksi"] = prediksi
    input_df.to_csv("riwayat_prediksi.csv", mode='a', index=False, header=False)

st.markdown("---")
st.caption("Made with ❤️ by RESTI AYU ANJANI - KP PT PNM Padang 2025")
