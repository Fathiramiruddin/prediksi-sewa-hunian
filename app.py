"""
Aplikasi Streamlit untuk prediksi harga sewa rumah.

Jalankan:  streamlit run app.py
"""
import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Prediksi Sewa Rumah", page_icon="🏠", layout="centered")


@st.cache_resource
def muat_model():
    return joblib.load("model_sewa.joblib")


paket = muat_model()
model, pilihan = paket["model"], paket["pilihan"]

st.title("🏠 Prediksi Harga Sewa Rumah")
m = paket["metrik_uji"]
st.caption(f"Model {paket['nama_model']} (sama dengan naskah) dilatih pada House Rent Prediction Dataset "
           f"(Kaggle, 6 kota India) dengan split berbasis waktu. Data uji: MAE ≈ ₹{m['MAE']:,.0f}, "
           f"R² = {m['R2']:.2f}, MAPE = {m['MAPE (%)']:.2f}%.")

with st.form("form_rumah"):
    c1, c2 = st.columns(2)
    with c1:
        city = st.selectbox("Kota", pilihan["City"])
        bhk = st.number_input("BHK (kamar tidur + hall + dapur)", 1, 6, 2)
        size = st.number_input("Luas (sq ft)", 100, 8000, 900, step=50)
        bathroom = st.number_input("Jumlah kamar mandi", 1, 10, 2)
        area_type = st.selectbox("Tipe luas", pilihan["Area Type"],
                                 index=pilihan["Area Type"].index("Super Area"))
    with c2:
        furnishing = st.selectbox("Status perabot", pilihan["Furnishing Status"],
                                  index=pilihan["Furnishing Status"].index("Semi-Furnished"))
        tenant = st.selectbox("Penyewa yang diutamakan", pilihan["Tenant Preferred"])
        contact = st.selectbox("Kontak", pilihan["Point of Contact"],
                               index=pilihan["Point of Contact"].index("Contact Owner"))
        total_floors = st.number_input("Total lantai gedung", 0, 100, 4)
        floor_no = st.number_input("Unit di lantai ke- (0 = ground)", -2, 100, 1)
    kirim = st.form_submit_button("Prediksi sewa", use_container_width=True)

if kirim:
    if floor_no > total_floors:
        st.warning("Lantai unit lebih tinggi dari total lantai gedung. Mohon periksa lagi.")
    data = pd.DataFrame([{
        "BHK": bhk, "Size": size, "Bathroom": bathroom,
        "FloorNo": floor_no, "TotalFloors": total_floors,
        "City": city, "Furnishing Status": furnishing, "Tenant Preferred": tenant,
        "Area Type": area_type, "Point of Contact": contact,
    }])[paket["fitur"]]
    sewa = float(np.exp(model.predict(data)[0]))

    st.subheader("Hasil prediksi")
    st.metric("Perkiraan sewa per bulan", f"₹ {sewa:,.0f}")
    p = m["MAPE (%)"] / 100
    st.write(f"Kisaran perkiraan (± MAPE {m['MAPE (%)']:.0f}%): "
             f"**₹ {sewa * (1 - p):,.0f} – ₹ {sewa * (1 + p):,.0f}**")
    st.write(f"Setara ≈ ₹ {sewa / size:,.1f} per sq ft.")
    with st.expander("Data yang dimasukkan"):
        st.dataframe(data, hide_index=True)

st.divider()
st.caption("Catatan: prediksi bersifat perkiraan untuk data tahun 2022 dan tidak memakai lokasi "
           "spesifik (Area Locality), sehingga unit mewah bisa meleset cukup jauh.")
