# Aplikasi Prediksi Sewa Hunian (Streamlit)

Model di aplikasi ini adalah model terbaik dari Notebook.ipynb (Random Forest, split berbasis waktu),
yaitu model yang sama dengan yang dilaporkan di naskah.

## Cara menjalankan
1. Jalankan Notebook.ipynb sampai selesai (di Colab atau lokal). Notebook menghasilkan `model_sewa.joblib`.
2. Salin `model_sewa.joblib` ke folder ini.
3. `pip install -r requirements.txt`
4. `streamlit run app.py`

## Deploy (Streamlit Community Cloud)
1. Unggah folder ini (app.py, requirements.txt, model_sewa.joblib) ke repositori GitHub.
2. Buka https://share.streamlit.io, pilih repositori, file utama `app.py`, lalu Deploy.
3. Samakan versi scikit-learn di requirements.txt dengan versi yang dicetak notebook,
   supaya model_sewa.joblib bisa dibuka.
