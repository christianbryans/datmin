# Analisis Data Lagu Spotify 2024

Dashboard interaktif untuk analisis data lagu Spotify menggunakan Streamlit.

## Fitur
- Analisis Clustering dengan K-Means
- Klasifikasi dengan Logistic Regression
- Visualisasi interaktif
- Statistik deskriptif

## Cara Menjalankan Lokal
1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Jalankan aplikasi:
```bash
streamlit run datmin.py
```

## Cara Deploy ke Streamlit Cloud
1. Buat akun di [Streamlit Cloud](https://streamlit.io/cloud)
2. Hubungkan dengan repository GitHub
3. Pilih repository dan file `datmin.py`
4. Deploy

## Struktur File
- `datmin.py`: Kode utama aplikasi
- `requirements.txt`: Daftar dependencies
- `Most Streamed Spotify Songs 2024.csv`: Dataset
- `output_charts/`: Folder untuk menyimpan visualisasi 