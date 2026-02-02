# 📈 BTC Forecast — Streamlit App (LSTM + Attention)

![Python](https://img.shields.io/badge/Python-3.11%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.30%2B-ff4b4b)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.16%2B-orange)

Aplikasi **Streamlit** untuk demo **prediksi harga penutupan (Close) Bitcoin** menggunakan model **LSTM Seq2Seq + Attention** (pre-trained). Repo ini sudah siap dipush ke GitHub dan dideploy ke **Streamlit Community Cloud**.

> ⚠️ **Disclaimer:** Ini hanya demo/data science project, **bukan saran finansial**.

## Fitur
- Sumber data fleksibel: **default CSV** (link publik) atau **upload CSV**
- Pilih 3 model pre-trained:
  - `best_model_seq2seq_LSTM.keras` (recommended)
  - `model_seq2seq_LSTM.keras`
  - `model_baseline_LSTM.keras`
- Forecast **24 step ke depan** berdasarkan **window 168 step terakhir**
- Visualisasi grafik + tabel hasil
- Export hasil prediksi ke `forecast.csv`

## Jalankan secara lokal

### 1) Siapkan environment
Disarankan pakai virtual environment.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
```

### 2) Jalankan Streamlit
```bash
streamlit run streamlit_app.py
```

### 3) Cara pakai
1. Pilih sumber data (default / upload CSV)
2. Pilih model
3. Klik **🚀 Jalankan Forecast**
4. Unduh hasil prediksi jika diperlukan

## Format CSV yang didukung
Agar preprocessing berhasil, CSV minimal harus punya:
- **Kolom waktu** (salah satu): `date`, `datetime`, `timestamp`, `time`, `open_time`, `close_time`
  - atau kolom unix time: `unix` / `epoch` (detik)
- **Kolom `Close`** (case-insensitive)

Kolom tambahan seperti `Open`, `High`, `Low`, `Volume` akan otomatis dipakai jika tersedia.

## Deploy ke Streamlit Community Cloud
1. Buat repo GitHub baru, lalu push semua isi folder project ini.
2. Buka Streamlit Community Cloud → **Create app**
3. Pilih:
   - **Repository**: repo kamu
   - **Branch**: `main`
   - **Main file path (Entrypoint)**: `streamlit_app.py`
4. Klik **Deploy**

Catatan:
- Repo ini sudah menyertakan `requirements.txt` dan `runtime.txt` (Python 3.11) agar lebih stabil saat deploy.
- Pertama kali deploy bisa agak lama karena install **TensorFlow**.

## Struktur project
```text
.
├─ streamlit_app.py            # Entrypoint Streamlit
├─ preprocessing.py            # Preprocessing + scaler
├─ model_defs.py               # Custom layers/model untuk load .keras
├─ models/                     # Model pre-trained (.keras)
├─ notebook.ipynb              # Notebook training/eksperimen (referensi)
├─ requirements.txt            # Dependency Python
├─ runtime.txt                 # Pin Python version (Streamlit Cloud)
└─ .streamlit/config.toml      # Konfigurasi tema Streamlit
```

## Troubleshooting singkat
- **Error kolom waktu / Close tidak ditemukan** → pastikan CSV punya kolom waktu + `Close`.
- **Data terlalu sedikit** → app butuh data cukup panjang untuk rolling feature (24 & 168) + window input.
- **TensorFlow susah ter-install di lokal** → coba gunakan Python 3.11 dan upgrade pip:
  ```bash
  python -m pip install --upgrade pip
  ```

## Lisensi
MIT — lihat file `LICENSE`.

---

Made by **Subhan Irsyad**
