from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

from preprocessing import (
    DEFAULT_CSV_URL,
    INPUT_LEN,
    HORIZON,
    preprocess_dataframe,
    fit_scaler_and_transform,
    infer_freq,
)
from model_defs import get_custom_objects

st.set_page_config(
    page_title="BTC Forecast (LSTM + Attention)",
    page_icon="📈",
    layout="wide",
)

st.title("📈 Prediksi Harga Bitcoin (LSTM + Attention)")
st.caption("Streamlit demo dari project DLTM (Subhan Irsyad)")

with st.sidebar:
    st.header("Pengaturan")
    data_mode = st.radio("Sumber data", ["Default (link Google Drive)", "Upload CSV"], index=0)
    uploaded = None
    if data_mode == "Upload CSV":
        uploaded = st.file_uploader("Upload CSV", type=["csv"])
        st.caption("CSV perlu punya kolom waktu (date/datetime/timestamp) dan kolom Close.")
    else:
        st.text_input("Default CSV URL", value=DEFAULT_CSV_URL, disabled=True)

    model_choice = st.selectbox(
        "Pilih model",
        [
            ("best_model_seq2seq_LSTM.keras (recommended)", "models/best_model_seq2seq_LSTM.keras"),
            ("model_seq2seq_LSTM.keras", "models/model_seq2seq_LSTM.keras"),
            ("model_baseline_LSTM.keras", "models/model_baseline_LSTM.keras"),
        ],
        index=0,
    )
    show_advanced = st.toggle("Tampilkan advanced", value=False)
    if show_advanced:
        st.write(f"INPUT_LEN = {INPUT_LEN} | HORIZON = {HORIZON}")
        st.caption("Nilai ini mengikuti notebook training.")

@st.cache_data(show_spinner=False)
def load_raw_csv(uploaded_file):
    if uploaded_file is None:
        return pd.read_csv(DEFAULT_CSV_URL)
    return pd.read_csv(uploaded_file)

@st.cache_data(show_spinner=False)
def preprocess_all(df_raw: pd.DataFrame):
    df, features = preprocess_dataframe(df_raw)
    return df, features

@st.cache_resource(show_spinner=False)
def load_tf_model(path: str):
    import tensorflow as tf
    from tensorflow import keras
    model = keras.models.load_model(
        path,
        compile=False,
        custom_objects=get_custom_objects(),
    )
    return model

def plot_history_and_forecast(hist_idx, hist_close, fc_idx, fc_close, title="Forecast"):
    fig = plt.figure(figsize=(10, 4))
    plt.plot(hist_idx, hist_close, label="History (Close)")
    plt.plot(fc_idx, fc_close, label="Forecast (Close)")
    plt.title(title)
    plt.xlabel("Time")
    plt.ylabel("Price")
    plt.xticks(rotation=25)
    plt.legend()
    plt.tight_layout()
    st.pyplot(fig, clear_figure=True)

# ---------------- UI Flow ----------------
try:
    df_raw = load_raw_csv(uploaded)
except Exception as e:
    st.error(f"Gagal load CSV: {e}")
    st.stop()

try:
    df, features = preprocess_all(df_raw)
except Exception as e:
    st.error(f"Gagal preprocess data: {e}")
    st.stop()

colA, colB = st.columns([2, 1], gap="large")

with colA:
    st.subheader("Data")
    st.write(f"Baris setelah preprocessing: **{len(df):,}** | Fitur: **{len(features)}**")
    st.dataframe(df.tail(20), use_container_width=True)

with colB:
    st.subheader("Ringkas")
    st.write("Fitur terpilih:")
    st.code("\n".join(features))
    st.metric("Close terbaru", f"{float(df['Close'].iloc[-1]):,.2f}")
    freq = infer_freq(df.index)
    st.write(f"Perkiraan interval data: **{freq}**")

st.divider()

run = st.button("🚀 Jalankan Forecast", type="primary")

if run:
    if len(df) < (INPUT_LEN + 200):
        st.warning("Data terlalu sedikit untuk window 168 + rolling 168. Upload data yang lebih panjang.")
        st.stop()

    with st.spinner("Menyiapkan scaler & input window..."):
        scaler, (df_train, df_val, df_test), (train_np, val_np, test_np) = fit_scaler_and_transform(df, features)
        # ambil window terakhir untuk inference
        last_block = df[features].iloc[-INPUT_LEN:].copy()
        last_scaled = scaler.transform(last_block).values.astype(np.float32)
        enc_in = np.expand_dims(last_scaled, axis=0)  # [1, INPUT_LEN, F]

    with st.spinner("Memuat model & prediksi... (TensorFlow)"):
        model_path = model_choice[1]
        model = load_tf_model(model_path)

        # Jika model punya infer_autoregressive (seq2seq subclass), pakai itu.
        if hasattr(model, "infer_autoregressive"):
            pred_scaled = model.infer_autoregressive(enc_in, HORIZON, training=False).numpy()
        else:
            # fallback: teacher forcing dengan dec_in zeros
            dec_in = np.zeros((1, HORIZON, 1), dtype=np.float32)
            pred_scaled = model((enc_in, dec_in), training=False).numpy()

        pred_scaled = pred_scaled.reshape(-1)
        pred_close = scaler.inverse_transform_col(pred_scaled, "Close")

    # buat index forecast
    freq = infer_freq(df.index)
    last_t = df.index[-1]
    fc_idx = [last_t + (i + 1) * freq for i in range(HORIZON)]
    fc_idx = pd.DatetimeIndex(fc_idx)

    # plot
    hist_view = df["Close"].iloc[-INPUT_LEN:].astype(float)
    plot_history_and_forecast(hist_view.index, hist_view.values, fc_idx, pred_close, title="BTC Close Forecast")

    # tabel output
    out_df = pd.DataFrame({"timestamp": fc_idx, "pred_close": pred_close})
    st.subheader("Hasil Forecast")
    st.dataframe(out_df, use_container_width=True)

    # download
    csv_bytes = out_df.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Download forecast.csv", data=csv_bytes, file_name="forecast.csv", mime="text/csv")

with st.expander("📌 Cara deploy ke Streamlit Community Cloud"):
    st.markdown(
        """
1) Buat repo GitHub baru, lalu upload isi folder project ini (atau `git push`).
2) Buka Streamlit Community Cloud, klik **Create app** dan pilih repo + entrypoint **`streamlit_app.py`**.
3) (Opsional) Atur **App URL** (subdomain) biar link-nya sesuai keinginanmu.
4) Klik Deploy.

Catatan: App ini pakai TensorFlow, jadi pertama kali deploy biasanya butuh beberapa menit buat install dependency.
        """.strip()
    )
