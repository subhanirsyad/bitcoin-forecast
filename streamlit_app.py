from __future__ import annotations

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from preprocessing import (
    DEFAULT_CSV_URL,
    INPUT_LEN,
    HORIZON,
    preprocess_dataframe,
    fit_scaler_and_transform,
    infer_freq,
)

# Akurasi di data test (sel 19 notebook, dihitung ulang dalam USDT): MAE rata-rata 24 jam ke depan.
MODELS = {
    "Seq2Seq LSTM + Attention": {
        "path": "models/best_model_seq2seq_LSTM.keras",
        "mae_usd": 241.69,
        "how": "Encoder-decoder dengan attention. Memprediksi jam demi jam, tiap jam memakai hasil jam sebelumnya.",
    },
    "Baseline LSTM": {
        "path": "models/model_baseline_LSTM.keras",
        "mae_usd": 238.97,
        "how": "LSTM biasa yang mengeluarkan 24 jam sekaligus. Dipakai sebagai pembanding.",
    },
}
NAIVE_MAE_USD = 237.71
TRAIN_PERIOD = "28 Sep 2017 sampai 26 Des 2021"
TEST_PERIOD = "22 Nov 2022 sampai 19 Okt 2023"
TEST_WINDOWS = 7757
MIN_ROWS = INPUT_LEN + 200
BULAN = ["Jan", "Feb", "Mar", "Apr", "Mei", "Jun", "Jul", "Agu", "Sep", "Okt", "Nov", "Des"]
HIST = "#CFCAC2"
ACCENT = "#F7931A"
MONTH_EXPR = "date(datum.value) + ' ' + " + str(BULAN) + "[month(datum.value)]"


def fmt_time(t: pd.Timestamp) -> str:
    return f"{t.day} {BULAN[t.month - 1]} {t.year}, {t:%H:%M} UTC"


def fmt_date(t: pd.Timestamp) -> str:
    return f"{t.day} {BULAN[t.month - 1]} {t.year}"


st.set_page_config(page_title="Prediksi Bitcoin 24 Jam", page_icon="📈", layout="centered")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Archivo:wght@600;700&family=IBM+Plex+Mono:wght@500&display=swap');

#MainMenu, footer, [data-testid="stHeader"] { display: none; }
.block-container { max-width: 1000px; padding-top: 2.5rem; padding-bottom: 4rem; }

h1, h2, h3 { font-family: 'Archivo', sans-serif !important; letter-spacing: -0.015em; }
h1 { font-size: clamp(2rem, 5vw, 2.9rem) !important; font-weight: 700 !important; padding-bottom: .3rem !important; }
h2 { font-size: 1.45rem !important; font-weight: 600 !important; margin-top: 1rem; }

.lede { font-size: 1.08rem; line-height: 1.6; color: #C9C5BE; max-width: 640px; margin: 0 0 .35rem; }
.fine { font-size: .85rem; color: #A3A09A; margin: 0 0 1.75rem; }

/* Strip angka utama: tiga angka yang jadi inti layar ini. */
.quote {
    display: grid; grid-template-columns: repeat(3, 1fr);
    background: #17191C; border: 1px solid #26292E; border-radius: 12px; margin: 1.25rem 0 1rem;
}
.q { padding: 16px 20px; }
.q + .q { border-left: 1px solid #26292E; }
.q-label { font-size: .82rem; color: #A3A09A; margin-bottom: 4px; }
.q-num {
    font-family: 'IBM Plex Mono', ui-monospace, monospace; font-weight: 500;
    font-size: clamp(1.45rem, 3.6vw, 2rem); line-height: 1.2; font-variant-numeric: tabular-nums; color: #ECEAE6;
}
.q-num.accent { color: #F7931A; }
.q-num.down { color: #F2787A; }
.q-num.up { color: #5BD08D; }
.q-wait { font-size: 1rem; color: #A3A09A; line-height: 2.4rem; }
.q-sub { font-size: .8rem; color: #A3A09A; margin-top: 4px; }
@media (max-width: 640px) {
    .quote { grid-template-columns: 1fr; }
    .q + .q { border-left: none; border-top: 1px solid #26292E; }
}

/* Angka prediksi muncul sekali saat hasil baru keluar, sebagai tanda ada hasil baru. */
@keyframes rise { from { opacity: 0; transform: translateY(6px); } to { opacity: 1; transform: none; } }
.reveal { animation: rise .6s cubic-bezier(.2, .7, .2, 1) both; }
.reveal.later { animation-delay: .12s; }
@media (prefers-reduced-motion: reduce) { .reveal { animation: none; } }

/* Tombol utama: teks gelap agar kontras di atas oranye, glow tipis hanya saat hover. */
[data-testid="stBaseButton-primary"] {
    color: #1A1206 !important; font-weight: 600;
    transition: transform .15s ease, box-shadow .2s ease;
}
[data-testid="stBaseButton-primary"] p { color: #1A1206 !important; }
[data-testid="stBaseButton-primary"]:hover:not(:disabled) {
    transform: translateY(-1px);
    box-shadow: 0 0 0 4px rgba(247, 147, 26, .16), 0 8px 24px rgba(247, 147, 26, .22);
}
[data-testid="stBaseButton-primary"]:active:not(:disabled) { transform: translateY(0); }
[data-testid="stBaseButton-primary"]:focus-visible { outline: 2px solid #ECEAE6; outline-offset: 3px; }

.legend { display: flex; gap: 18px; font-size: .82rem; color: #A3A09A; margin: 0 0 -6px 4px; }
.legend i { display: inline-block; width: 18px; height: 0; border-top: 2px solid; vertical-align: middle; margin-right: 6px; }
hr.rule { border: none; border-top: 1px solid #26292E; margin: 2.25rem 0 1rem; }
</style>
""", unsafe_allow_html=True)


@st.cache_data(show_spinner=False)
def load_raw_csv(uploaded_file):
    if uploaded_file is None:
        return pd.read_csv(DEFAULT_CSV_URL)
    return pd.read_csv(uploaded_file)


@st.cache_data(show_spinner=False)
def preprocess_all(df_raw: pd.DataFrame):
    return preprocess_dataframe(df_raw)


@st.cache_resource(show_spinner=False)
def load_tf_model(path: str):
    # TensorFlow di-import di sini supaya halaman tampil dulu sebelum TF selesai dimuat.
    from tensorflow import keras
    from model_defs import get_custom_objects
    return keras.models.load_model(path, compile=False, custom_objects=get_custom_objects())


def run_forecast(df: pd.DataFrame, features: list[str], model_path: str) -> pd.DataFrame:
    scaler, _, _ = fit_scaler_and_transform(df, features)
    last_block = df[features].iloc[-INPUT_LEN:]
    enc_in = np.expand_dims(scaler.transform(last_block).values.astype(np.float32), axis=0)  # [1, INPUT_LEN, F]

    model = load_tf_model(model_path)
    if hasattr(model, "infer_autoregressive"):
        pred_pct = model.infer_autoregressive(enc_in, HORIZON, training=False).numpy()
    else:
        # baseline LSTM: satu input (window encoder) -> [1, HORIZON, 1]
        pred_pct = model(enc_in, training=False).numpy()

    # Output model = % log-return tiap jam dibanding harga terakhir (lihat notebook, sel 8).
    last_close = float(df["Close"].iloc[-1])
    pred_close = last_close * np.exp(pred_pct.reshape(-1) / 100.0)

    freq = infer_freq(df.index)
    last_t = df.index[-1]
    return pd.DataFrame({
        "time": [last_t + (i + 1) * freq for i in range(HORIZON)],
        "close": pred_close.astype(float),
        "change": (pred_close / last_close - 1) * 100,
    })


def price_chart(df: pd.DataFrame, fc: pd.DataFrame | None) -> alt.Chart:
    # Timestamp dibuat naive (tetap UTC) supaya browser tidak menggesernya ke zona waktu lokal.
    hist = df["Close"].iloc[-INPUT_LEN:].rename("close").reset_index()
    hist.columns = ["time", "close"]
    hist["time"] = hist["time"].dt.tz_localize(None)
    hist["seri"] = "Harga asli"
    parts = [hist]
    if fc is not None:
        # Garis prediksi disambung dari titik historis terakhir agar tidak terputus.
        f = pd.concat([hist.tail(1)[["time", "close"]], fc[["time", "close"]].assign(time=fc["time"].dt.tz_localize(None))])
        f["seri"] = "Prediksi"
        parts.append(f)
    data = pd.concat(parts, ignore_index=True)
    data["label"] = data["time"].map(fmt_time)

    lo, hi = data["close"].min(), data["close"].max()
    pad = (hi - lo) * 0.08
    x = alt.X("time:T", title=None, axis=alt.Axis(labelExpr=MONTH_EXPR, labelAngle=0, tickCount=8))
    y = alt.Y("close:Q", title=None, scale=alt.Scale(domain=[lo - pad, hi + pad], nice=False), axis=alt.Axis(format=",.0f"))
    colors = alt.Scale(domain=["Harga asli", "Prediksi"], range=[HIST, ACCENT])

    # Gradasi di bawah harga asli: konteks dibaca sebagai latar, prediksi jadi garis paling terang.
    area = alt.Chart(hist).mark_area(
        line={"color": HIST, "strokeWidth": 1.6},
        color=alt.Gradient(gradient="linear", x1=1, x2=1, y1=0, y2=1, stops=[
            alt.GradientStop(color="rgba(207,202,194,0.22)", offset=0),
            alt.GradientStop(color="rgba(207,202,194,0)", offset=1),
        ]),
    ).encode(x=x, y=y, y2=alt.datum(lo - pad))
    layers = [area]
    if fc is not None:
        pred = alt.Chart(data[data["seri"] == "Prediksi"])
        layers += [
            pred.mark_line(color=ACCENT, strokeWidth=9, opacity=0.22, strokeCap="round").encode(x=x, y=y),  # glow
            pred.mark_line(color=ACCENT, strokeWidth=2.2, strokeDash=[6, 3]).encode(x=x, y=y),
        ]

    hover = alt.selection_point(fields=["time"], nearest=True, on="pointerover", clear="pointerout", empty=False)
    dots = alt.Chart(data).mark_circle(size=70, stroke="#0F1012", strokeWidth=2).encode(
        x=x, y=y, color=alt.Color("seri:N", scale=colors, legend=None),
        opacity=alt.condition(hover, alt.value(1), alt.value(0)),
        tooltip=[alt.Tooltip("seri:N", title="Seri"),
                 alt.Tooltip("label:N", title="Waktu"),
                 alt.Tooltip("close:Q", title="Harga (USDT)", format=",.2f")],
    ).add_params(hover)
    crosshair = alt.Chart(data).mark_rule(color="#6B6862").encode(x=x).transform_filter(hover)
    now = alt.Chart(pd.DataFrame({"time": [hist["time"].iloc[-1]]})).mark_rule(color="#4A4D52", strokeDash=[2, 3]).encode(x="time:T")

    return (
        alt.layer(*layers, now, crosshair, dots)
        .properties(height=380)
        .configure_view(strokeWidth=0)
        .configure_axis(gridColor="#23262A", domainColor="#2E3136", tickColor="#2E3136",
                        labelColor="#A3A09A", labelFontSize=11)
    )


def render_about(df: pd.DataFrame | None, features: list[str] | None):
    st.markdown('<hr class="rule">', unsafe_allow_html=True)
    st.header("Tentang model dan data")

    left, right = st.columns([3, 2], gap="large")
    with left:
        st.markdown("**Model yang tersedia**")
        for name, m in MODELS.items():
            st.markdown(f"- **{name}**. {m['how']}")
        st.markdown("**Seberapa akurat?**")
        rows = [("Tebakan naif (harga tidak berubah)", NAIVE_MAE_USD)] + [(n, m["mae_usd"]) for n, m in MODELS.items()]
        st.dataframe(
            pd.DataFrame({
                "Model": [r[0] for r in rows],
                "Rata-rata meleset (USDT)": [r[1] for r in rows],
                "Dibanding tebakan naif": [r[1] / NAIVE_MAE_USD for r in rows],
            }),
            hide_index=True, width="stretch",
            column_config={
                "Rata-rata meleset (USDT)": st.column_config.NumberColumn(format="%.0f"),
                "Dibanding tebakan naif": st.column_config.NumberColumn(format="%.2fx"),
            },
        )
        st.caption(
            f"Rata-rata selisih harga prediksi dan harga asli untuk 24 jam ke depan, di data test "
            f"{TEST_PERIOD} ({TEST_WINDOWS:,} window) yang tidak dipakai saat training."
        )
        st.markdown("**Batasan**")
        st.markdown(
            "- Di data test, kedua model hampir sama akuratnya dengan tebakan naif \"harga tidak berubah\", "
            "jadi garis prediksi belum bisa dipakai sebagai sinyal beli atau jual.\n"
            "- Model hanya membaca pola harga dan indikator di data. Berita, regulasi, dan sentimen pasar tidak ikut dihitung.\n"
            f"- Model dilatih dengan data {TRAIN_PERIOD}. Pola harga di luar rentang itu belum pernah dilihat model."
        )
    with right:
        if df is not None:
            st.markdown("**Data yang dipakai**")
            st.markdown(
                f"- {len(df):,} baris per jam\n"
                f"- {fmt_date(df.index[0])} sampai {fmt_date(df.index[-1])}\n"
                f"- Fitur: {', '.join(features)}"
            )
            with st.expander("Lihat 20 baris terakhir"):
                st.dataframe(df.tail(20), width="stretch")


# ---------------- Halaman ----------------
st.title("Bitcoin, 24 jam ke depan")
st.markdown(
    f'<p class="lede">Model LSTM membaca {INPUT_LEN} jam terakhir ({INPUT_LEN // 24} hari) harga penutupan '
    f'BTC/USDT, lalu memperkirakan harga untuk {HORIZON} jam berikutnya.</p>'
    '<p class="fine">Proyek data science untuk portofolio, bukan saran keuangan.</p>',
    unsafe_allow_html=True,
)

c1, c2, c3 = st.columns([1.1, 1.2, 0.8], gap="medium", vertical_alignment="bottom")
with c1:
    source = st.radio("Sumber data", ["Dataset bawaan", "Upload CSV"], horizontal=True)
with c2:
    model_name = st.selectbox("Model", list(MODELS))
btn_slot = c3.empty()
BTN_LABEL = f"Prediksi {HORIZON} jam"


def stop_without_data(df=None, features=None):
    # Tombol tetap terlihat tapi nonaktif selama belum ada data yang bisa diprediksi.
    btn_slot.button(BTN_LABEL, type="primary", width="stretch", disabled=True)
    render_about(df, features)
    st.stop()


uploaded = None
if source == "Upload CSV":
    uploaded = st.file_uploader(
        "File CSV harga per jam", type=["csv"],
        help="Wajib ada kolom waktu (date, datetime, atau timestamp) dan kolom Close. Open, High, Low, Volume dipakai kalau ada.",
    )
    if uploaded is None:
        st.info(f"Belum ada file. Upload CSV per jam dengan kolom waktu dan kolom Close, minimal {MIN_ROWS} baris.")
        stop_without_data()

try:
    with st.spinner("Memuat data harga..."):
        df, features = preprocess_all(load_raw_csv(uploaded))
except Exception as e:
    hint = ("Coba muat ulang halaman. Kalau tetap gagal, Google Drive mungkin sedang membatasi unduhan, jadi pakai Upload CSV."
            if uploaded is None else "Periksa formatnya: perlu kolom waktu (date/datetime/timestamp) dan kolom Close.")
    st.error(f"Data tidak bisa dimuat ({e}). {hint}")
    stop_without_data()

if len(df) < MIN_ROWS:
    st.warning(f"Data hanya {len(df):,} baris setelah dibersihkan. Model butuh minimal {MIN_ROWS} baris per jam, jadi upload data yang lebih panjang.")
    stop_without_data(df, features)

last_t = df.index[-1]
last_close = float(df["Close"].iloc[-1])
data_key = uploaded.file_id if uploaded is not None else "default"

if btn_slot.button(BTN_LABEL, type="primary", width="stretch"):
    try:
        with st.spinner("Menjalankan model. Pertama kali bisa lebih lama karena TensorFlow perlu dimuat."):
            st.session_state.forecast = (data_key, model_name, run_forecast(df, features, MODELS[model_name]["path"]))
    except Exception as e:
        st.error(f"Model gagal berjalan ({e}). Coba pilih model lain atau muat ulang halaman.")

saved = st.session_state.get("forecast")
fc = saved[2] if saved and saved[:2] == (data_key, model_name) else None

if fc is None:
    pred_html = '<div class="q-wait">Belum dijalankan</div>'
    pred_sub = f"tekan {BTN_LABEL}"
    change_html = '<div class="q-wait">Belum ada</div>'
else:
    end = fc.iloc[-1]
    arrow, cls = ("▼", "down") if end["change"] < 0 else ("▲", "up")
    pred_html = f'<div class="q-num accent reveal">{end["close"]:,.2f}</div>'
    pred_sub = f"USDT · {fmt_time(end['time'])}"
    change_html = f'<div class="q-num {cls} reveal later">{arrow} {abs(end["change"]):.2f}%</div>'

st.markdown(f"""
<div class="quote">
    <div class="q">
        <div class="q-label">Harga terakhir di data</div>
        <div class="q-num">{last_close:,.2f}</div>
        <div class="q-sub">USDT · {fmt_time(last_t)}</div>
    </div>
    <div class="q">
        <div class="q-label">Prediksi +{HORIZON} jam</div>
        {pred_html}
        <div class="q-sub">{pred_sub}</div>
    </div>
    <div class="q">
        <div class="q-label">Perubahan</div>
        {change_html}
        <div class="q-sub">{model_name}</div>
    </div>
</div>
""", unsafe_allow_html=True)

legend = f'<span><i style="border-color:{HIST}"></i>Harga asli</span>'
if fc is not None:
    legend += f'<span><i style="border-color:{ACCENT}; border-top-style:dashed"></i>Prediksi</span>'
st.markdown(f'<div class="legend">{legend}</div>', unsafe_allow_html=True)
st.altair_chart(price_chart(df, fc), width="stretch", theme=None)
if fc is None:
    st.caption(f"Grafik menunjukkan {INPUT_LEN} jam terakhir. Garis prediksi muncul setelah kamu menekan {BTN_LABEL}.")
else:
    st.caption("Arahkan kursor ke grafik untuk melihat harga per jam. Garis titik-titik vertikal menandai data terakhir.")

    st.subheader("Prediksi per jam")
    table = pd.DataFrame({
        "Waktu (UTC)": fc["time"].map(lambda t: fmt_time(t).removesuffix(" UTC")),
        "Prediksi (USDT)": fc["close"],
        "Selisih": fc["change"],
    })
    st.dataframe(
        table, hide_index=True, width="stretch",
        column_config={
            "Prediksi (USDT)": st.column_config.NumberColumn(format="%.2f"),
            "Selisih": st.column_config.NumberColumn(format="%+.2f%%", help="Dibanding harga terakhir di data"),
        },
    )
    st.download_button(
        "Unduh prediksi (CSV)", data=table.to_csv(index=False).encode("utf-8"),
        file_name="btc_forecast_24h.csv", mime="text/csv", on_click="ignore",
    )

render_about(df, features)
