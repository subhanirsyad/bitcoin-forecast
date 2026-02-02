from __future__ import annotations

import numpy as np
import pandas as pd

DEFAULT_CSV_URL = "https://drive.google.com/uc?export=download&id=1hpsqSpfjdqIZWqwd259klQSeaNSe5Trr"

INPUT_LEN = 168   # 7 hari jika data per jam
HORIZON = 24      # prediksi 24 step ke depan

class MinMaxScalerPerColumn:
    def __init__(self, eps: float = 1e-8):
        self.eps = float(eps)
        self.min_: pd.Series | None = None
        self.max_: pd.Series | None = None

    def fit(self, df_: pd.DataFrame) -> "MinMaxScalerPerColumn":
        self.min_ = df_.min(axis=0)
        self.max_ = df_.max(axis=0)
        return self

    def transform(self, df_: pd.DataFrame) -> pd.DataFrame:
        if self.min_ is None or self.max_ is None:
            raise ValueError("Scaler belum di-fit.")
        return (df_ - self.min_) / (self.max_ - self.min_ + self.eps)

    def inverse_transform_col(self, arr: np.ndarray, col_name: str) -> np.ndarray:
        if self.min_ is None or self.max_ is None:
            raise ValueError("Scaler belum di-fit.")
        mn = float(self.min_[col_name])
        mx = float(self.max_[col_name])
        return arr * (mx - mn + self.eps) + mn


def _detect_and_set_time_index(df: pd.DataFrame) -> pd.DataFrame:
    time_candidates = [c for c in df.columns if c.lower() in ["date", "datetime", "timestamp", "time", "open_time", "close_time"]]
    if time_candidates:
        time_col = time_candidates[0]
        df[time_col] = pd.to_datetime(df[time_col], errors="coerce", utc=True)
        df = df.dropna(subset=[time_col]).sort_values(time_col).reset_index(drop=True)
        df = df.set_index(time_col)
        return df

    unix_candidates = [c for c in df.columns if c.lower() in ["unix", "epoch"]]
    if unix_candidates:
        ucol = unix_candidates[0]
        df[ucol] = pd.to_datetime(df[ucol], unit="s", errors="coerce", utc=True)
        df = df.dropna(subset=[ucol]).sort_values(ucol).reset_index(drop=True).set_index(ucol)
        return df

    raise ValueError("Tidak menemukan kolom waktu (date/datetime/timestamp). Upload CSV yang punya kolom waktu.")


def _rename_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    rename_map = {}
    for c in df.columns:
        lc = str(c).lower().strip()
        if lc == "close":
            rename_map[c] = "Close"
        elif lc == "open":
            rename_map[c] = "Open"
        elif lc == "high":
            rename_map[c] = "High"
        elif lc == "low":
            rename_map[c] = "Low"
        elif "volume" in lc and "quote" not in lc:
            rename_map[c] = "Volume"
    df = df.rename(columns=rename_map)

    if "Close" not in df.columns:
        close_alt = [c for c in df.columns if str(c).lower().strip() == "close"]
        if close_alt:
            df = df.rename(columns={close_alt[0]: "Close"})
        else:
            raise ValueError("Kolom 'Close' tidak ditemukan. Pastikan CSV punya Close.")
    return df


def preprocess_dataframe(df_raw: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """
    Meniru pipeline dari notebook:
    - Trim kolom
    - Filter BTC bila ada kolom symbol
    - Deteksi index waktu (UTC)
    - Rename OHLCV ke Open/High/Low/Close/Volume
    - Pilih kolom numerik, ffill, rolling FE, dropna
    Mengembalikan: df (index datetime), list fitur (termasuk Close).
    """
    df = df_raw.copy()
    df.columns = [str(c).strip() for c in df.columns]

    symbol_cols = [c for c in df.columns if str(c).lower() in ["symbol", "asset", "coin", "crypto", "name"]]
    if symbol_cols:
        sym_col = symbol_cols[0]
        mask = df[sym_col].astype(str).str.upper().str.contains(r"\bBTC\b|BITCOIN|BTC/|BTCUSDT", regex=True)
        df = df[mask].copy()

    df = _detect_and_set_time_index(df)
    df = _rename_ohlcv(df)

    # Ambil numerik saja
    df = df.select_dtypes(include=[np.number]).copy()
    df = df[~df.index.duplicated(keep="first")].sort_index()
    df = df.ffill()

    candidate_features = ["Open", "High", "Low", "Close", "Volume"]
    features = [c for c in candidate_features if c in df.columns]

    if len(features) < 3:
        features = list(df.columns[:3])
        if "Close" not in features and "Close" in df.columns:
            features = (features[:2] + ["Close"])

    # rolling features (shift(1) agar tidak leakage)
    df["Close_roll_mean_24"]  = df["Close"].shift(1).rolling(window=24).mean()
    df["Close_roll_std_24"]   = df["Close"].shift(1).rolling(window=24).std()
    df["Close_roll_mean_168"] = df["Close"].shift(1).rolling(window=168).mean()
    df["Close_roll_std_168"]  = df["Close"].shift(1).rolling(window=168).std()
    df = df.dropna()

    for new_f in ["Close_roll_mean_24","Close_roll_std_24","Close_roll_mean_168","Close_roll_std_168"]:
        if new_f not in features:
            features.append(new_f)

    if "Close" not in features:
        features.insert(0, "Close")

    return df, features


def time_split(df: pd.DataFrame, train_ratio: float = 0.70, val_ratio: float = 0.15):
    n = len(df)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    return df.iloc[:train_end].copy(), df.iloc[train_end:val_end].copy(), df.iloc[val_end:].copy()


def fit_scaler_and_transform(df: pd.DataFrame, features: list[str]):
    df_train, df_val, df_test = time_split(df)
    scaler = MinMaxScalerPerColumn().fit(df_train[features])
    train_scaled = scaler.transform(df_train[features]).values.astype(np.float32)
    val_scaled   = scaler.transform(df_val[features]).values.astype(np.float32)
    test_scaled  = scaler.transform(df_test[features]).values.astype(np.float32)
    return scaler, (df_train, df_val, df_test), (train_scaled, val_scaled, test_scaled)


def infer_freq(index: pd.DatetimeIndex) -> pd.Timedelta:
    # coba infer, fallback median delta
    if len(index) >= 3:
        diffs = index.to_series().diff().dropna()
        if len(diffs) > 0:
            return diffs.median()
    return pd.Timedelta(hours=1)
