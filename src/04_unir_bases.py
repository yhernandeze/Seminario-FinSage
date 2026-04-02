import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import MinMaxScaler

# ── Rutas ──────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PATH_BD     = os.path.join(BASE_DIR, 'data', 'raw', 'BD.parquet2')
PATH_SENT   = os.path.join(BASE_DIR, 'data', 'processed', 'salida_sentimientos_finbert.csv')
OUTPUT_PATH = os.path.join(BASE_DIR, 'data', 'processed', 'dataset_final.parquet')
OUTPUT_CSV  = os.path.join(BASE_DIR, 'data', 'processed', 'dataset_final.csv')

# ── Features ───────────────────────────────────────────────────────
FEATURES_TECNICOS = [
    "open", "high", "low", "close", "adj_close", "volume",
    "ema_9", "ema_21", "sma_50",
    "macd", "macd_signal", "macd_hist", "retorno_log",
    "rsi_14", "stoch_k", "stoch_d", "roc_10", "vol_ratio",
    "volatilidad_20", "atr_14", "bb_upper", "bb_mid", "bb_lower", "bb_width",
]
FEATURES_SENTIMIENTO = ["sent_pos", "sent_neg", "sent_neu"]
ALL_FEATURES = FEATURES_TECNICOS + FEATURES_SENTIMIENTO


# ══════════════════════════════════════════════════════════════════
#  PASO 1 — Cargar archivos
# ══════════════════════════════════════════════════════════════════

def cargar_datos():
    print("=" * 55)
    print("  PASO 1 — Cargando archivos")
    print("=" * 55)

    bd = pd.read_parquet(PATH_BD)
    bd["dt"] = pd.to_datetime(bd["dt"])
    print(f"  BD.parquet2        : {len(bd):,} filas · {bd['ticker'].nunique()} tickers")
    print(f"  Rango de fechas    : {bd['dt'].min().date()} → {bd['dt'].max().date()}")

    sent = pd.read_csv(PATH_SENT)
    print(f"\n  FinBERT CSV        : {len(sent):,} filas · {sent['ticker'].nunique()} tickers")
    print(f"  Tickers con noticia: {sorted(sent['ticker'].unique())}")

    return bd, sent


# ══════════════════════════════════════════════════════════════════
#  PASO 2 — Calcular promedio de sentimiento por ticker
# ══════════════════════════════════════════════════════════════════

def preparar_sentimiento(sent: pd.DataFrame) -> pd.DataFrame:
    print("\n" + "=" * 55)
    print("  PASO 2 — Promedio de sentimiento por ticker")
    print("=" * 55)

    # Detectar columnas automáticamente
    col_pos = next((c for c in sent.columns if 'pos' in c.lower()), None)
    col_neg = next((c for c in sent.columns if 'neg' in c.lower()), None)
    col_neu = next((c for c in sent.columns if 'neu' in c.lower()), None)

    print(f"  Columnas detectadas: {col_pos}, {col_neg}, {col_neu}")

    # Eliminar filas con error de API
    if 'sentimiento_finbert' in sent.columns:
        antes = len(sent)
        sent  = sent[sent['sentimiento_finbert'] != 'ERROR_API'].copy()
        if len(sent) < antes:
            print(f"  Filas con ERROR_API eliminadas: {antes - len(sent)}")

    # Promedio por ticker
    sent_avg = (
        sent.groupby("ticker")[[col_pos, col_neg, col_neu]]
        .mean()
        .reset_index()
        .rename(columns={
            col_pos: "sent_pos",
            col_neg: "sent_neg",
            col_neu: "sent_neu",
        })
    )

    # Renormalizar para que sumen exactamente 1.0
    total = sent_avg[["sent_pos", "sent_neg", "sent_neu"]].sum(axis=1)
    sent_avg[["sent_pos", "sent_neg", "sent_neu"]] = (
        sent_avg[["sent_pos", "sent_neg", "sent_neu"]].div(total, axis=0)
    )

    print(f"\n  Sentimiento promedio por ticker (valores reales de FinBERT):")
    print(sent_avg.to_string(index=False))

    return sent_avg


# ══════════════════════════════════════════════════════════════════
#  PASO 3 — Filtrar BD y unir con sentimiento
# ══════════════════════════════════════════════════════════════════

def unir_datasets(bd: pd.DataFrame, sent_avg: pd.DataFrame) -> pd.DataFrame:
    print("\n" + "=" * 55)
    print("  PASO 3 — Filtrando BD y uniendo con sentimiento")
    print("=" * 55)

    tickers_sent = set(sent_avg["ticker"].unique())
    print(f"  Tickers con sentimiento real: {sorted(tickers_sent)}")

    # Filtrar BD — solo tickers con noticias reales de FinBERT
    bd_filtrado = bd[bd["ticker"].isin(tickers_sent)].copy()
    print(f"\n  Filas del BD original       : {len(bd):,}")
    print(f"  Filas del BD filtrado       : {len(bd_filtrado):,}")
    print(f"  Tickers en BD filtrado      : {bd_filtrado['ticker'].nunique()}")

    # Left join por ticker
    df = bd_filtrado.merge(sent_avg, on="ticker", how="left")

    # Verificar que no queden NaN
    nans = df[FEATURES_SENTIMIENTO].isna().sum().sum()
    if nans > 0:
        print(f"  Aviso: {nans} NaN encontrados — rellenando con neutro")
        df[FEATURES_SENTIMIENTO] = df[FEATURES_SENTIMIENTO].fillna(1/3)

    # Verificación — todos deben ser distintos de 0.333
    print(f"\n  Verificación de sentimiento por ticker:")
    print(f"  (todos deben ser distintos de 0.3333)")
    muestra = (
        df[['ticker', 'sent_pos', 'sent_neg', 'sent_neu']]
        .drop_duplicates('ticker')
        .sort_values('ticker')
        .reset_index(drop=True)
    )
    print(muestra.to_string(index=False))

    return df


# ══════════════════════════════════════════════════════════════════
#  PASO 4 — Agregar target
# ══════════════════════════════════════════════════════════════════

def agregar_target(df: pd.DataFrame) -> pd.DataFrame:
    print("\n" + "=" * 55)
    print("  PASO 4 — Generando target (sube/baja día siguiente)")
    print("=" * 55)

    df = df.sort_values(["ticker", "dt"]).copy()

    # Target: 1 si adj_close sube al día siguiente, 0 si baja
    df["target"] = (
        df.groupby("ticker")["adj_close"].shift(-1) > df["adj_close"]
    ).astype(int)

    # Eliminar filas sin target y sin features
    df = df.dropna(subset=["target"] + ALL_FEATURES).reset_index(drop=True)

    balance = df["target"].mean()
    print(f"  Filas finales : {len(df):,}")
    print(f"  Tickers       : {df['ticker'].nunique()}")
    print(f"  Balance clases: Sube {balance:.1%} / Baja {1 - balance:.1%}")

    if abs(balance - 0.5) > 0.1:
        print("  Aviso: dataset desbalanceado — los modelos usarán class_weight='balanced'")

    return df


# ══════════════════════════════════════════════════════════════════
#  PASO 5 — Split temporal y normalización
# ══════════════════════════════════════════════════════════════════

def split_y_normalizar(df: pd.DataFrame, pct_train=0.70, pct_val=0.15):
    print("\n" + "=" * 55)
    print("  PASO 5 — Split cronológico y normalización")
    print("=" * 55)

    fechas = np.sort(df["dt"].unique())
    c1 = fechas[int(len(fechas) * pct_train)]
    c2 = fechas[int(len(fechas) * (pct_train + pct_val))]

    train = df[df["dt"] <  c1].copy()
    val   = df[(df["dt"] >= c1) & (df["dt"] < c2)].copy()
    test  = df[df["dt"] >= c2].copy()

    print(f"  Train : {len(train):,} filas  ({train['dt'].min().date()} → {train['dt'].max().date()})")
    print(f"  Val   : {len(val):,} filas  ({val['dt'].min().date()} → {val['dt'].max().date()})")
    print(f"  Test  : {len(test):,} filas  ({test['dt'].min().date()} → {test['dt'].max().date()})")

    # Normalización: fit SOLO en train para evitar data leakage
    scaler = MinMaxScaler()
    train[ALL_FEATURES] = scaler.fit_transform(train[ALL_FEATURES])
    val[ALL_FEATURES]   = scaler.transform(val[ALL_FEATURES])
    test[ALL_FEATURES]  = scaler.transform(test[ALL_FEATURES])

    print(f"\n  Scaler ajustado solo con datos de train (sin data leakage)")

    return train, val, test, scaler


# ══════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════

def main():
    print("\n" + "═" * 55)
    print("  UNIÓN DE BASES — BD.parquet2 + FinBERT")
    print("  Solo tickers con sentimiento real")
    print("═" * 55)

    bd, sent                 = cargar_datos()
    sent_avg                 = preparar_sentimiento(sent)
    df                       = unir_datasets(bd, sent_avg)
    df                       = agregar_target(df)
    train, val, test, scaler = split_y_normalizar(df)

    # Guardar dataset final completo (sin normalizar, para referencia)
    df.to_parquet(OUTPUT_PATH, index=False)
    df.to_csv(OUTPUT_CSV, index=False)

    print("\n" + "═" * 55)
    print("  LISTO — Dataset guardado")
    print("═" * 55)
    print(f"  Parquet  : {OUTPUT_PATH}")
    print(f"  CSV      : {OUTPUT_CSV}")
    print(f"  Features : {len(ALL_FEATURES)} (24 técnicos + 3 sentimiento)")
    print(f"  Tickers  : {df['ticker'].nunique()} con sentimiento real")
    print(f"\n  Siguiente paso: torneo_modelos.py")

    return train, val, test, scaler


if __name__ == "__main__":
    train, val, test, scaler = main()