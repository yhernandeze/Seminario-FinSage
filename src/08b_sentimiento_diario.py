"""
08b_sentimiento_diario.py

Mejora 2: Sentimiento FinBERT con granularidad DIARIA
- En lugar de un promedio estático por ticker, cada día
  tiene su propio score de sentimiento
- Días sin noticias reciben el promedio histórico del ticker
- Esto da al modelo señal real: "hoy hay noticias negativas de MSFT"

PREREQUISITO: Tener el 03_clasificacion.py corrido con TODAS las
noticias (no solo 300) y con columna de fecha por noticia.

Si aún no tienes fecha por noticia, este script también incluye
una función para simular fechas y probar el pipeline.
"""

import numpy as np
import pandas as pd
import os
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report, confusion_matrix
)

try:
    import xgboost as xgb
    XGB_DISPONIBLE = True
except ImportError:
    XGB_DISPONIBLE = False

# ── Rutas ──────────────────────────────────────────────────────────
BASE_DIR     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PATH_BD      = os.path.join(BASE_DIR, 'data', 'raw', 'BD.parquet2')
PATH_SENT    = os.path.join(BASE_DIR, 'data', 'processed', 'salida_sentimientos_finbert.csv')
OUTPUT_DS    = os.path.join(BASE_DIR, 'data', 'processed', 'dataset_sentimiento_diario.parquet')
OUTPUT_CSV   = os.path.join(BASE_DIR, 'data', 'processed', 'dataset_sentimiento_diario.csv')
OUTPUT_RES   = os.path.join(BASE_DIR, 'data', 'processed', 'resultados_sentimiento_diario.csv')

FEATURES_TECNICOS    = [
    "open", "high", "low", "close", "adj_close", "volume",
    "ema_9", "ema_21", "sma_50",
    "macd", "macd_signal", "macd_hist", "retorno_log",
    "rsi_14", "stoch_k", "stoch_d", "roc_10", "vol_ratio",
    "volatilidad_20", "atr_14", "bb_upper", "bb_mid", "bb_lower", "bb_width",
]
FEATURES_SENTIMIENTO = ["sent_pos", "sent_neg", "sent_neu"]
ALL_FEATURES         = FEATURES_TECNICOS + FEATURES_SENTIMIENTO

UMBRAL_PCT   = 0.02
VARIANZA_PCA = 0.90
TICKERS_SENT = ['DIS', 'GOOGL', 'JPM', 'META', 'MSFT', 'NFLX', 'NVDA']


# ══════════════════════════════════════════════════════════════════
#  DIAGNÓSTICO — detectar si el CSV tiene fecha
# ══════════════════════════════════════════════════════════════════

def detectar_columna_fecha(sent: pd.DataFrame):
    """Busca automáticamente la columna de fecha en el CSV de noticias."""
    candidatas = [c for c in sent.columns
                  if any(k in c.lower() for k in ['date', 'fecha', 'dt', 'time', 'published'])]
    if candidatas:
        print(f"  Columna de fecha detectada : {candidatas[0]}")
        return candidatas[0]
    print("  No se detectó columna de fecha en el CSV de noticias")
    return None


# ══════════════════════════════════════════════════════════════════
#  MODO A — CSV con fecha por noticia (ideal)
# ══════════════════════════════════════════════════════════════════

def preparar_sentimiento_con_fecha(sent: pd.DataFrame, col_fecha: str) -> pd.DataFrame:
    """
    Agrega el sentimiento por ticker+fecha cuando el CSV tiene fecha.
    Días sin noticias reciben el promedio histórico del ticker.
    """
    print("\n  Modo: sentimiento DIARIO (CSV tiene fecha)")

    sent = sent[sent['sentimiento_finbert'] != 'ERROR_API'].copy()
    sent[col_fecha] = pd.to_datetime(sent[col_fecha], dayfirst=True, errors='coerce')
    sent = sent.dropna(subset=[col_fecha])
    sent = sent.rename(columns={col_fecha: "dt"})
    sent["dt"] = sent["dt"].dt.normalize()  # quitar hora, solo fecha

    # Detectar columnas de scores
    col_pos = next((c for c in sent.columns if c.lower() == 'positive'), None)
    col_neg = next((c for c in sent.columns if c.lower() == 'negative'), None)
    col_neu = next((c for c in sent.columns if c.lower() == 'neutral'),  None)

    # Promedio por ticker+fecha
    sent_diario = (
        sent.groupby(["ticker", "dt"])[[col_pos, col_neg, col_neu]]
        .mean()
        .reset_index()
        .rename(columns={col_pos: "sent_pos", col_neg: "sent_neg", col_neu: "sent_neu"})
    )

    # Promedio histórico por ticker (para días sin noticias)
    sent_avg = (
        sent.groupby("ticker")[[col_pos, col_neg, col_neu]]
        .mean()
        .reset_index()
        .rename(columns={col_pos: "sent_pos_avg", col_neg: "sent_neg_avg", col_neu: "sent_neu_avg"})
    )

    print(f"  Días con noticias por ticker:")
    print(sent_diario.groupby("ticker").size().to_string())

    return sent_diario, sent_avg


# ══════════════════════════════════════════════════════════════════
#  MODO B — CSV sin fecha (simulación para demostración)
# ══════════════════════════════════════════════════════════════════

def simular_sentimiento_diario(sent: pd.DataFrame, bd: pd.DataFrame) -> pd.DataFrame:
    """
    Cuando el CSV no tiene fecha, simula sentimiento diario para
    demostrar el pipeline. Distribuye las noticias por ticker
    a lo largo del rango de fechas del BD con variación realista.
    """
    print("\n  Modo: sentimiento SIMULADO (CSV sin fecha)")
    print("  Para resultados reales, agrega columna 'date' al CSV de noticias")

    sent = sent[sent['sentimiento_finbert'] != 'ERROR_API'].copy()

    resultados = []
    for ticker in TICKERS_SENT:
        bd_ticker   = bd[bd["ticker"] == ticker].sort_values("dt")
        sent_ticker = sent[sent["ticker"] == ticker]

        if len(bd_ticker) == 0 or len(sent_ticker) == 0:
            continue

        fechas = bd_ticker["dt"].values
        avg_pos = sent_ticker["positive"].mean()
        avg_neg = sent_ticker["negative"].mean()
        avg_neu = sent_ticker["neutral"].mean()

        # Añadir ruido realista ±10% alrededor del promedio
        np.random.seed(42)
        for fecha in fechas:
            noise = np.random.normal(0, 0.05, 3)
            pos = np.clip(avg_pos + noise[0], 0, 1)
            neg = np.clip(avg_neg + noise[1], 0, 1)
            neu = np.clip(avg_neu + noise[2], 0, 1)
            total = pos + neg + neu
            resultados.append({
                "ticker":   ticker,
                "dt":       fecha,
                "sent_pos": pos / total,
                "sent_neg": neg / total,
                "sent_neu": neu / total,
            })

    sent_diario = pd.DataFrame(resultados)
    sent_avg    = sent_diario.groupby("ticker")[["sent_pos","sent_neg","sent_neu"]].mean().reset_index()
    sent_avg    = sent_avg.rename(columns={
        "sent_pos": "sent_pos_avg",
        "sent_neg": "sent_neg_avg",
        "sent_neu": "sent_neu_avg",
    })

    print(f"  Filas de sentimiento simulado: {len(sent_diario):,}")
    return sent_diario, sent_avg


# ══════════════════════════════════════════════════════════════════
#  CONSTRUIR DATASET CON SENTIMIENTO DIARIO
# ══════════════════════════════════════════════════════════════════

def construir_dataset(bd, sent_diario, sent_avg):
    print("\n" + "=" * 60)
    print("  Construyendo dataset con sentimiento diario")
    print("=" * 60)

    # Filtrar BD a solo los 7 tickers con noticias
    bd_f = bd[bd["ticker"].isin(TICKERS_SENT)].copy()

    # Merge diario: cada fila del BD recibe el score de ese día
    df = bd_f.merge(sent_diario[["ticker","dt","sent_pos","sent_neg","sent_neu"]],
                    on=["ticker","dt"], how="left")

    # Días sin noticias → score promedio histórico del ticker
    sin_noticia = df["sent_pos"].isna()
    if sin_noticia.sum() > 0:
        df = df.merge(sent_avg, on="ticker", how="left")
        df.loc[sin_noticia, "sent_pos"] = df.loc[sin_noticia, "sent_pos_avg"]
        df.loc[sin_noticia, "sent_neg"] = df.loc[sin_noticia, "sent_neg_avg"]
        df.loc[sin_noticia, "sent_neu"] = df.loc[sin_noticia, "sent_neu_avg"]
        df = df.drop(columns=["sent_pos_avg","sent_neg_avg","sent_neu_avg"], errors="ignore")
        print(f"  Días sin noticias rellenados con promedio: {sin_noticia.sum():,}")

    # Target: retorno en 3 días, solo movimientos fuertes
    df = df.sort_values(["ticker","dt"])
    df["retorno_3d"] = (
        df.groupby("ticker")["adj_close"]
        .transform(lambda x: x.shift(-3) / x - 1)
    )
    df = df[df["retorno_3d"].abs() > UMBRAL_PCT].copy()
    df["target"] = (df["retorno_3d"] > 0).astype(int)
    df = df.dropna(subset=["target"] + ALL_FEATURES).reset_index(drop=True)

    balance = df["target"].mean()
    print(f"  Filas totales  : {len(df):,}")
    print(f"  Balance clases : Sube {balance:.1%} / Baja {1-balance:.1%}")

    # Guardar
    df.to_parquet(OUTPUT_DS, index=False)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"  Dataset guardado: {OUTPUT_DS}")

    return df


# ══════════════════════════════════════════════════════════════════
#  PIPELINE COMPLETO: SPLIT + PCA + MODELO
# ══════════════════════════════════════════════════════════════════

def pipeline_modelo(df):
    print("\n" + "=" * 60)
    print("  Split + PCA + Modelo")
    print("=" * 60)

    fechas = np.sort(df["dt"].unique())
    c1 = fechas[int(len(fechas) * 0.70)]
    c2 = fechas[int(len(fechas) * 0.85)]

    train = df[df["dt"] <  c1].copy()
    val   = df[(df["dt"] >= c1) & (df["dt"] < c2)].copy()
    test  = df[df["dt"] >= c2].copy()

    print(f"  Train : {len(train):,} filas")
    print(f"  Val   : {len(val):,} filas")
    print(f"  Test  : {len(test):,} filas")

    # PCA técnicos
    scaler = MinMaxScaler()
    pca    = PCA(n_components=VARIANZA_PCA, random_state=42)

    X_tr_tec = pca.fit_transform(scaler.fit_transform(train[FEATURES_TECNICOS]))
    X_v_tec  = pca.transform(scaler.transform(val[FEATURES_TECNICOS]))
    X_te_tec = pca.transform(scaler.transform(test[FEATURES_TECNICOS]))

    # Sentimiento diario (sin PCA)
    sc_sent  = MinMaxScaler()
    X_tr_s   = sc_sent.fit_transform(train[FEATURES_SENTIMIENTO])
    X_v_s    = sc_sent.transform(val[FEATURES_SENTIMIENTO])
    X_te_s   = sc_sent.transform(test[FEATURES_SENTIMIENTO])

    X_tr = np.hstack([X_tr_tec, X_tr_s])
    X_v  = np.hstack([X_v_tec,  X_v_s])
    X_te = np.hstack([X_te_tec, X_te_s])

    y_tr = train["target"].values
    y_v  = val["target"].values
    y_te = test["target"].values

    print(f"\n  Shape X_train : {X_tr.shape} ({pca.n_components_} PCA + 3 FinBERT diario)")

    # Random Forest
    modelo = RandomForestClassifier(
        n_estimators=300, max_depth=8,
        min_samples_leaf=10, class_weight="balanced",
        random_state=42, n_jobs=-1
    )
    modelo.fit(X_tr, y_tr)

    # Validación
    pred_v = modelo.predict(X_v)
    acc_v  = accuracy_score(y_v, pred_v)
    f1_v   = f1_score(y_v, pred_v, zero_division=0)
    print(f"\n  Validación — Accuracy: {acc_v:.1%} · F1: {f1_v:.4f}")

    # Test final
    pred_te = modelo.predict(X_te)
    acc_te  = accuracy_score(y_te, pred_te)
    f1_te   = f1_score(y_te, pred_te, zero_division=0)

    print(f"\n{'─'*60}")
    print(f"  Random Forest [TEST] — Sentimiento diario")
    print(f"{'─'*60}")
    print(f"  Accuracy  : {acc_te:.4f}  ({acc_te:.1%})")
    print(f"  F1-score  : {f1_te:.4f}")
    print()
    print(classification_report(y_te, pred_te,
          target_names=["Baja fuerte","Sube fuerte"], zero_division=0))

    # Señales finales
    print("\n  SEÑALES — últimas 7 muestras del test:")
    proba = modelo.predict_proba(X_te[-7:])
    for i, (p0, p1) in enumerate(proba):
        senal = "COMPRA" if p1 >= 0.60 else "NO INVERTIR" if p0 >= 0.60 else "MANTENER"
        print(f"  Muestra {i+1:>2}: Baja {p0:.1%} | Sube {p1:.1%}  → {senal}")

    return modelo, acc_te, f1_te


# ══════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════

def main():
    print("\n" + "═" * 60)
    print("  MEJORA 2 — Sentimiento FinBERT con granularidad diaria")
    print("═" * 60)

    bd   = pd.read_parquet(PATH_BD)
    bd["dt"] = pd.to_datetime(bd["dt"])
    sent = pd.read_csv(PATH_SENT)
    sent.columns = [c.strip().lower() for c in sent.columns]

    print(f"\n  BD cargado   : {len(bd):,} filas · {bd['ticker'].nunique()} tickers")
    print(f"  CSV noticias : {len(sent):,} filas")
    print(f"  Columnas CSV : {sent.columns.tolist()}")

    # Detectar si el CSV tiene fecha
    col_fecha = detectar_columna_fecha(sent)

    if col_fecha:
        sent_diario, sent_avg = preparar_sentimiento_con_fecha(sent, col_fecha)
        modo = "DIARIO REAL"
    else:
        sent_diario, sent_avg = simular_sentimiento_diario(sent, bd)
        modo = "DIARIO SIMULADO"

    df = construir_dataset(bd, sent_diario, sent_avg)
    modelo, acc, f1 = pipeline_modelo(df)

    print("\n" + "═" * 60)
    print("  RESUMEN COMPARATIVO")
    print("═" * 60)
    print(f"  Modo sentimiento   : {modo}")
    print(f"  Accuracy en test   : {acc:.1%}")
    print(f"  F1-score en test   : {f1:.4f}")
    print(f"\n  Baseline anterior  : ~47.6% accuracy (sentimiento estático)")
    print(f"  Mejora esperada    : +5-10% con sentimiento diario real")
    print(f"\n  Para activar modo REAL:")
    print(f"  → Asegúrate que el CSV de noticias tenga columna 'date' o 'fecha'")
    print(f"  → Corre el 03_clasificacion.py con las 3,900 noticias completas")

    return modelo


if __name__ == "__main__":
    modelo = main()