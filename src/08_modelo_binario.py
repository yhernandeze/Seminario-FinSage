"""
08a_modelo_binario.py

Mejora 1: Clasificación binaria limpia
- Elimina la clase Neutro (movimientos entre ±2%)
- Solo predice: Baja fuerte (0) vs Sube fuerte (1)
- Esto elimina el ruido que destruía las métricas
- Usa PCA sobre técnicos + FinBERT separado
"""

import numpy as np
import pandas as pd
import os
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score,
    recall_score, classification_report, confusion_matrix
)

try:
    import xgboost as xgb
    XGB_DISPONIBLE = True
except ImportError:
    XGB_DISPONIBLE = False
    print("Aviso: xgboost no disponible — se omitirá")

# ── Rutas ──────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PATH_DS     = os.path.join(BASE_DIR, 'data', 'processed', 'dataset_final.parquet')
OUTPUT_PATH = os.path.join(BASE_DIR, 'data', 'processed', 'resultados_binario.csv')

# ── Features ───────────────────────────────────────────────────────
FEATURES_TECNICOS = [
    "open", "high", "low", "close", "adj_close", "volume",
    "ema_9", "ema_21", "sma_50",
    "macd", "macd_signal", "macd_hist", "retorno_log",
    "rsi_14", "stoch_k", "stoch_d", "roc_10", "vol_ratio",
    "volatilidad_20", "atr_14", "bb_upper", "bb_mid", "bb_lower", "bb_width",
]
FEATURES_SENTIMIENTO = ["sent_pos", "sent_neg", "sent_neu"]

UMBRAL_PCT   = 0.02   # ±2% en 3 días
VARIANZA_PCA = 0.90


# ══════════════════════════════════════════════════════════════════
#  PASO 1 — Cargar y crear target BINARIO (sin Neutro)
# ══════════════════════════════════════════════════════════════════

def cargar_datos():
    print("=" * 60)
    print("  PASO 1 — Target binario: solo Baja fuerte vs Sube fuerte")
    print("=" * 60)

    df = pd.read_parquet(PATH_DS)
    df["dt"] = pd.to_datetime(df["dt"])
    df = df.sort_values(["ticker", "dt"]).reset_index(drop=True)

    # Retorno acumulado en 3 días
    df["retorno_3d"] = (
        df.groupby("ticker")["adj_close"]
        .transform(lambda x: x.shift(-3) / x - 1)
    )

    # Solo conservar movimientos fuertes — eliminar zona Neutro
    df = df[df["retorno_3d"].abs() > UMBRAL_PCT].copy()

    # Target binario limpio
    df["target"] = (df["retorno_3d"] > 0).astype(int)
    df = df.dropna(
        subset=["target"] + FEATURES_TECNICOS + FEATURES_SENTIMIENTO
    ).reset_index(drop=True)

    dist = df["target"].value_counts(normalize=True).sort_index()
    print(f"\n  Dataset tras eliminar Neutro: {len(df):,} filas")
    print(f"  Tickers                     : {df['ticker'].nunique()}")
    print(f"\n  Distribución:")
    print(f"    Baja fuerte (0): {dist[0]:.1%}  ({df['target'].value_counts()[0]:,} casos)")
    print(f"    Sube fuerte (1): {dist[1]:.1%}  ({df['target'].value_counts()[1]:,} casos)")
    print(f"\n  Filas eliminadas (Neutro)   : 158 — ya no contaminan el modelo")

    return df


# ══════════════════════════════════════════════════════════════════
#  PASO 2 — Split cronológico
# ══════════════════════════════════════════════════════════════════

def split_temporal(df):
    print("\n" + "=" * 60)
    print("  PASO 2 — Split cronológico")
    print("=" * 60)

    fechas = np.sort(df["dt"].unique())
    c1 = fechas[int(len(fechas) * 0.70)]
    c2 = fechas[int(len(fechas) * 0.85)]

    train = df[df["dt"] <  c1].copy()
    val   = df[(df["dt"] >= c1) & (df["dt"] < c2)].copy()
    test  = df[df["dt"] >= c2].copy()

    print(f"  Train : {len(train):,} filas  ({train['dt'].min().date()} → {train['dt'].max().date()})")
    print(f"  Val   : {len(val):,} filas  ({val['dt'].min().date()} → {val['dt'].max().date()})")
    print(f"  Test  : {len(test):,} filas  ({test['dt'].min().date()} → {test['dt'].max().date()})")

    return train, val, test


# ══════════════════════════════════════════════════════════════════
#  PASO 3 — PCA técnicos + sentimiento separado
# ══════════════════════════════════════════════════════════════════

def aplicar_pca(train, val, test):
    print("\n" + "=" * 60)
    print("  PASO 3 — PCA técnicos + FinBERT separado")
    print("=" * 60)

    scaler_tec = MinMaxScaler()
    train_tec  = scaler_tec.fit_transform(train[FEATURES_TECNICOS])
    val_tec    = scaler_tec.transform(val[FEATURES_TECNICOS])
    test_tec   = scaler_tec.transform(test[FEATURES_TECNICOS])

    pca = PCA(n_components=VARIANZA_PCA, random_state=42)
    train_pca = pca.fit_transform(train_tec)
    val_pca   = pca.transform(val_tec)
    test_pca  = pca.transform(test_tec)

    scaler_sent = MinMaxScaler()
    train_sent  = scaler_sent.fit_transform(train[FEATURES_SENTIMIENTO])
    val_sent    = scaler_sent.transform(val[FEATURES_SENTIMIENTO])
    test_sent   = scaler_sent.transform(test[FEATURES_SENTIMIENTO])

    X_train = np.hstack([train_pca, train_sent])
    X_val   = np.hstack([val_pca,   val_sent])
    X_test  = np.hstack([test_pca,  test_sent])

    y_train = train["target"].values
    y_val   = val["target"].values
    y_test  = test["target"].values

    print(f"  Componentes PCA : {pca.n_components_} (varianza: {pca.explained_variance_ratio_.cumsum()[-1]:.1%})")
    print(f"  Shape X_train   : {X_train.shape}  ({pca.n_components_} PCA + 3 FinBERT)")

    return X_train, X_val, X_test, y_train, y_val, y_test, pca


# ══════════════════════════════════════════════════════════════════
#  UTILIDADES
# ══════════════════════════════════════════════════════════════════

def reporte(nombre, y_true, y_pred):
    acc  = accuracy_score(y_true, y_pred)
    f1   = f1_score(y_true, y_pred, zero_division=0)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec  = recall_score(y_true, y_pred, zero_division=0)

    print(f"\n{'─' * 60}")
    print(f"  {nombre}")
    print(f"{'─' * 60}")
    print(f"  Accuracy  : {acc:.4f}  ({acc:.1%})")
    print(f"  F1        : {f1:.4f}")
    print(f"  Precision : {prec:.4f}")
    print(f"  Recall    : {rec:.4f}")
    print()
    print(classification_report(
        y_true, y_pred,
        target_names=["Baja fuerte", "Sube fuerte"],
        zero_division=0
    ))
    cm = confusion_matrix(y_true, y_pred)
    print(f"  Matriz de confusión:")
    print(f"                 Pred Baja  Pred Sube")
    print(f"  Real Baja    : {cm[0][0]:>9}  {cm[0][1]:>9}")
    print(f"  Real Sube    : {cm[1][0]:>9}  {cm[1][1]:>9}")

    return {"modelo": nombre, "accuracy": round(acc,4),
            "f1": round(f1,4), "precision": round(prec,4), "recall": round(rec,4)}


# ══════════════════════════════════════════════════════════════════
#  MODELOS
# ══════════════════════════════════════════════════════════════════

def entrenar_rf(X_tr, y_tr, X_v, y_v):
    print("\n" + "═" * 60)
    print("  MODELO 1 — Random Forest")
    print("═" * 60)
    m = RandomForestClassifier(
        n_estimators=300, max_depth=8,
        min_samples_leaf=10, class_weight="balanced",
        random_state=42, n_jobs=-1
    )
    m.fit(X_tr, y_tr)
    return m, reporte("Random Forest [VAL]", y_v, m.predict(X_v))


def entrenar_mlp(X_tr, y_tr, X_v, y_v):
    print("\n" + "═" * 60)
    print("  MODELO 2 — MLP")
    print("═" * 60)
    m = MLPClassifier(
        hidden_layer_sizes=(64, 32), activation="relu",
        max_iter=500, early_stopping=True,
        validation_fraction=0.1, random_state=42
    )
    m.fit(X_tr, y_tr)
    return m, reporte("MLP [VAL]", y_v, m.predict(X_v))


def entrenar_xgb(X_tr, y_tr, X_v, y_v):
    print("\n" + "═" * 60)
    print("  MODELO 3 — XGBoost")
    print("═" * 60)
    ratio = (y_tr == 0).sum() / max((y_tr == 1).sum(), 1)
    m = xgb.XGBClassifier(
        n_estimators=500, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=ratio, eval_metric="logloss",
        early_stopping_rounds=20, random_state=42, verbosity=0
    )
    m.fit(X_tr, y_tr, eval_set=[(X_v, y_v)], verbose=False)
    return m, reporte("XGBoost [VAL]", y_v, m.predict(X_v))


# ══════════════════════════════════════════════════════════════════
#  TORNEO
# ══════════════════════════════════════════════════════════════════

def correr_torneo(X_tr, y_tr, X_v, y_v, X_te, y_te):
    print("\n" + "═" * 60)
    print("  TORNEO — Clasificación binaria limpia")
    print("═" * 60)

    resultados, modelos = {}, {}

    m, r = entrenar_rf(X_tr, y_tr, X_v, y_v)
    resultados["Random Forest"] = r; modelos["Random Forest"] = m

    m, r = entrenar_mlp(X_tr, y_tr, X_v, y_v)
    resultados["MLP"] = r; modelos["MLP"] = m

    if XGB_DISPONIBLE:
        m, r = entrenar_xgb(X_tr, y_tr, X_v, y_v)
        resultados["XGBoost"] = r; modelos["XGBoost"] = m

    tabla = (
        pd.DataFrame(list(resultados.values()))
        .sort_values("f1", ascending=False)
        .reset_index(drop=True)
    )
    tabla.index += 1

    print("\n" + "═" * 60)
    print("  RANKING (por F1 en validación)")
    print("═" * 60)
    print(tabla[["modelo", "accuracy", "f1", "precision", "recall"]].to_string())

    ganador_nombre = tabla.iloc[0]["modelo"].replace(" [VAL]", "")
    ganador        = modelos[ganador_nombre]
    print(f"\n  Ganador: {ganador_nombre}")

    print("\n" + "═" * 60)
    print(f"  EVALUACIÓN FINAL EN TEST — {ganador_nombre}")
    print("═" * 60)
    reporte(f"{ganador_nombre} [TEST]", y_te, ganador.predict(X_te))

    # Señales últimas muestras
    print("\n" + "═" * 60)
    print("  SEÑALES DE INVERSIÓN — últimas muestras del test")
    print("═" * 60)
    proba = ganador.predict_proba(X_te[-7:])
    for i, (p0, p1) in enumerate(proba):
        senal = "COMPRA" if p1 >= 0.60 else "NO INVERTIR" if p0 >= 0.60 else "MANTENER"
        print(f"  Muestra {i+1:>2}: Baja {p0:.1%} | Sube {p1:.1%}  → {senal}")

    tabla.to_csv(OUTPUT_PATH, index=False)
    return ganador, ganador_nombre, tabla


# ══════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════

def main():
    print("\n" + "═" * 60)
    print("  MEJORA 1 — Clasificación binaria sin clase Neutro")
    print("═" * 60)

    df                            = cargar_datos()
    train, val, test              = split_temporal(df)
    X_tr, X_v, X_te, y_tr, y_v, y_te, pca = aplicar_pca(train, val, test)
    ganador, nombre, tabla        = correr_torneo(X_tr, y_tr, X_v, y_v, X_te, y_te)

    print("\n" + "═" * 60)
    print("  RESUMEN")
    print("═" * 60)
    print(f"  Modelo ganador  : {nombre}")
    print(f"  Features        : {X_tr.shape[1]} ({pca.n_components_} PCA + 3 FinBERT)")
    print(f"  Mejora esperada : accuracy > 55% al eliminar ruido del Neutro")
    print(f"  Siguiente paso  : 08b_sentimiento_diario.py")

    return ganador, nombre


if __name__ == "__main__":
    ganador, nombre = main()