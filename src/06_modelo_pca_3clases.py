"""
07_modelo_pca_3clases.py

Estrategia:
    - PCA sobre los 24 indicadores técnicos (reduce redundancia)
    - Sentimiento FinBERT se mantiene como features separadas
    - Target de 3 clases: Baja fuerte / Neutro / Sube fuerte
    - Torneo: XGBoost vs Random Forest vs MLP
    - Manejo de desbalance con class_weight
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
from sklearn.utils.class_weight import compute_class_weight

try:
    import xgboost as xgb
    XGB_DISPONIBLE = True
except ImportError:
    XGB_DISPONIBLE = False
    print("Aviso: xgboost no instalado — se omitirá ese modelo")

# ── Rutas ──────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PATH_DS     = os.path.join(BASE_DIR, 'data', 'processed', 'dataset_final.parquet')
OUTPUT_PATH = os.path.join(BASE_DIR, 'data', 'processed', 'resultados_pca_3clases.csv')

# ── Features ───────────────────────────────────────────────────────
FEATURES_TECNICOS = [
    "open", "high", "low", "close", "adj_close", "volume",
    "ema_9", "ema_21", "sma_50",
    "macd", "macd_signal", "macd_hist", "retorno_log",
    "rsi_14", "stoch_k", "stoch_d", "roc_10", "vol_ratio",
    "volatilidad_20", "atr_14", "bb_upper", "bb_mid", "bb_lower", "bb_width",
]
FEATURES_SENTIMIENTO = ["sent_pos", "sent_neg", "sent_neu"]

# ── Configuración ──────────────────────────────────────────────────
UMBRAL_PCT    = 0.02    # ±2% en 3 días define sube/baja fuerte
VARIANZA_PCA  = 0.90    # PCA conserva el 90% de la varianza
ETIQUETAS     = {0: "Baja fuerte", 1: "Neutro", 2: "Sube fuerte"}


# ══════════════════════════════════════════════════════════════════
#  PASO 1 — Cargar y crear target de 3 clases
# ══════════════════════════════════════════════════════════════════

def cargar_datos():
    print("=" * 60)
    print("  PASO 1 — Cargando datos y creando target de 3 clases")
    print("=" * 60)

    df = pd.read_parquet(PATH_DS)
    df["dt"] = pd.to_datetime(df["dt"])
    df = df.sort_values(["ticker", "dt"]).reset_index(drop=True)

    # Retorno acumulado en 3 días por ticker
    df["retorno_3d"] = (
        df.groupby("ticker")["adj_close"]
        .transform(lambda x: x.shift(-3) / x - 1)
    )

    # Clasificación en 3 clases
    def clasificar(r):
        if r >  UMBRAL_PCT: return 2   # Sube fuerte
        if r < -UMBRAL_PCT: return 0   # Baja fuerte
        return 1                        # Neutro

    df["target_3c"] = df["retorno_3d"].apply(
        lambda x: clasificar(x) if pd.notna(x) else np.nan
    )
    df = df.dropna(
        subset=["retorno_3d", "target_3c"] + FEATURES_TECNICOS + FEATURES_SENTIMIENTO
    ).reset_index(drop=True)
    df["target_3c"] = df["target_3c"].astype(int)

    # Distribución
    dist = df["target_3c"].value_counts(normalize=True).sort_index()
    print(f"\n  Dataset: {len(df):,} filas · {df['ticker'].nunique()} tickers")
    print(f"  Umbral : ±{UMBRAL_PCT*100:.0f}% en 3 días\n")
    print(f"  Distribución de clases:")
    for cls, pct in dist.items():
        n = df["target_3c"].value_counts()[cls]
        print(f"    {ETIQUETAS[cls]:15}: {pct:.1%}  ({n:,} casos)")

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
#  PASO 3 — Normalización + PCA sobre técnicos
# ══════════════════════════════════════════════════════════════════

def aplicar_pca(train, val, test):
    print("\n" + "=" * 60)
    print("  PASO 3 — Normalización + PCA sobre indicadores técnicos")
    print(f"  Sentimiento FinBERT se mantiene separado")
    print("=" * 60)

    # ── Normalizar técnicos ───────────────────────────────────────
    scaler_tec = MinMaxScaler()
    train_tec  = scaler_tec.fit_transform(train[FEATURES_TECNICOS])
    val_tec    = scaler_tec.transform(val[FEATURES_TECNICOS])
    test_tec   = scaler_tec.transform(test[FEATURES_TECNICOS])

    # ── PCA sobre técnicos — fit solo en train ────────────────────
    pca = PCA(n_components=VARIANZA_PCA, random_state=42)
    train_pca = pca.fit_transform(train_tec)
    val_pca   = pca.transform(val_tec)
    test_pca  = pca.transform(test_tec)

    n_comp = pca.n_components_
    var_exp = pca.explained_variance_ratio_.cumsum()[-1]
    print(f"\n  Componentes PCA seleccionados : {n_comp} (de 24 originales)")
    print(f"  Varianza explicada            : {var_exp:.1%}")
    print(f"  Reducción de dimensionalidad  : 24 → {n_comp} features técnicas")

    # ── Normalizar sentimiento (sin PCA) ─────────────────────────
    scaler_sent = MinMaxScaler()
    train_sent  = scaler_sent.fit_transform(train[FEATURES_SENTIMIENTO])
    val_sent    = scaler_sent.transform(val[FEATURES_SENTIMIENTO])
    test_sent   = scaler_sent.transform(test[FEATURES_SENTIMIENTO])

    # ── Combinar PCA + sentimiento ────────────────────────────────
    X_train = np.hstack([train_pca, train_sent])
    X_val   = np.hstack([val_pca,   val_sent])
    X_test  = np.hstack([test_pca,  test_sent])

    y_train = train["target_3c"].values
    y_val   = val["target_3c"].values
    y_test  = test["target_3c"].values

    print(f"\n  Shape final X_train : {X_train.shape}  ({n_comp} PCA + 3 sentimiento)")
    print(f"  Shape final X_val   : {X_val.shape}")
    print(f"  Shape final X_test  : {X_test.shape}")

    return X_train, X_val, X_test, y_train, y_val, y_test, pca, scaler_tec


# ══════════════════════════════════════════════════════════════════
#  UTILIDADES
# ══════════════════════════════════════════════════════════════════

def reporte(nombre, y_true, y_pred):
    acc  = accuracy_score(y_true, y_pred)
    f1   = f1_score(y_true, y_pred, average="macro", zero_division=0)
    prec = precision_score(y_true, y_pred, average="macro", zero_division=0)
    rec  = recall_score(y_true, y_pred, average="macro", zero_division=0)

    print(f"\n{'─' * 60}")
    print(f"  {nombre}")
    print(f"{'─' * 60}")
    print(f"  Accuracy  : {acc:.4f}  ({acc:.1%})")
    print(f"  F1 macro  : {f1:.4f}")
    print(f"  Precision : {prec:.4f}")
    print(f"  Recall    : {rec:.4f}")
    print()
    print(classification_report(
        y_true, y_pred,
        target_names=["Baja fuerte", "Neutro", "Sube fuerte"],
        zero_division=0
    ))

    # Matriz de confusión
    cm = confusion_matrix(y_true, y_pred)
    print(f"  Matriz de confusión (filas=real, cols=predicho):")
    print(f"  {'':15} {'Pred Baja':>10} {'Pred Neutro':>12} {'Pred Sube':>10}")
    for i, etq in enumerate(["Real Baja", "Real Neutro", "Real Sube"]):
        print(f"  {etq:15} {cm[i][0]:>10} {cm[i][1]:>12} {cm[i][2]:>10}")

    return {
        "modelo":    nombre,
        "accuracy":  round(acc,  4),
        "f1_macro":  round(f1,   4),
        "precision": round(prec, 4),
        "recall":    round(rec,  4),
    }


def pesos_clase(y):
    clases = np.unique(y)
    pesos  = compute_class_weight("balanced", classes=clases, y=y)
    return dict(zip(clases, pesos))


# ══════════════════════════════════════════════════════════════════
#  MODELOS
# ══════════════════════════════════════════════════════════════════

def entrenar_xgb(X_tr, y_tr, X_v, y_v):
    print("\n" + "═" * 60)
    print("  MODELO 1 — XGBoost")
    print("═" * 60)

    model = xgb.XGBClassifier(
        n_estimators=500,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="multi:softmax",
        num_class=3,
        eval_metric="mlogloss",
        early_stopping_rounds=20,
        random_state=42,
        verbosity=0,
    )
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_v, y_v)],
        verbose=False
    )
    pred = model.predict(X_v)
    r = reporte("XGBoost [VAL]", y_v, pred)
    return model, r


def entrenar_rf(X_tr, y_tr, X_v, y_v):
    print("\n" + "═" * 60)
    print("  MODELO 2 — Random Forest")
    print("═" * 60)

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=8,
        min_samples_leaf=10,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_tr, y_tr)
    pred = model.predict(X_v)
    r = reporte("Random Forest [VAL]", y_v, pred)
    return model, r


def entrenar_mlp(X_tr, y_tr, X_v, y_v):
    print("\n" + "═" * 60)
    print("  MODELO 3 — MLP (Red neuronal simple)")
    print("═" * 60)

    model = MLPClassifier(
        hidden_layer_sizes=(64, 32),
        activation="relu",
        max_iter=500,
        early_stopping=True,
        validation_fraction=0.1,
        random_state=42,
    )
    model.fit(X_tr, y_tr)
    pred = model.predict(X_v)
    r = reporte("MLP [VAL]", y_v, pred)
    return model, r


# ══════════════════════════════════════════════════════════════════
#  TORNEO Y EVALUACIÓN FINAL
# ══════════════════════════════════════════════════════════════════

def correr_torneo(X_tr, y_tr, X_v, y_v, X_test, y_test):
    print("\n" + "═" * 60)
    print("  TORNEO DE MODELOS — 3 clases")
    print("  PCA técnicos + sentimiento FinBERT separado")
    print("═" * 60)

    resultados = {}
    modelos    = {}

    # Random Forest siempre disponible
    m, r = entrenar_rf(X_tr, y_tr, X_v, y_v)
    resultados["Random Forest"] = r
    modelos["Random Forest"]    = m

    # MLP siempre disponible
    m, r = entrenar_mlp(X_tr, y_tr, X_v, y_v)
    resultados["MLP"] = r
    modelos["MLP"]    = m

    # XGBoost solo si está instalado
    if XGB_DISPONIBLE:
        m, r = entrenar_xgb(X_tr, y_tr, X_v, y_v)
        resultados["XGBoost"] = r
        modelos["XGBoost"]    = m

    # ── Ranking por F1 macro ──────────────────────────────────────
    tabla = (
        pd.DataFrame(list(resultados.values()))
        .sort_values("f1_macro", ascending=False)
        .reset_index(drop=True)
    )
    tabla.index += 1

    print("\n" + "═" * 60)
    print("  RANKING FINAL (por F1 macro en validación)")
    print("═" * 60)
    print(tabla[["modelo", "accuracy", "f1_macro", "precision", "recall"]].to_string())

    ganador_nombre = tabla.iloc[0]["modelo"].replace(" [VAL]", "")
    ganador        = modelos[ganador_nombre]

    print(f"\n  Ganador: {ganador_nombre}")

    # ── Evaluación final en TEST ──────────────────────────────────
    print("\n" + "═" * 60)
    print(f"  EVALUACIÓN FINAL EN TEST — {ganador_nombre}")
    print("═" * 60)
    pred_test = ganador.predict(X_test)
    r_test    = reporte(f"{ganador_nombre} [TEST]", y_test, pred_test)

    # ── Señales del último día disponible ─────────────────────────
    print("\n" + "═" * 60)
    print("  SEÑALES DE INVERSIÓN — última fecha disponible")
    print("═" * 60)
    proba = ganador.predict_proba(X_test[-7:])
    for i, (p0, p1, p2) in enumerate(proba):
        senal = ETIQUETAS[np.argmax([p0, p1, p2])]
        print(f"  Muestra {i+1:>2}: "
              f"Baja {p0:.1%} | Neutro {p1:.1%} | Sube {p2:.1%}  → {senal}")

    # Guardar resultados
    tabla.to_csv(OUTPUT_PATH, index=False)
    print(f"\n  Resultados guardados en: {OUTPUT_PATH}")

    return ganador, ganador_nombre, tabla


# ══════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════

def main():
    print("\n" + "═" * 60)
    print("  MODELO PCA + 3 CLASES — FinSage")
    print("  Estrategia: PCA técnicos + FinBERT separado")
    print("═" * 60)

    df                                          = cargar_datos()
    train, val, test                            = split_temporal(df)
    X_tr, X_v, X_te, y_tr, y_v, y_te, pca, sc = aplicar_pca(train, val, test)
    ganador, nombre, tabla                      = correr_torneo(
                                                    X_tr, y_tr,
                                                    X_v,  y_v,
                                                    X_te, y_te
                                                  )

    print("\n" + "═" * 60)
    print("  RESUMEN EJECUTIVO")
    print("═" * 60)
    print(f"  Modelo ganador     : {nombre}")
    print(f"  Features usadas    : {X_tr.shape[1]} "
          f"({pca.n_components_} PCA + 3 FinBERT)")
    print(f"  Clases             : Baja fuerte / Neutro / Sube fuerte")
    print(f"  Umbral             : ±{UMBRAL_PCT*100:.0f}% en 3 días")
    print(f"  Tickers evaluados  : DIS, GOOGL, JPM, META, MSFT, NFLX, NVDA")
    print(f"\n  Interpretación de señales:")
    print(f"    Sube fuerte → COMPRA")
    print(f"    Neutro      → MANTENER")
    print(f"    Baja fuerte → NO INVERTIR / VENDER")

    return ganador, nombre, pca, sc


if __name__ == "__main__":
    ganador, nombre, pca, scaler = main()