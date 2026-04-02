import numpy as np
import pandas as pd
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.preprocessing import MinMaxScaler

# ── Rutas ──────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PATH_DS     = os.path.join(BASE_DIR, 'data', 'processed', 'dataset_final.parquet')

# ── FEATURES (OPTIMIZADAS) ─────────────────────────────────────────
FEATURES = [
    "retorno_log",
    "rsi_14",
    "macd",
    "volatilidad_20",
    "volume",
    "sent_pos",
    "sent_neg",
    "sent_neu"
]

# ══════════════════════════════════════════════════════════════════
#  CARGA + TARGET MEJORADO
# ══════════════════════════════════════════════════════════════════

def cargar_datos():
    df = pd.read_parquet(PATH_DS)
    df["dt"] = pd.to_datetime(df["dt"])

    # 🎯 NUEVO TARGET (3 días adelante)
    df["target"] = (df["retorno_log"].shift(-3) > 0).astype(int)

    df = df.dropna()

    print("\nBalance de clases:")
    print(df["target"].value_counts(normalize=True))

    return df


# ══════════════════════════════════════════════════════════════════
#  SPLIT TEMPORAL
# ══════════════════════════════════════════════════════════════════

def split(df):
    fechas = np.sort(df["dt"].unique())

    c1 = fechas[int(len(fechas) * 0.7)]
    c2 = fechas[int(len(fechas) * 0.85)]

    train = df[df["dt"] < c1].copy()
    val   = df[(df["dt"] >= c1) & (df["dt"] < c2)].copy()
    test  = df[df["dt"] >= c2].copy()

    scaler = MinMaxScaler()
    train[FEATURES] = scaler.fit_transform(train[FEATURES])
    val[FEATURES]   = scaler.transform(val[FEATURES])
    test[FEATURES]  = scaler.transform(test[FEATURES])

    return train, val, test


def xy(df):
    return df[FEATURES].values, df["target"].values


# ══════════════════════════════════════════════════════════════════
#  MODELO
# ══════════════════════════════════════════════════════════════════

def entrenar_rf(train, val):
    X_tr, y_tr = xy(train)
    X_v,  y_v  = xy(val)

    model = RandomForestClassifier(
        n_estimators=500,
        max_depth=6,
        min_samples_leaf=50,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )

    model.fit(X_tr, y_tr)

    pred = model.predict(X_v)

    acc = accuracy_score(y_v, pred)
    f1  = f1_score(y_v, pred)

    print("\nVALIDACIÓN")
    print(f"Accuracy: {acc:.4f}")
    print(f"F1-score: {f1:.4f}")
    print(classification_report(y_v, pred))

    return model


# ══════════════════════════════════════════════════════════════════
#  TEST FINAL
# ══════════════════════════════════════════════════════════════════

def evaluar(model, test):
    X_t, y_t = xy(test)
    pred = model.predict(X_t)

    acc = accuracy_score(y_t, pred)
    f1  = f1_score(y_t, pred)

    print("\nTEST FINAL")
    print(f"Accuracy: {acc:.4f}")
    print(f"F1-score: {f1:.4f}")
    print(classification_report(y_t, pred))


# ══════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════

def main():
    print("\n=== MODELO OPTIMIZADO ===")

    df = cargar_datos()
    train, val, test = split(df)

    model = entrenar_rf(train, val)
    evaluar(model, test)


if __name__ == "__main__":
    main()