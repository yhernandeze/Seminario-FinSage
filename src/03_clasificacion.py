import pandas as pd
import os
import requests
import time

# Configuración de API
API_TOKEN = "pon el token"
API_URL   = "https://router.huggingface.co/hf-inference/models/ProsusAI/finbert"
HEADERS   = {"Authorization": f"Bearer {API_TOKEN}"}

# Rutas
BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_PATH  = os.path.join(BASE_DIR, 'data', 'processed', 'noticias_enriquecidas.csv')
OUTPUT_PATH = os.path.join(BASE_DIR, 'data', 'processed', 'salida_sentimientos_finbert.csv')

# Etiquetas en español
ETIQUETAS_ES = {
    'positive': 'Positivo',
    'negative': 'Negativo',
    'neutral':  'Neutro'
}


def consultar_finbert(texto, reintentos=3):
    """
    Llama a la API de FinBERT con reintentos automáticos.
    Espera 30s si hay rate limit (429), 5s si hay otro error.
    """
    for intento in range(reintentos):
        try:
            response = requests.post(
                API_URL,
                headers=HEADERS,
                json={"inputs": texto},
                timeout=30
            )
            if response.status_code == 200:
                return response.json()
            elif response.status_code == 429:
                print(f"  Rate limit alcanzado. Esperando 30s...")
                time.sleep(30)
            else:
                print(f"  Error {response.status_code}, intento {intento + 1}/{reintentos}")
                time.sleep(5)
        except Exception as e:
            print(f"  Error de conexión: {e}, intento {intento + 1}/{reintentos}")
            time.sleep(5)
    return None


def extraer_scores(data):
    """
    Recibe la lista de dicts que retorna FinBERT y extrae
    positive, negative, neutral, total, label en español.
    """
    scores = {item['label'].lower(): round(item['score'], 4) for item in data}

    pos   = scores.get('positive', 0.0)
    neg   = scores.get('negative', 0.0)
    neu   = scores.get('neutral',  0.0)
    total = round(pos + neg + neu, 4)
    label = ETIQUETAS_ES.get(max(scores, key=scores.get), 'Neutro')

    return pos, neg, neu, total, label


def ejecutar_nlp_real():
    # ── Verificar que existe el archivo de entrada ─────────────────
    if not os.path.exists(INPUT_PATH):
        print(f"Error: No se encontró el archivo {INPUT_PATH}")
        print("Asegúrate de haber corrido 01_limpieza.py y 02_vocab_ner.py primero.")
        return

    df = pd.read_csv(INPUT_PATH)
    total_filas = len(df)
    print(f"Noticias a procesar: {total_filas}")
    print(f"Tickers únicos     : {df['ticker'].nunique()}")
    print("-" * 50)

    resultados = []

    for index, row in df.iterrows():
        texto = str(row.get('title_clean', ''))

        print(f"[{index + 1}/{total_filas}] {row.get('ticker', '?')} — {texto[:60]}...")

        # Llamada a la API
        respuesta = consultar_finbert(texto)

        if respuesta and isinstance(respuesta, list):
            # FinBERT puede retornar [[...]] o [...]
            data = respuesta[0] if isinstance(respuesta[0], list) else respuesta
            pos, neg, neu, total, label = extraer_scores(data)
        else:
            # Si la API falla completamente, asignar neutro por defecto
            pos, neg, neu, total, label = 0.0, 0.0, 1.0, 1.0, 'Neutro'
            print(f"  Sin respuesta válida — asignando Neutro por defecto")

        resultados.append({
            'positive':            pos,
            'negative':            neg,
            'neutral':             neu,
            'total_scores':        total,
            'sentimiento_finbert': label,
            'exactitud_finbert':   round(max(pos, neg, neu), 4)
        })

        # Pausa entre requests para respetar el free tier
        time.sleep(1.5)

    # Unir resultados con el DataFrame original
    df_resultados = pd.DataFrame(resultados)
    df_final = pd.concat([df.reset_index(drop=True), df_resultados], axis=1)

    # Guardar output
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    df_final.to_csv(OUTPUT_PATH, index=False)

    # Resumen final
    print("\n" + "=" * 70)
    print("Proceso Finalizado")
    print("=" * 70)

    cols_vista = [
        'ticker', 'positive', 'negative', 'neutral',
        'total_scores', 'sentimiento_finbert', 'exactitud_finbert'
    ]
    # Mostrar solo columnas que existan (por si ticker no está en el CSV)
    cols_existentes = [c for c in cols_vista if c in df_final.columns]
    print(df_final[cols_existentes].to_string(index=False))

    print(f"\nArchivo guardado en: {OUTPUT_PATH}")
    print(f"Total procesadas   : {len(df_final):,} noticias")
    print(f"Tickers únicos     : {df_final['ticker'].nunique():,}")

    # Distribución de sentimientos
    print("\nDistribución de sentimientos:")
    print(df_final['sentimiento_finbert'].value_counts().to_string())


if __name__ == "__main__":
    ejecutar_nlp_real()
