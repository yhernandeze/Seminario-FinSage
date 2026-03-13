import pandas as pd
import os
from transformers import pipeline
from sentence_transformers import SentenceTransformer

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_PATH = os.path.join(BASE_DIR, 'data', 'processed', 'noticias_limpias.csv')
OUTPUT_PATH = os.path.join(BASE_DIR, 'data', 'processed', 'noticias_enriquecidas.csv')

def ejecutar_clasificacion():
    if not os.path.exists(INPUT_PATH):
        print("Error: Archivo procesado no encontrado.")
        return

    df = pd.read_csv(INPUT_PATH)
    titulares = df['title_clean'].dropna().tolist()

    # Clasificación Zero-Shot (Enriquecida)
    classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
    clases = ["Corporate Mergers", "Earnings Report", "Macroeconomics", "Stock Volatility"]

    # Procesamiento (ejemplo con los primeros 20 para la demo)
    resultados = classifier(titulares[:20], clases)
    df_res = pd.DataFrame(resultados)
    df.loc[:19, 'category'] = df_res['labels'].apply(lambda x: x[0])
    df.loc[:19, 'score'] = df_res['scores'].apply(lambda x: x[0])

    # Generación de Embeddings para LSTM/GRU
    embedder = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = embedder.encode(titulares[:20])

    df.to_csv(OUTPUT_PATH, index=False)
    print(f"Proceso finalizado. Resultados en: {OUTPUT_PATH}")

if __name__ == "__main__":
    ejecutar_clasificacion()