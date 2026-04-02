import pandas as pd
import re
import os
import csv

# Configuración de rutas
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_PATH = os.path.join(BASE_DIR, 'data', 'raw', 'noticias_exportadas.csv')
OUTPUT_PATH = os.path.join(BASE_DIR, 'data', 'processed', 'noticias_limpias.csv')

def filtrar_caracteres(texto):
    if not isinstance(texto, str): return ""
    # Quitar URLs
    texto = re.sub(r'http\S+|www\S+|https\S+', '', texto, flags=re.MULTILINE)
    # Quitar iconos/emojis pero mantener puntuación clave para FinBERT (. , ? ! -)
    texto = re.sub(r'[^\w\s.,!?\-]', '', texto)
    return " ".join(texto.split())

def ejecutar_limpieza():
    if not os.path.exists(INPUT_PATH):
        print(f"No se encontró el archivo inicial en: {INPUT_PATH}")
        return

    try:
        # Ajuste de ingeniería: añadimos quoting y on_bad_lines para saltar errores de comillas
        df = pd.read_csv(
            INPUT_PATH, 
            sep=None, 
            engine='python', 
            quoting=csv.QUOTE_MINIMAL, # Maneja comillas de forma estándar
            on_bad_lines='skip',       # Si una línea está muy rota, la salta para no detener el proceso
            encoding='utf-8'           # Asegura la lectura de caracteres latinos
        )
        
        df.columns = [c.strip().lower() for c in df.columns]
        
        # Buscamos la columna de título
        col_objetivo = None
        for col in df.columns:
            if 'title' in col:
                col_objetivo = col
                break

        if col_objetivo:
            df['title_clean'] = df[col_objetivo].apply(filtrar_caracteres)
            
            # Limpieza extra: quitar filas donde el título haya quedado vacío
            df = df.dropna(subset=['title_clean'])
            
            os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
            df.to_csv(OUTPUT_PATH, index=False)
            print(f"Archivo de noticias limpias generado en: {OUTPUT_PATH}")
        else:
            print(f"Error: No se detectó la columna 'title'. columnas actuales: {df.columns.tolist()}")

    except Exception as e:
        print(f"Error crítico al leer el CSV: {e}")

if __name__ == "__main__":
    ejecutar_limpieza()
