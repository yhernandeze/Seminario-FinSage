import pandas as pd
import re
import os

# Configuración de rutas dinámicas absolutas
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_PATH = os.path.join(BASE_DIR, 'data', 'raw', 'noticias_exportadas.csv')
OUTPUT_PATH = os.path.join(BASE_DIR, 'data', 'processed', 'noticias_limpias.csv')

def limpiar_texto(texto):
    if not isinstance(texto, str): return ""
    # Mantiene la estructura para WordPiece (puntuación y conectores)
    texto = re.sub(r'http\S+|www\S+|https\S+', '', texto, flags=re.MULTILINE)
    texto = re.sub(r'[^\w\s.,!?\'"-]', '', texto)
    return " ".join(texto.split())

def ejecutar_limpieza():
    if not os.path.exists(INPUT_PATH):
        print(f"Error: No se encontró el archivo en {INPUT_PATH}")
        return

    # Intentar leer con detección automática de separador (sep=None)
    try:
        df = pd.read_csv(INPUT_PATH, sep=None, engine='python', on_bad_lines='skip')
        
        # Limpiar nombres de columnas: quitar espacios y pasar a minúsculas
        df.columns = [str(c).strip().lower() for c in df.columns]
        
        # Filtrar columnas que no sean 'unnamed'
        df = df[[c for c in df.columns if 'unnamed' not in c]]
        
        # Buscar la columna que contenga el título
        col_titulo = None
        if 'title' in df.columns:
            col_titulo = 'title'
        else:
            # Si el CSV tiene una columna pegada como 'ticker,date,title...', la expandimos
            for c in df.columns:
                if 'title' in c:
                    col_titulo = c
                    break

        if col_titulo:
            df['title_clean'] = df[col_titulo].apply(limpiar_texto)
            os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
            df.to_csv(OUTPUT_PATH, index=False)
            print(f"Limpieza finalizada con éxito.")
            print(f"Archivo guardado en: {OUTPUT_PATH}")
            print(f"Columnas finales: {df.columns.tolist()}")
        else:
            print(f"Error: No se encontró columna de títulos. Columnas detectadas: {df.columns.tolist()}")

    except Exception as e:
        print(f"Error al procesar el CSV: {e}")

if __name__ == "__main__":
    ejecutar_limpieza()