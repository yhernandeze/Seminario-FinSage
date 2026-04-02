import pandas as pd
import os
import spacy

# Configuración de rutas
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_PATH = os.path.join(BASE_DIR, 'data', 'processed', 'noticias_limpias.csv')
OUTPUT_PATH = os.path.join(BASE_DIR, 'data', 'processed', 'noticias_enriquecidas.csv')

def ejecutar_ner_ligero():
    if not os.path.exists(INPUT_PATH):
        print("Error: No se encontró noticias_limpias.csv")
        return

    print("Cargando modelo de lenguaje simplificado...")
    # Cargamos el modelo
    try:
        nlp = spacy.load("en_core_web_sm", disable=["parser", "projects"])
    except:
        print("Descargando modelo necesario...")
        os.system("python3 -m spacy download en_core_web_sm")
        nlp = spacy.load("en_core_web_sm")

    df = pd.read_csv(INPUT_PATH)
    
    # Procesamos solo una muestra de 300 para tu validación
    df_muestra = df.copy()
    
    print("Extrayendo entidades (NER) de forma secuencial...")
    
    def extraer_entidades(texto):
        doc = nlp(str(texto))
        # Solo extraemos 
        entidades = [ent.text for ent in doc.ents if ent.label_ == "ORG"]
        return ", ".join(entidades) if entidades else "None"

    # Aplicamos fila por fila sin usar hilos paralelos (evita el error de mutex)
    df_muestra['entidades_org'] = df_muestra['title_clean'].apply(extraer_entidades)

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    df_muestra.to_csv(OUTPUT_PATH, index=False)
    print(f"Completado con éxito. Archivo en: {OUTPUT_PATH}")

if __name__ == "__main__":
    ejecutar_ner_ligero()
