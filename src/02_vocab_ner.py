import pandas as pd
import spacy
import os
from transformers import AutoTokenizer, AutoModel

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_PATH = os.path.join(BASE_DIR, 'data', 'processed', 'noticias_limpias.csv')

def ejecutar_ner():
    if not os.path.exists(INPUT_PATH):
        print("Error: Ejecute primero el script de limpieza.")
        return

    df = pd.read_csv(INPUT_PATH)
    nlp = spacy.load("en_core_web_sm")
    tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
    model = AutoModel.from_pretrained("ProsusAI/finbert")

    entidades = set()
    for texto in df['title_clean'].dropna():
        doc = nlp(texto)
        for ent in doc.ents:
            if ent.label_ in ['ORG', 'PRODUCT', 'MONEY']:
                entidades.add(ent.text)

    nuevos_tokens = [e for e in entidades if len(tokenizer.tokenize(e)) > 1]

    if nuevos_tokens:
        tokenizer.add_tokens(nuevos_tokens)
        model.resize_token_embeddings(len(tokenizer))
        print(f"Vocabulario actualizado con {len(nuevos_tokens)} nuevos términos.")
    else:
        print("El modelo ya conoce todas las entidades detectadas.")

if __name__ == "__main__":
    ejecutar_ner()