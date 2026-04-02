# 00_muestra_inteligente.py
"""
Paso previo al pipeline de FinBERT.
Filtra las noticias para quedarse SOLO con tickers que existen
en el BD.parquet y toma una muestra equilibrada de 300.

Orden de ejecución:
    00_muestra_inteligente.py  ← NUEVO
    01_limpieza.py
    02_vocab_ner.py
    03_clasificacion.py
"""
import pandas as pd
import os

BASE_DIR     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PATH_BD      = os.path.join(BASE_DIR, 'data', 'raw', 'BD.parquet2')
PATH_NOTICIAS = os.path.join(BASE_DIR, 'data', 'raw', 'noticias_exportadas.csv')
OUTPUT_PATH  = os.path.join(BASE_DIR, 'data', 'raw', 'noticias_exportadas.csv')  # sobreescribe el original

MUESTRA_TOTAL    = 300   # noticias a procesar con FinBERT
NOTICIAS_POR_TICKER = 5  # máximo por ticker para que sea equilibrado


def ejecutar():
    # ── 1. Cargar tickers del BD.parquet ─────────────────────────
    if not os.path.exists(PATH_BD):
        print(f"Error: No se encontró BD.parquet en {PATH_BD}")
        return

    bd = pd.read_parquet(PATH_BD)
    tickers_bd = set(bd['ticker'].unique())
    print(f"Tickers en BD.parquet        : {len(tickers_bd):,}")

    # ── 2. Cargar noticias ────────────────────────────────────────
    if not os.path.exists(PATH_NOTICIAS):
        print(f"Error: No se encontró noticias_exportadas.csv en {PATH_NOTICIAS}")
        return

    df = pd.read_csv(PATH_NOTICIAS, on_bad_lines='skip', encoding='utf-8')
    df.columns = [c.strip().lower() for c in df.columns]

    if 'ticker' not in df.columns:
        print(f"Error: columna 'ticker' no encontrada. Columnas: {df.columns.tolist()}")
        return

    print(f"Noticias totales en CSV      : {len(df):,}")
    print(f"Tickers únicos en noticias   : {df['ticker'].nunique():,}")

    # ── 3. Filtrar solo tickers que están en el BD ────────────────
    df_filtrado = df[df['ticker'].isin(tickers_bd)].copy()
    tickers_en_comun = df_filtrado['ticker'].nunique()
    print(f"\nTickers en común (BD ∩ noticias): {tickers_en_comun:,}")
    print(f"Noticias de esos tickers        : {len(df_filtrado):,}")

    if len(df_filtrado) == 0:
        print("\nError: No hay tickers en común. Verifica que los nombres coincidan.")
        print(f"  Ejemplos BD       : {list(tickers_bd)[:5]}")
        print(f"  Ejemplos noticias : {df['ticker'].unique()[:5].tolist()}")
        return

    # ── 4. Muestra equilibrada: máx N noticias por ticker ─────────
    noticias_por_ticker = max(
        1,
        min(NOTICIAS_POR_TICKER, MUESTRA_TOTAL // tickers_en_comun)
    )

    df_muestra = (
        df_filtrado
        .groupby('ticker', group_keys=False)
        .apply(lambda g: g.sample(
            min(len(g), noticias_por_ticker),
            random_state=42
        ))
        .reset_index(drop=True)
    )

    # Si sobran cupos (tickers con menos noticias de lo esperado),
    # rellenar con más noticias de los tickers con más disponibilidad
    if len(df_muestra) < MUESTRA_TOTAL:
        ya_incluidos = df_muestra.index
        restantes = df_filtrado[~df_filtrado.index.isin(ya_incluidos)]
        faltantes  = MUESTRA_TOTAL - len(df_muestra)
        if len(restantes) > 0:
            extra = restantes.sample(min(faltantes, len(restantes)), random_state=42)
            df_muestra = pd.concat([df_muestra, extra]).reset_index(drop=True)

    df_muestra = df_muestra.head(MUESTRA_TOTAL)

    # ── 5. Diagnóstico final ──────────────────────────────────────
    print(f"\n{'─'*45}")
    print(f"  Muestra final           : {len(df_muestra):,} noticias")
    print(f"  Tickers representados   : {df_muestra['ticker'].nunique():,}")
    print(f"  Noticias por ticker     : ~{len(df_muestra) / df_muestra['ticker'].nunique():.1f}")
    print(f"\n  Distribución por ticker:")
    dist = df_muestra['ticker'].value_counts()
    print(dist.to_string())

    # ── 6. Guardar — sobreescribe noticias_exportadas.csv ─────────
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    df_muestra.to_csv(OUTPUT_PATH, index=False)
    print(f"\n  Archivo guardado: {OUTPUT_PATH}")
    print(f"  Listo para correr: 01_limpieza.py → 02_vocab_ner.py → 03_clasificacion.py")


if __name__ == "__main__":
    ejecutar()