# FinSage: Análisis de Sentimiento de Noticias Financieras para Estrategias de Inversión en el S&P 500

## Descripción del Proyecto

Este proyecto tiene como objetivo **mejorar la precisión predictiva y la rentabilidad de estrategias de inversión en el índice S&P 500** mediante el análisis de sentimiento de titulares de noticias financieras.

Para ello se recolectarán titulares desde Google News mediante técnicas de **web scraping**, y posteriormente se aplicarán técnicas de **Procesamiento de Lenguaje Natural (NLP)** para transformar información textual en **señales cuantitativas de sentimiento de mercado**.

## Metodología del Módulo

### 1. Ingesta de Base de Datos
El sistema ingesta los datos desde un archivo tabular estático (`noticias_exportadas.csv`) que contiene los titulares financieros previamente extraídos. Esta etapa asegura que los textos se carguen correctamente, manejando delimitadores específicos y valores nulos antes de pasar al motor de procesamiento.

### 2. Preprocesamiento de Texto (WordPiece Friendly)
A diferencia de los pipelines tradicionales de NLP, este módulo omite intencionalmente técnicas destructivas como la lematización, el *stemming* o la eliminación de *stopwords*. 

La limpieza se restringe a:
* Eliminación de URLs y código residual.
* Estandarización de espacios y eliminación de caracteres no semánticos.
* Conservación intacta de la gramática, puntuación y conectores.

Esta decisión metodológica asegura que el tokenizador **WordPiece** nativo de FinBERT reciba el contexto sintáctico completo y bidireccional necesario para no perder la intención original de las oraciones financieras.

### 3. Validación NER y Expansión de Vocabulario
Para abordar el problema de palabras fuera de vocabulario (OOV) en un dominio altamente especializado, se implementa un modelo de Reconocimiento de Entidades Nombradas (NER) a través de `spaCy`. 
* Se identifican organizaciones, empresas y productos financieros en el texto limpio.
* Se valida si el tokenizador de FinBERT reconoce la palabra completa o si la fragmenta en sub-palabras.
* Los términos desconocidos se añaden dinámicamente al vocabulario del modelo y se redimensiona su matriz de *embeddings* para preservar la integridad de conceptos clave (ej. *tickers* bursátiles o nombres propios de empresas).

### 4. Clasificación Enriquecida (Zero-Shot & Sentence Transformers)
Para proporcionar un contexto de mayor valor a la arquitectura de agentes predictivos de FinSage, se descarta la clasificación de sentimiento de tres clases. En su lugar, se utilizan **Sentence Transformers** para vectorizar los titulares y un modelo de **Zero-Shot Classification** para categorizar las noticias en temáticas de dominio específico definidas por el investigador (ej. Fusiones y Adquisiciones, Reportes de Ganancias, Macroeconomía).

---

## Estructura del Repositorio

La arquitectura de carpetas refleja exclusivamente las fases de limpieza y modelado correspondientes a este entregable:

```text
FinSage-NLP-Module/
├── data/
│   ├── raw/                 # Base de datos original (noticias_exportadas.csv)
│   └── processed/           # Datos procesados, limpios y listos para inferencia
├── src/
│   ├── 01_limpieza.py       # Algoritmos de limpieza estructural (sin lematización)
│   ├── 02_vocab_ner.py      # Extracción de entidades y actualización de tokenizador
│   └── 03_clasificacion.py  # Vectorización y clasificación Zero-Shot
├── requirements.txt         # Dependencias del entorno de ejecución
└── README.md                # Documentación del módulo  

```
---
## Categoria y Score ¿Qué esperamos?
* Category: Es el tema de la noticia coincida lógicamente con la etiqueta asignada.

*** Macroeconomics: El modelo evalúa si el titular "implica" el concepto de macroeconomía. Relacionará automáticamente términos como Federal Reserve, Inflation, Interest Rates, GDP o Central Banks con esta categoría, porque en su entrenamiento previo aprendió que esos conceptos pertenecen al campo de la economía global.

*** Earnings Report: Buscará la relación semántica con el desempeño financiero de las empresas. Palabras como Quarterly results, Revenue, Profits, EPS, Fiscal year o Crashed/Soared after reporting activarán esta categoría.

*** Stock Volatility: Se activará con noticias que describen movimientos bruscos o incertidumbre en el mercado. Términos como Market plunge, Rally, Sell-off, Uncertainty, Bulls/Bears o Fluctuations tendrán un alto relacionamiento aquí.

*** Corporate Mergers: El modelo detectará la relación entre empresas. Términos como Acquisition, Buyout, Merger, Agreement to buy o Hostile takeover son los que dispararán esta clasificación.

* Score: Que las noticias muy claras tengan números altos (cercanos a 1) y las confusas tengan números bajos.
* Title_clean: Que el texto no tenga basura (URLs o símbolos raros) pero mantenga las palabras completas.
 
