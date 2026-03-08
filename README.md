# FinSage: Análisis de Sentimiento de Noticias Financieras para Estrategias de Inversión en el S&P 500

## Descripción del Proyecto

Este proyecto tiene como objetivo **mejorar la precisión predictiva y la rentabilidad de estrategias de inversión en el índice S&P 500** mediante el análisis de sentimiento de titulares de noticias financieras.

Para ello se recolectarán titulares desde Google News mediante técnicas de **web scraping**, y posteriormente se aplicarán técnicas de **Procesamiento de Lenguaje Natural (NLP)** para transformar información textual en **señales cuantitativas de sentimiento de mercado**.

A diferencia de enfoques tradicionales que solo clasifican noticias como **positivas, negativas o neutrales**, este proyecto propone generar un **score continuo de sentimiento** utilizando **FinBERT**, un modelo de lenguaje entrenado específicamente con textos financieros.

Este score permitirá construir un **índice de sentimiento del mercado** que posteriormente será analizado en relación con el comportamiento del S&P 500.

---

# Objetivos

## Objetivo general

Desarrollar un modelo basado en técnicas de procesamiento de lenguaje natural que permita extraer señales de sentimiento del mercado a partir de titulares de noticias financieras relacionadas con el S&P 500.

## Objetivos específicos

* Recolectar titulares de noticias financieras mediante web scraping desde Google News.
* Realizar limpieza y preprocesamiento del texto.
* Aplicar modelos avanzados de NLP entrenados en lenguaje financiero.
* Generar un **score continuo de sentimiento** para cada noticia.
* Construir un **índice de sentimiento del mercado** agregando noticias diarias.
* Analizar la relación entre el sentimiento extraído y el comportamiento del S&P 500.
* Evaluar el potencial del sentimiento como señal para estrategias de inversión.

---

# Metodología

El proyecto sigue un pipeline de procesamiento de datos basado en técnicas modernas de NLP:

Recolección de datos → Preprocesamiento de texto → Análisis de sentimiento con FinBERT → Generación de score → Construcción del índice de sentimiento → Evaluación frente al mercado

---

# 1. Recolección de datos

Los datos serán recolectados mediante **web scraping de titulares de noticias financieras desde Google News**.

Cada registro incluirá:

* fecha de publicación
* titular de la noticia
* fuente
* enlace de la noticia

Los datos recolectados se almacenarán inicialmente en formato **CSV** para su posterior procesamiento.

---

# 2. Preprocesamiento del texto

Antes de aplicar modelos de NLP, los textos serán limpiados y normalizados.

El proceso incluye:

* Conversión a minúsculas
* Eliminación de caracteres especiales
* Eliminación de stopwords
* Tokenización
* Lematización
* Normalización del texto

Este paso permite preparar los textos para su análisis mediante modelos de lenguaje.

---

# 3. Análisis de sentimiento con FinBERT

Para capturar correctamente el contexto financiero de las noticias se utilizará **FinBERT**, un modelo basado en la arquitectura BERT entrenado específicamente con textos financieros.

FinBERT permite identificar el sentimiento de un texto en tres categorías:

* Positivo
* Neutral
* Negativo

En lugar de utilizar directamente estas clases, se construirá un **score continuo de sentimiento**.

---

# 4. Construcción del Score de Sentimiento

El modelo generará probabilidades para cada clase de sentimiento. A partir de estas probabilidades se calculará un **score de impacto del mercado**:

Score = Probabilidad(Positivo) − Probabilidad(Negativo)

Este score tendrá un rango aproximado entre:

* **-1 → Sentimiento muy negativo**
* **0 → Sentimiento neutral**
* **+1 → Sentimiento muy positivo**

Esto permite representar el impacto potencial de cada noticia en el mercado de forma cuantitativa.

---

# 5. Construcción del Índice de Sentimiento del Mercado

Los scores individuales de las noticias serán agregados para construir un **Market Sentiment Index**.

Por ejemplo, calculando el promedio diario:

Sentiment_Index_día = Promedio(score_noticias_día)

Este índice permitirá analizar si el sentimiento agregado del mercado tiene relación con el comportamiento del S&P 500.

---

# 6. Evaluación del Modelo

El índice de sentimiento será comparado con datos históricos del mercado para evaluar su utilidad como señal predictiva.

Se analizarán aspectos como:

* correlación entre sentimiento y retornos del mercado
* capacidad predictiva en modelos de regresión
* comportamiento de estrategias de inversión basadas en sentimiento

---

# Estructura del repositorio

El proyecto sigue una estructura modular diseñada para facilitar el trabajo colaborativo y la reproducibilidad del análisis.

```
Seminario-FinSage
│
├── data
│   ├── raw                # Datos originales recolectados mediante scraping
│   ├── processed          # Datos limpios y preparados para el modelo
│
├── notebooks
│   ├── 01_scraping.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_feature_engineering.ipynb
│   ├── 04_finbert_sentiment.ipynb
│
├── src
│   ├── scraping.py           # Recolección de noticias
│   ├── preprocessing.py      # Limpieza y normalización de texto
│   ├── features.py           # Transformación de texto (TF-IDF / N-grams)
│   ├── finbert_model.py      # Implementación del modelo FinBERT
│   ├── sentiment_index.py    # Construcción del índice de sentimiento
│   ├── model.py              # Modelos de análisis del impacto en el mercado
│
├── results
│   ├── models                # Modelos entrenados
│   ├── figures               # Gráficos y visualizaciones
│
├── requirements.txt          # Dependencias del proyecto
├── README.md                 # Documentación del proyecto
└── .gitignore
```

---

# Tecnologías utilizadas

El proyecto se desarrollará utilizando las siguientes herramientas:

* Python
* Pandas
* NumPy
* Scikit-learn
* Transformers (HuggingFace)
* PyTorch
* BeautifulSoup
* Requests
* NLTK / SpaCy
* Matplotlib
* Seaborn
* Jupyter Notebook

---

# Trabajo colaborativo

El proyecto será desarrollado utilizando control de versiones con Git.

Se recomienda trabajar mediante **ramas de desarrollo**, por ejemplo:

* scraping
* preprocessing
* sentiment-model
* sentiment-index
* modeling

Los cambios serán integrados mediante **Pull Requests** para mantener un flujo de trabajo ordenado.

---

# Trabajo futuro

Posibles extensiones del proyecto incluyen:

* Incorporación de más fuentes de noticias financieras
* Análisis de temas mediante Topic Modeling
* Incorporación de pesos de relevancia de noticias
* Construcción de un indicador avanzado de **News Impact Score**
* Evaluación de estrategias de trading basadas en señales de sentimiento

---

# Autores

Proyecto desarrollado como parte del seminario de Inteligencia Artificial aplicado a Finanzas.
