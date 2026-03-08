# Seminario-FinSage

# Análisis de Sentimiento de Noticias para Estrategias de Inversión en el S&P 500

## Descripción del Proyecto

Este proyecto tiene como objetivo mejorar la **precisión predictiva y la rentabilidad de estrategias de inversión** en el índice S&P 500 mediante el análisis de sentimiento de titulares de noticias financieras.

Para ello se recolectarán titulares desde Google News mediante técnicas de **web scraping**, y posteriormente se aplicarán técnicas de **Procesamiento de Lenguaje Natural (NLP)** para transformar información textual en **señales cuantitativas** que permitan inferir el sentimiento del mercado.

A diferencia de enfoques tradicionales de clasificación (positivo, negativo o neutral), este proyecto busca generar un **score continuo de sentimiento del mercado**, el cual posteriormente se analizará en relación con los movimientos del S&P 500.

---

# Objetivos

## Objetivo general

Desarrollar un modelo basado en técnicas de procesamiento de lenguaje natural que permita extraer señales de sentimiento del mercado a partir de titulares de noticias financieras relacionadas con el S&P 500.

## Objetivos específicos

* Recolectar titulares de noticias financieras mediante web scraping desde Google News.
* Realizar un proceso de limpieza y preprocesamiento del texto.
* Transformar los textos en representaciones numéricas mediante técnicas de NLP.
* Construir un modelo que genere un **score de sentimiento de mercado**.
* Analizar la relación entre el sentimiento extraído de las noticias y el comportamiento del S&P 500. (Agente de portafolio)
* Evaluar la utilidad del sentimiento como señal para estrategias de inversión.

---

# Metodología

El proyecto sigue el siguiente flujo metodológico:

Recolección de datos → Preprocesamiento de texto → Extracción de características → Modelado de sentimiento → Construcción de índice de sentimiento → Evaluación frente al mercado

---

# 1. Recolección de datos

Se realizará **web scraping de titulares de noticias financieras desde Google News**, recolectando información como:

* fecha de publicación
* titular de la noticia
* fuente
* enlace de la noticia

Los datos recolectados se almacenarán inicialmente en formato CSV para su posterior procesamiento.

---

# 2. Preprocesamiento del texto

El texto será limpiado y normalizado para facilitar su análisis mediante técnicas de NLP. El proceso incluirá:

* Conversión a minúsculas
* Eliminación de caracteres especiales
* Eliminación de stopwords
* Tokenización
* Lematización
* Stemming

Este paso permite transformar el texto en una forma más adecuada para el modelado computacional.

---

# 3. Extracción de características

Para representar el texto como datos numéricos se utilizarán distintas técnicas de extracción de características:

### Bag of Words (BoW)

Representación basada en la frecuencia de aparición de palabras dentro del conjunto de documentos.

### N-gramas de palabras

Capturan relaciones entre palabras consecutivas para mejorar la comprensión del contexto.

Ejemplos:

* market rally
* inflation surprise
* recession fears

### N-gramas de caracteres

Permiten capturar patrones en palabras incluso cuando existen variaciones o errores ortográficos.

### TF-IDF (Term Frequency – Inverse Document Frequency)

Permite ponderar la importancia de las palabras dentro de un documento respecto al conjunto total de documentos.

---

# 4. Modelado de sentimiento

En lugar de clasificar las noticias únicamente en positivo, negativo o neutral, el modelo generará un **score continuo de sentimiento**, representando el posible impacto de la noticia en el mercado.

Ejemplo de escala:

-1 → Sentimiento muy negativo
0 → Sentimiento neutral
+1 → Sentimiento muy positivo

Este score permitirá construir indicadores agregados de sentimiento del mercado.

---

# 5. Índice de sentimiento del mercado

A partir de los scores individuales de cada noticia se construirá un **Market Sentiment Index**, agregando el sentimiento promedio de las noticias publicadas en un periodo determinado (por ejemplo, diariamente).

Este índice será utilizado para analizar su relación con los rendimientos del S&P 500.

---

# Estructura del repositorio

El proyecto sigue una estructura modular para facilitar el trabajo colaborativo y la reproducibilidad.

```
sp500-news-sentiment/

│
├── data/
│   ├── raw/                 # Datos originales recolectados mediante scraping
│   ├── processed/           # Datos preprocesados listos para modelado
│
├── notebooks/
│   ├── 01_scraping.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_feature_engineering.ipynb
│   ├── 04_model_training.ipynb
│
├── src/
│   ├── scraping.py          # Funciones para recolección de noticias
│   ├── preprocessing.py     # Limpieza y procesamiento de texto
│   ├── features.py          # Extracción de características (TF-IDF, n-grams)
│   ├── model.py             # Entrenamiento de modelos de NLP
│   ├── sentiment_index.py   # Construcción del índice de sentimiento
│
├── resul
```

