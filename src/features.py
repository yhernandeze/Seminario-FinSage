from sklearn.feature_extraction.text import TfidfVectorizer
def build_tfidf(corpus):
    vectorizer = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1,2)
    )
    X = vectorizer.fit_transform(corpus)
    return X, vectorizer