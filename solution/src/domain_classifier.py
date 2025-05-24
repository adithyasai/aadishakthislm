# src/domain_classifier.py
"""
Lightweight domain classifier for texts into 'legal', 'medical', or 'technical'.
"""
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

# Sample training data for domain classification
TRAIN_DATA = [
    ("This agreement is made between the parties and shall be governed by law.", "legal"),
    ("The plaintiff filed a claim in the court against the defendant.", "legal"),
    ("The patient exhibited symptoms of hypertension and required treatment.", "medical"),
    ("A diagnosis of pneumonia was confirmed by the radiologist.", "medical"),
    ("We implemented a new algorithm for better performance of the database.", "technical"),
    ("The software engineer optimized the code and fixed a bug.", "technical"),
]

# Train a simple TF-IDF + Naive Bayes classifier
_texts, _labels = zip(*TRAIN_DATA)
_vectorizer = TfidfVectorizer()
_features = _vectorizer.fit_transform(_texts)
_classifier = MultinomialNB()
_classifier.fit(_features, _labels)

def classify_domain(text: str) -> str:
    """
    Classify input text into one of 'legal', 'medical', or 'technical'.
    Returns 'unknown' if text is empty.
    """
    if not text or not text.strip():
        return "unknown"
    features = _vectorizer.transform([text])
    return _classifier.predict(features)[0]
