import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
try:
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer = None

class DenseRetriever:
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2', use_model: bool = False):
        self.model_name = model_name
        self.use_model = use_model
        self.model = None
        self.embeddings = None
        self.docs = []
        self.vectorizer = None

    def build_index(self, docs):
        self.docs = docs
        if self.use_model and SentenceTransformer is not None:
            try:
                self.model = SentenceTransformer(self.model_name)
                self.embeddings = self.model.encode(docs, show_progress_bar=False, convert_to_numpy=True)
                return
            except Exception:
                self.model = None
        # fallback to tfidf
        self.vectorizer = TfidfVectorizer().fit(docs)
        self.embeddings = self.vectorizer.transform(docs)

    def search(self, query: str, top_k: int = 10):
        if self.embeddings is None:
            return []
        if self.model is not None:
            q_emb = self.model.encode([query], convert_to_numpy=True)
            sims = cosine_similarity(q_emb, self.embeddings)[0]
        else:
            q_vec = self.vectorizer.transform([query])
            sims = cosine_similarity(q_vec, self.embeddings)[0]
        ranked = sorted(enumerate(sims), key=lambda x: x[1], reverse=True)[:top_k]
        return ranked
