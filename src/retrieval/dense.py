import os

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

try:
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer = None


class DenseRetriever:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", use_model: bool = True):
        self.model_name = model_name
        self.use_model = use_model
        self.model = None
        self.embeddings = None
        self.docs = []
        self.vectorizer = None

    def build_index(self, docs):
        self.docs = list(docs)
        if self.use_model and SentenceTransformer is not None:
            try:
                self.model = SentenceTransformer(self.model_name)
                self.embeddings = self.model.encode(
                    self.docs,
                    show_progress_bar=True,
                    convert_to_numpy=True,
                    batch_size=64,
                )
                return
            except Exception:
                self.model = None
        # TF-IDF fallback (used in unit tests and when ST unavailable)
        self.vectorizer = TfidfVectorizer().fit(self.docs)
        self.embeddings = self.vectorizer.transform(self.docs)

    def save_index(self, directory: str) -> None:
        """Persist sentence-transformer embeddings to directory/dense_embeddings.npy."""
        np.save(os.path.join(directory, "dense_embeddings.npy"), self.embeddings)

    def load_index(self, directory: str, docs: list) -> None:
        """Load pre-computed embeddings and warm the model for query-time encoding."""
        self.docs = list(docs)
        self.embeddings = np.load(os.path.join(directory, "dense_embeddings.npy"))
        if SentenceTransformer is None:
            raise RuntimeError(
                "sentence-transformers is not installed; cannot encode queries against loaded embeddings."
            )
        self.model = SentenceTransformer(self.model_name)

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
