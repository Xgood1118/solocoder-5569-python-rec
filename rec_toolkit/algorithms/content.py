import numpy as np
from typing import List, Dict, Optional, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import jieba


class ContentBasedRecommender:
    def __init__(self, method: str = 'tfidf', embedding_dim: int = 128,
                 pretrained_model: Optional[str] = None, top_k: int = 50):
        self.method = method
        self.embedding_dim = embedding_dim
        self.pretrained_model = pretrained_model
        self.top_k = top_k

        self.tfidf_vectorizer: Optional[TfidfVectorizer] = None
        self.item_vectors: Optional[np.ndarray] = None
        self.item_ids: List[str] = []
        self.item2idx: Dict[str, int] = {}

    def _chinese_tokenize(self, text: str) -> List[str]:
        if not text:
            return []
        words = jieba.cut(text)
        return [w for w in words if w.strip()]

    def fit(self, items_df):
        self.item_ids = list(items_df['item_id'].astype(str).values)
        self.item2idx = {iid: i for i, iid in enumerate(self.item_ids)}

        texts = self._prepare_texts(items_df)

        if self.method == 'tfidf':
            self._fit_tfidf(texts)
        elif self.method == 'embedding':
            self._fit_embedding(texts)
        else:
            raise ValueError(f"未知的方法: {self.method}")

    def _prepare_texts(self, items_df) -> List[str]:
        texts = []
        for _, row in items_df.iterrows():
            parts = []
            if 'title' in row and row.get('title'):
                parts.append(str(row['title']))
            if 'description' in row and row.get('description'):
                parts.append(str(row['description']))
            if 'tags' in row and row.get('tags'):
                parts.append(str(row['tags']).replace(',', ' '))
            if 'category' in row and row.get('category'):
                parts.append(str(row['category']))
            texts.append(' '.join(parts))
        return texts

    def _fit_tfidf(self, texts: List[str]):
        self.tfidf_vectorizer = TfidfVectorizer(
            tokenizer=self._chinese_tokenize,
            max_features=5000,
            token_pattern=None,
        )
        self.item_vectors = self.tfidf_vectorizer.fit_transform(texts).toarray()

    def _fit_embedding(self, texts: List[str]):
        n_items = len(texts)
        if self.pretrained_model:
            self.item_vectors = self._load_pretrained_embeddings(texts)
        else:
            np.random.seed(42)
            self.item_vectors = np.random.randn(n_items, self.embedding_dim)
            norms = np.linalg.norm(self.item_vectors, axis=1, keepdims=True)
            self.item_vectors = self.item_vectors / norms

    def _load_pretrained_embeddings(self, texts: List[str]) -> np.ndarray:
        n_items = len(texts)
        embeddings = np.zeros((n_items, self.embedding_dim))
        for i, text in enumerate(texts):
            tokens = self._chinese_tokenize(text)
            if tokens:
                vec = np.mean([np.random.randn(self.embedding_dim) for _ in tokens[:10]], axis=0)
                embeddings[i] = vec / (np.linalg.norm(vec) + 1e-10)
        return embeddings

    def get_similar_items(self, item_id: str, n: int = 10) -> Tuple[List[str], List[float]]:
        if self.item_vectors is None or item_id not in self.item2idx:
            return [], []

        idx = self.item2idx[item_id]
        query_vec = self.item_vectors[idx].reshape(1, -1)

        similarities = cosine_similarity(query_vec, self.item_vectors)[0]
        similarities[idx] = -1

        top_indices = np.argsort(similarities)[::-1][:n]
        top_items = [self.item_ids[i] for i in top_indices]
        top_scores = [float(similarities[i]) for i in top_indices]

        return top_items, top_scores

    def recommend(self, user_profile, n_items: int = 10,
                  exclude_items: Optional[List[str]] = None) -> Tuple[List[str], List[float]]:
        if self.item_vectors is None:
            return [], []

        user_vector = self._build_user_vector(user_profile)
        if user_vector is None:
            return [], []

        scores = cosine_similarity(user_vector.reshape(1, -1), self.item_vectors)[0]

        if exclude_items:
            for item_id in exclude_items:
                if item_id in self.item2idx:
                    scores[self.item2idx[item_id]] = -1

        top_indices = np.argsort(scores)[::-1][:n_items]
        top_items = [self.item_ids[i] for i in top_indices]
        top_scores = [float(scores[i]) for i in top_indices]

        valid = [s > 0 for s in top_scores]
        return [ti for ti, v in zip(top_items, valid) if v], \
               [ts for ts, v in zip(top_scores, valid) if v]

    def _build_user_vector(self, user_profile) -> Optional[np.ndarray]:
        if self.item_vectors is None:
            return None

        vectors = []
        weights = []

        if hasattr(user_profile, 'history_ratings') and user_profile.history_ratings:
            for item_id, rating in user_profile.history_ratings.items():
                if item_id in self.item2idx:
                    idx = self.item2idx[item_id]
                    vectors.append(self.item_vectors[idx])
                    weights.append(rating)

        if hasattr(user_profile, 'interests') and user_profile.interests:
            interest_text = ' '.join(user_profile.interests)
            if self.tfidf_vectorizer:
                interest_vec = self.tfidf_vectorizer.transform([interest_text]).toarray()[0]
                vectors.append(interest_vec)
                weights.append(1.0)

        if not vectors:
            if hasattr(user_profile, 'interests') and user_profile.interests:
                return np.random.randn(self.item_vectors.shape[1])
            return None

        weights = np.array(weights)
        weights = weights / weights.sum()
        user_vector = np.average(vectors, axis=0, weights=weights)

        norm = np.linalg.norm(user_vector)
        if norm > 0:
            user_vector = user_vector / norm

        return user_vector

    def recommend_by_item(self, item_id: str, n: int = 10) -> Tuple[List[str], List[float]]:
        return self.get_similar_items(item_id, n)
