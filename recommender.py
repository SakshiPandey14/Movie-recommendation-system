"""
recommender.py — Vectorisation + cosine-similarity recommendation engine.
"""

import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ── Model class ───────────────────────────────────────────────────────────────

class ContentRecommender:
    """
    Content-based movie recommender built on CountVectorizer + cosine similarity.

    Workflow
    --------
    1. fit(df)        → vectorise the 'tags' column, compute similarity matrix
    2. recommend(title, n) → return top-n similar movies
    3. save/load      → persist the fitted model to disk
    """

    def __init__(self, max_features: int = 5000, stop_words: str = "english"):
        self.max_features = max_features
        self.stop_words   = stop_words
        self.vectorizer   = CountVectorizer(
            max_features=max_features,
            stop_words=stop_words
        )
        self._df         = None   # processed dataframe
        self._matrix     = None   # (n_movies × max_features) count matrix
        self._similarity = None   # (n_movies × n_movies) cosine similarity
        self._index_map  = {}     # title → integer index (lowercased keys)

    # ── fitting ──────────────────────────────────────────────────────────────

    def fit(self, df: pd.DataFrame) -> "ContentRecommender":
        """
        Vectorise the 'tags' column with CountVectorizer and compute the
        full cosine-similarity matrix.

        Parameters
        ----------
        df : processed DataFrame that must contain columns
             ['title', 'tags'] and optionally the display helpers.
        """
        if "tags" not in df.columns:
            raise ValueError("DataFrame must contain a 'tags' column.")

        self._df = df.reset_index(drop=True).copy()

        # Build count matrix  (n_movies × vocab)
        self._matrix = self.vectorizer.fit_transform(
            self._df["tags"].fillna(""))

        # Pairwise cosine similarity  (n_movies × n_movies)
        self._similarity = cosine_similarity(self._matrix)

        # Title → row-index look-up  (case-insensitive)
        self._index_map = {
            title.lower(): idx
            for idx, title in enumerate(self._df["title"])
        }

        return self

    # ── recommendation ────────────────────────────────────────────────────────

    def recommend(self, title: str, n: int = 5) -> pd.DataFrame:
        """
        Return the top *n* most similar movies to *title*.

        Returns
        -------
        DataFrame with columns:
          title, similarity_score, genres_display, cast_display,
          vote_average, year, overview  (subset available in self._df)
        """
        key = title.strip().lower()
        if key not in self._index_map:
            raise KeyError(
                f"'{title}' not found in the dataset. "
                "Try searching for the exact title."
            )

        idx    = self._index_map[key]
        scores = list(enumerate(self._similarity[idx]))
        # sort descending, skip the movie itself (score == 1.0)
        scores = sorted(scores, key=lambda x: x[1], reverse=True)
        scores = [(i, s) for i, s in scores if i != idx][:n]

        result_rows = []
        for rank, (movie_idx, score) in enumerate(scores, start=1):
            row = self._df.iloc[movie_idx].to_dict()
            row["similarity_score"] = round(float(score), 4)
            row["rank"]             = rank
            result_rows.append(row)

        return pd.DataFrame(result_rows)

    # ── search ────────────────────────────────────────────────────────────────

    def search(self, query: str, limit: int = 10) -> list[str]:
        """Return movie titles containing *query* (case-insensitive)."""
        q = query.lower()
        return [t for t in self._df["title"] if q in t.lower()][:limit]

    def get_movie(self, title: str) -> pd.Series | None:
        """Return a single movie row by exact title (case-insensitive)."""
        key = title.strip().lower()
        if key not in self._index_map:
            return None
        return self._df.iloc[self._index_map[key]]

    # ── persistence ───────────────────────────────────────────────────────────

    def save(self, path: str = "model/recommender.pkl") -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)
        print(f"Model saved → {path}")

    @classmethod
    def load(cls, path: str = "model/recommender.pkl") -> "ContentRecommender":
        with open(path, "rb") as f:
            return pickle.load(f)

    # ── diagnostics ───────────────────────────────────────────────────────────

    @property
    def vocab_size(self) -> int:
        return len(self.vectorizer.vocabulary_) if self._matrix is not None else 0

    @property
    def n_movies(self) -> int:
        return len(self._df) if self._df is not None else 0

    def __repr__(self) -> str:
        return (f"ContentRecommender("
                f"movies={self.n_movies}, vocab={self.vocab_size})")
