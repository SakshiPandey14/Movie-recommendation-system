"""
preprocess.py — Data loading, cleaning, and feature engineering
for the TMDB content-based movie recommender.
"""

import ast
import numpy as np
import pandas as pd


# ── helpers ──────────────────────────────────────────────────────────────────

def _safe_parse(val):
    """Parse a JSON-ish string column; return [] on failure."""
    try:
        return ast.literal_eval(val)
    except (ValueError, SyntaxError):
        return []


def _extract_names(obj, n=None):
    """Return up to *n* 'name' values from a list of dicts."""
    items = obj if isinstance(obj, list) else _safe_parse(obj)
    names = [d["name"] for d in items if isinstance(d, dict) and "name" in d]
    return names[:n] if n else names


def _extract_director(crew):
    """Pull the director's name from the crew list."""
    items = crew if isinstance(crew, list) else _safe_parse(crew)
    for member in items:
        if isinstance(member, dict) and member.get("job") == "Director":
            return member.get("name", "")
    return ""


def _sanitize(text: str) -> str:
    """Lowercase and remove spaces so 'Sam Worthington' → 'samworthington'."""
    return text.lower().replace(" ", "")


# ── main pipeline ─────────────────────────────────────────────────────────────

def load_and_merge(movies_path: str, credits_path: str) -> pd.DataFrame:
    """Load the two TMDB CSV files and merge on title."""
    movies = pd.read_csv(movies_path)
    credits = pd.read_csv(credits_path)
    # tmdb_5000_credits has columns: movie_id, title, cast, crew
    df = movies.merge(credits, on="title")
    return df


def select_features(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only the columns needed for content-based filtering."""
    cols = ["movie_id", "title", "overview", "genres",
            "keywords", "cast", "crew",
            "vote_average", "vote_count", "release_date",
            "runtime", "popularity"]
    existing = [c for c in cols if c in df.columns]
    return df[existing].copy()


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Drop duplicates and rows with critical nulls."""
    df = df.drop_duplicates(subset="title")
    df = df.dropna(subset=["overview", "genres"])
    df = df.reset_index(drop=True)
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build a single 'tags' column from overview + genres + keywords +
    top-3 cast + director.  Each token is collapsed to one word to
    prevent partial-match pollution (e.g. "Sam Worthington" →
    "samworthington").
    """
    df = df.copy()

    # Parse list-of-dict columns
    df["genres_list"]   = df["genres"].apply(lambda x: _extract_names(x))
    df["keywords_list"] = df["keywords"].apply(lambda x: _extract_names(x))
    df["cast_list"]     = df["cast"].apply(lambda x: _extract_names(x, n=3))
    df["director"]      = df["crew"].apply(_extract_director)

    # Sanitise every token
    df["genres_clean"]   = df["genres_list"].apply(
        lambda lst: [_sanitize(g) for g in lst])
    df["keywords_clean"] = df["keywords_list"].apply(
        lambda lst: [_sanitize(k) for k in lst])
    df["cast_clean"]     = df["cast_list"].apply(
        lambda lst: [_sanitize(a) for a in lst])
    df["director_clean"] = df["director"].apply(_sanitize)

    # Overview → list of words (lowercased)
    df["overview_tokens"] = df["overview"].fillna("").apply(
        lambda txt: txt.lower().split())

    # Concatenate everything into one 'tags' string
    def build_tags(row):
        parts = (row["overview_tokens"]
                 + row["genres_clean"]
                 + row["keywords_clean"]
                 + row["cast_clean"]
                 + ([row["director_clean"]] if row["director_clean"] else []))
        return " ".join(parts)

    df["tags"] = df.apply(build_tags, axis=1)

    # Derived numeric helpers
    if "release_date" in df.columns:
        df["year"] = pd.to_datetime(
            df["release_date"], errors="coerce").dt.year.fillna(0).astype(int)

    return df


def build_display_df(df: pd.DataFrame) -> pd.DataFrame:
    """Human-readable version of list columns for the UI."""
    out = df.copy()
    out["genres_display"]   = out["genres_list"].apply(
        lambda lst: ", ".join(lst))
    out["cast_display"]     = out["cast_list"].apply(
        lambda lst: ", ".join(lst))
    out["keywords_display"] = out["keywords_list"].apply(
        lambda lst: ", ".join(lst[:6]))
    return out


# ── convenience entry-point ───────────────────────────────────────────────────

def run_pipeline(movies_path: str, credits_path: str) -> pd.DataFrame:
    """End-to-end: load → select → clean → engineer → display columns."""
    df = load_and_merge(movies_path, credits_path)
    df = select_features(df)
    df = clean(df)
    df = engineer_features(df)
    df = build_display_df(df)
    return df
