"""
streamlit_app.py — Cinema-grade Streamlit UI for the movie recommender.
Run with:  streamlit run streamlit_app.py
"""

import sys
import os

# Allow imports from app/ when running from root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "app"))

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

from preprocess import run_pipeline, engineer_features, build_display_df
from recommender import ContentRecommender
from sample_data import generate_sample_dataset


# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="CineMatch · Movie Recommender",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────

st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=Inter:wght@300;400;600&display=swap');

  html, body, [class*="css"] {
      background-color: #0a0a0f;
      color: #e8e8e8;
  }

  h1, h2, h3 { font-family: 'Bebas Neue', sans-serif; letter-spacing: 0.06em; }

  .hero-title {
      font-family: 'Bebas Neue', sans-serif;
      font-size: clamp(2.5rem, 6vw, 5rem);
      background: linear-gradient(135deg, #e50914 0%, #ff6b35 50%, #ffd700 100%);
      -webkit-background-clip: text; -webkit-text-fill-color: transparent;
      line-height: 1.0; margin-bottom: 0;
  }
  .hero-sub {
      font-family: 'Inter', sans-serif;
      font-size: 0.95rem; color: #888; letter-spacing: 0.15em; text-transform: uppercase;
      margin-top: 0.3rem; margin-bottom: 2rem;
  }

  .movie-card {
      background: linear-gradient(145deg, #141420, #1a1a2e);
      border: 1px solid #2a2a3e;
      border-radius: 12px;
      padding: 1.2rem 1.4rem;
      margin-bottom: 1rem;
      transition: border-color 0.2s, transform 0.15s;
      position: relative; overflow: hidden;
  }
  .movie-card::before {
      content: '';
      position: absolute; top: 0; left: 0;
      width: 4px; height: 100%;
      background: linear-gradient(180deg, #e50914, #ff6b35);
  }
  .movie-card:hover { border-color: #e50914; transform: translateX(3px); }

  .rank-badge {
      display: inline-block;
      background: #e50914;
      color: white;
      font-family: 'Bebas Neue', sans-serif;
      font-size: 1.1rem; letter-spacing: 0.05em;
      padding: 0.1rem 0.55rem;
      border-radius: 6px; margin-right: 0.6rem;
  }
  .movie-title-card {
      font-family: 'Bebas Neue', sans-serif;
      font-size: 1.5rem; color: #ffffff; display: inline;
  }
  .sim-score {
      float: right;
      font-family: 'Inter', sans-serif; font-size: 0.8rem;
      color: #ffd700; font-weight: 600;
  }
  .meta-tag {
      display: inline-block;
      background: #1e1e32;
      border: 1px solid #333355;
      border-radius: 20px;
      font-family: 'Inter', sans-serif; font-size: 0.72rem;
      color: #aaa; padding: 0.15rem 0.55rem; margin: 0.2rem 0.2rem 0 0;
  }
  .overview-text {
      font-family: 'Inter', sans-serif; font-size: 0.82rem;
      color: #999; line-height: 1.5; margin-top: 0.6rem;
  }

  .selected-card {
      background: linear-gradient(145deg, #1a0808, #2a0a0a);
      border: 1px solid #e50914;
      border-radius: 12px; padding: 1.4rem;
      margin-bottom: 1.5rem; position: relative; overflow: hidden;
  }
  .selected-card::before {
      content: 'SELECTED';
      position: absolute; top: 0.7rem; right: 0.8rem;
      font-family: 'Bebas Neue', sans-serif;
      font-size: 0.75rem; letter-spacing: 0.12em;
      color: #e50914; opacity: 0.7;
  }

  .stat-box {
      background: #141420; border: 1px solid #2a2a3e;
      border-radius: 8px; padding: 0.8rem 1rem;
      text-align: center;
  }
  .stat-val {
      font-family: 'Bebas Neue', sans-serif;
      font-size: 2rem; color: #e50914; display: block;
  }
  .stat-label {
      font-family: 'Inter', sans-serif;
      font-size: 0.7rem; color: #666; text-transform: uppercase; letter-spacing: 0.1em;
  }

  div[data-testid="stSelectbox"] > div > div {
      background: #141420 !important; border: 1px solid #333 !important;
      color: #e8e8e8 !important;
  }
  .stSlider > div { color: #e8e8e8; }
  footer { display: none; }
  #MainMenu { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ── Model loading ─────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def load_model():
    """Build and cache the recommender from sample data."""
    raw = generate_sample_dataset()
    # Manually run engineer_features + build_display_df
    # (skip CSV loading; raw already has the right shape)
    from preprocess import engineer_features, build_display_df
    df = engineer_features(raw)
    df = build_display_df(df)
    model = ContentRecommender(max_features=5000)
    model.fit(df)
    return model, df


@st.cache_resource(show_spinner=False)
def load_model_from_csv(movies_path, credits_path):
    df = run_pipeline(movies_path, credits_path)
    model = ContentRecommender(max_features=5000)
    model.fit(df)
    return model, df


# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## ⚙️ Data Source")
    data_mode = st.radio(
        "Choose dataset",
        ["Demo (built-in 30 movies)", "Upload TMDB CSVs"],
        index=0,
    )

    model, df = None, None

    if data_mode == "Upload TMDB CSVs":
        st.info("Download from Kaggle: **tmdb-movie-metadata**")
        mf = st.file_uploader("tmdb_5000_movies.csv", type="csv", key="movies")
        cf = st.file_uploader("tmdb_5000_credits.csv", type="csv", key="credits")
        if mf and cf:
            import tempfile, shutil
            with tempfile.TemporaryDirectory() as tmp:
                mp = os.path.join(tmp, "movies.csv")
                cp = os.path.join(tmp, "credits.csv")
                with open(mp, "wb") as f: f.write(mf.read())
                with open(cp, "wb") as f: f.write(cf.read())
                with st.spinner("Processing TMDB dataset…"):
                    model, df = load_model_from_csv(mp, cp)
            st.success(f"✅ Loaded {len(df):,} movies")
        else:
            st.warning("Upload both files to enable full dataset.")
    else:
        with st.spinner("Initialising demo model…"):
            model, df = load_model()
        st.success(f"✅ Demo: {len(df)} movies loaded")

    st.markdown("---")
    st.markdown("### 🔧 Settings")
    n_recs = st.slider("Recommendations to show", 3, 10, 5)
    show_chart = st.checkbox("Show similarity chart", value=True)
    show_genres = st.checkbox("Show genre breakdown", value=True)

    st.markdown("---")
    st.markdown("""
    <div style='font-family:Inter,sans-serif;font-size:0.72rem;color:#555;line-height:1.6'>
    <b style='color:#888'>Tech Stack</b><br>
    Python · Pandas · NumPy<br>
    Scikit-Learn · Streamlit · Plotly<br><br>
    <b style='color:#888'>Algorithm</b><br>
    CountVectorizer + Cosine Similarity<br>
    Content-based filtering on<br>
    genres, keywords, cast, crew & overview
    </div>
    """, unsafe_allow_html=True)


# ── Main area ─────────────────────────────────────────────────────────────────

st.markdown('<h1 class="hero-title">CineMatch</h1>', unsafe_allow_html=True)
st.markdown('<p class="hero-sub">AI-Powered Content-Based Movie Recommender</p>', unsafe_allow_html=True)

if model is None:
    st.info("⬅️  Upload the TMDB CSV files in the sidebar to get started.")
    st.stop()

# ── Stats row ─────────────────────────────────────────────────────────────────

c1, c2, c3, c4 = st.columns(4)
with c1:
    st.markdown(f'<div class="stat-box"><span class="stat-val">{model.n_movies}</span><span class="stat-label">Movies</span></div>', unsafe_allow_html=True)
with c2:
    st.markdown(f'<div class="stat-box"><span class="stat-val">{model.vocab_size:,}</span><span class="stat-label">Vocab size</span></div>', unsafe_allow_html=True)
with c3:
    avg_rating = df["vote_average"].mean()
    st.markdown(f'<div class="stat-box"><span class="stat-val">{avg_rating:.1f}</span><span class="stat-label">Avg rating</span></div>', unsafe_allow_html=True)
with c4:
    genres_set = set(g for row in df["genres_list"] for g in row)
    st.markdown(f'<div class="stat-box"><span class="stat-val">{len(genres_set)}</span><span class="stat-label">Genres</span></div>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── Movie search & selection ───────────────────────────────────────────────────

st.markdown("### 🎬 Find a Movie")
col_search, col_btn = st.columns([4, 1])

with col_search:
    all_titles = sorted(df["title"].tolist())
    selected_title = st.selectbox(
        "Select or type a movie title",
        options=all_titles,
        index=all_titles.index("Inception") if "Inception" in all_titles else 0,
        label_visibility="collapsed",
    )

with col_btn:
    go_btn = st.button("🔍 Recommend", use_container_width=True, type="primary")

# ── Selected movie card ────────────────────────────────────────────────────────

movie = model.get_movie(selected_title)
if movie is not None:
    year = movie.get("year", "")
    rating = movie.get("vote_average", 0)
    runtime = movie.get("runtime", "?")
    genres_d = movie.get("genres_display", "")
    cast_d = movie.get("cast_display", "")
    director = movie.get("director", "")
    overview = movie.get("overview", "")

    st.markdown(f"""
    <div class="selected-card">
      <div style="margin-bottom:0.5rem">
        <span style="font-family:'Bebas Neue',sans-serif;font-size:2rem;color:#fff">{movie['title']}</span>
        <span style="font-family:'Inter',sans-serif;font-size:0.9rem;color:#888;margin-left:0.8rem">({year})</span>
      </div>
      <span class="meta-tag">⭐ {rating}</span>
      <span class="meta-tag">⏱ {runtime} min</span>
      {''.join(f'<span class="meta-tag">{g}</span>' for g in genres_d.split(', ') if g)}
      {'<span class="meta-tag">🎬 ' + director + '</span>' if director else ''}
      {'<span class="meta-tag">🎭 ' + cast_d + '</span>' if cast_d else ''}
      <p class="overview-text">{overview}</p>
    </div>
    """, unsafe_allow_html=True)

# ── Recommendations ────────────────────────────────────────────────────────────

if go_btn or True:   # always show on first load
    try:
        recs = model.recommend(selected_title, n=n_recs)
    except KeyError as e:
        st.error(str(e))
        st.stop()

    st.markdown(f"### 🍿 Top {n_recs} Recommendations")

    # Similarity bar chart
    if show_chart and not recs.empty:
        fig = go.Figure(go.Bar(
            x=recs["similarity_score"],
            y=recs["title"],
            orientation="h",
            marker=dict(
                color=recs["similarity_score"],
                colorscale=[[0, "#2a0a0a"], [0.5, "#e50914"], [1, "#ffd700"]],
                showscale=False,
            ),
            text=[f"{s:.2%}" for s in recs["similarity_score"]],
            textposition="outside",
            textfont=dict(color="#ccc", size=11),
        ))
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(tickfont=dict(color="#ddd", family="Inter", size=12)),
            margin=dict(l=10, r=60, t=10, b=10),
            height=max(220, n_recs * 44),
            font=dict(color="#ddd"),
        )
        st.plotly_chart(fig, use_container_width=True)

    # Recommendation cards
    for _, row in recs.iterrows():
        rank = int(row["rank"])
        title = row["title"]
        score = row["similarity_score"]
        year = int(row.get("year", 0)) if row.get("year", 0) else ""
        rating = row.get("vote_average", 0)
        runtime = row.get("runtime", "?")
        genres_d = row.get("genres_display", "")
        cast_d = row.get("cast_display", "")
        director = row.get("director", "")
        overview = str(row.get("overview", ""))[:200] + "…"

        genre_tags = "".join(
            f'<span class="meta-tag">{g.strip()}</span>'
            for g in genres_d.split(",") if g.strip()
        )
        st.markdown(f"""
        <div class="movie-card">
          <div>
            <span class="rank-badge">#{rank}</span>
            <span class="movie-title-card">{title}</span>
            <span class="sim-score">{score:.2%} match</span>
          </div>
          <div style="margin-top:0.5rem">
            <span class="meta-tag">⭐ {rating}</span>
            <span class="meta-tag">⏱ {runtime} min</span>
            {'<span class="meta-tag">📅 ' + str(year) + '</span>' if year else ''}
            {genre_tags}
            {'<span class="meta-tag">🎬 ' + director + '</span>' if director else ''}
          </div>
          {'<p class="overview-text">' + overview + '</p>' if overview.strip() else ''}
        </div>
        """, unsafe_allow_html=True)

    # Genre breakdown
    if show_genres and not recs.empty:
        st.markdown("### 📊 Genre Distribution in Recommendations")
        genre_counts: dict = {}
        for _, row in recs.iterrows():
            for g in row.get("genres_list", []):
                genre_counts[g] = genre_counts.get(g, 0) + 1

        if genre_counts:
            gdf = pd.DataFrame(
                list(genre_counts.items()), columns=["Genre", "Count"]
            ).sort_values("Count", ascending=False)

            fig2 = px.bar(
                gdf, x="Genre", y="Count",
                color="Count",
                color_continuous_scale=["#2a0a0a", "#e50914", "#ffd700"],
            )
            fig2.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                xaxis=dict(tickfont=dict(color="#ccc"), gridcolor="#1a1a2e"),
                yaxis=dict(tickfont=dict(color="#ccc"), gridcolor="#1a1a2e"),
                coloraxis_showscale=False,
                margin=dict(l=10, r=10, t=10, b=10),
                height=260,
                font=dict(color="#ddd"),
            )
            st.plotly_chart(fig2, use_container_width=True)

# ── Footer ─────────────────────────────────────────────────────────────────────

st.markdown("---")
st.markdown("""
<div style='text-align:center;font-family:Inter,sans-serif;font-size:0.75rem;color:#444;padding:1rem 0'>
  CineMatch · Content-Based Movie Recommender · Built with Python, Scikit-Learn & Streamlit
</div>
""", unsafe_allow_html=True)
