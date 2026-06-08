# 🎬 CineMatch — Content-Based Movie Recommender

A production-grade movie recommendation system built with Python, Scikit-Learn, and Streamlit.  
Uses the **TMDB 5000 Movie Dataset** and content-based filtering to surface the most similar films for any title.

---

## 🗂️ Project Structure

```
movie-recommender/
├── app/
│   ├── preprocess.py      # Data loading, cleaning, feature engineering
│   ├── recommender.py     # CountVectorizer + cosine-similarity engine
│   └── sample_data.py     # 30-movie demo dataset (no CSV needed)
├── data/                  # Place TMDB CSV files here (optional)
│   ├── tmdb_5000_movies.csv
│   └── tmdb_5000_credits.csv
├── model/                 # Auto-created; stores trained .pkl
├── streamlit_app.py       # Streamlit web application
├── train.py               # CLI training script
├── requirements.txt
└── README.md
```

---

## ⚙️ Setup & Installation

```bash
# 1. Clone / copy the project
cd movie-recommender

# 2. Create a virtual environment
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

---

## 🚀 Running the App

### Option A — Demo mode (no data download needed)

```bash
streamlit run streamlit_app.py
```

The app starts in **demo mode** with 30 hand-picked TMDB movies.

### Option B — Full TMDB Dataset

1. Download from Kaggle: [tmdb-movie-metadata](https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata)
2. Place `tmdb_5000_movies.csv` and `tmdb_5000_credits.csv` in `data/`
3. Run the app and use the sidebar **"Upload TMDB CSVs"** option, **or** pre-train offline:

```bash
python train.py --movies data/tmdb_5000_movies.csv \
                --credits data/tmdb_5000_credits.csv
```

---

## 🧠 How It Works

### 1. Data Preprocessing (`preprocess.py`)
- Merges `movies` + `credits` CSVs on `title`
- Drops duplicates and rows missing `overview` / `genres`
- Parses JSON-string columns (genres, keywords, cast, crew)

### 2. Feature Engineering
- Extracts **top 3 cast members** and **director** from crew list
- Concatenates: `overview tokens + genres + keywords + cast + director` → **`tags`** column
- Collapses multi-word names to single tokens (`Sam Worthington` → `samworthington`) to prevent partial matches

### 3. Vectorisation (`CountVectorizer`)
- Fits `CountVectorizer(max_features=5000, stop_words='english')` on the `tags` column
- Produces a sparse **(n_movies × 5000)** count matrix

### 4. Cosine Similarity
- `sklearn.metrics.pairwise.cosine_similarity` computes pairwise similarity across all movies
- Result: **(n_movies × n_movies)** matrix stored in memory

### 5. Recommendation
- Look up a movie's row index
- Sort its similarity vector descending
- Return the top-*n* movies (excluding itself)

---

## 🌐 Deployment

### Streamlit Community Cloud (free)

1. Push the project to a **public GitHub repo**
2. Go to [share.streamlit.io](https://share.streamlit.io) → **New app**
3. Point to `streamlit_app.py`
4. Click **Deploy**

### Docker

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 8501
CMD ["streamlit", "run", "streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

```bash
docker build -t cinematch .
docker run -p 8501:8501 cinematch
```

### Hugging Face Spaces

1. Create a new Space with **Streamlit** SDK
2. Upload all project files
3. The Space auto-runs `streamlit_app.py`

---

## 📄 Resume Bullet Points

See `RESUME_BULLETS.md` for four detailed, metrics-driven project bullets ready to paste into a CV or LinkedIn profile.

---

## 🛠️ Tech Stack

| Layer | Library |
|---|---|
| Data wrangling | Pandas, NumPy |
| NLP vectorisation | Scikit-Learn `CountVectorizer` |
| Similarity | Scikit-Learn `cosine_similarity` |
| Web UI | Streamlit |
| Charts | Plotly |
| Persistence | Python `pickle` |

---

## 📌 Kaggle Dataset Citation

> TMDB 5000 Movie Dataset — The Movie Database (TMDb) via Kaggle  
> https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata
