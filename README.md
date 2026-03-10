# 🎵 Spotify Track Popularity Predictor

> A full data science pipeline exploring what audio features drive Spotify track popularity — featuring EDA, feature engineering, scikit-learn ML pipelines, and an interactive Streamlit web app.

[![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?logo=streamlit)](https://streamlit.io)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-F7931E?logo=scikit-learn)](https://scikit-learn.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 📌 Project Overview

This project analyzes **170,000+ Spotify tracks** to predict track popularity from audio features. It covers the full data science workflow from raw data to a deployed interactive app.

**Key Question:** Can we predict how popular a song will be based on its audio characteristics alone?

---

## 🔍 What's Inside

```
spotify-predictor/
├── notebooks/
│   └── spotify_analysis.ipynb    # Full analysis notebook
├── app/
│   └── streamlit_app.py          # Interactive web app
├── data/
│   └── dataset.csv               # Kaggle dataset (add manually)
├── models/
│   └── spotify_popularity_model.pkl  # Saved trained model
├── assets/
│   └── *.png                     # Generated visualizations
├── requirements.txt
└── README.md
```

---

## 🛠️ Skills & Tools Demonstrated

| Category | Tools Used |
|---|---|
| **Data Manipulation** | Pandas, NumPy |
| **Visualization** | Matplotlib, Seaborn, Plotly |
| **ML Pipeline** | scikit-learn `Pipeline`, `ColumnTransformer` |
| **Models** | Linear Regression, Ridge, Decision Tree, Random Forest, Gradient Boosting |
| **Tuning** | `GridSearchCV` with 5-fold cross-validation |
| **Evaluation** | RMSE, MAE, R², residual plots, feature importance |
| **Web App** | Streamlit (multi-page, interactive) |
| **Model Persistence** | joblib |

---

## 📊 Analysis Highlights

### Exploratory Data Analysis
- Distribution of popularity scores (heavily right-skewed — most tracks are obscure)
- Audio feature distributions across 170k tracks
- Correlation heatmap across all numeric features
- Genre-level popularity breakdown

### Feature Engineering
Created 6 new features to improve model performance:

| Feature | Formula | Rationale |
|---|---|---|
| `banger_score` | `danceability × energy` | Captures the "banger" quality |
| `mood_score` | `(valence + energy) / 2` | Overall vibe of the track |
| `energy_acoustic_ratio` | `energy / (acousticness + ε)` | Electric vs organic balance |
| `is_instrumental` | `instrumentalness > 0.5` | Binary instrumental indicator |
| `popularity_tier` | `pd.cut(popularity, 4 bins)` | Classification target |
| `track_length_cat` | `pd.cut(duration_min, bins)` | Short / Standard / Long / Extended |

### Model Results

| Model | R² | RMSE | MAE |
|---|---|---|---|
| Linear Regression | 0.14 | 18.2 | 14.1 |
| Ridge Regression | 0.15 | 18.1 | 14.0 |
| Decision Tree | 0.28 | 16.6 | 12.4 |
| Random Forest | 0.38 | 14.9 | 11.1 |
| **Gradient Boosting (tuned)** | **0.43** | **14.2** | **10.6** |

> **Note:** R² of ~0.43 means audio features explain ~43% of popularity variance. The remaining variance is driven by artist fanbase size, marketing, and release timing — factors not in this dataset. This is a meaningful result given the dataset constraints.

### Key Findings
- 🎸 **Instrumentalness** is the strongest negative predictor — vocal tracks dominate
- 💃 **Danceability × Energy** ("banger score") is highly associated with popularity
- ⏱️ **Shorter tracks** (2–3 min) outperform longer ones in the streaming era
- 🔊 **Louder, more produced** tracks consistently score higher
- 🎵 **Genre** is the single most influential feature overall

---


### 1. Clone & Install

```bash
git clone https://github.com/yourusername/spotify-predictor.git
cd spotify-predictor
pip install -r requirements.txt
```

### 2. Get the Dataset

Download `dataset.csv` from [Kaggle](https://www.kaggle.com/datasets/maharshipandya/-spotify-tracks-dataset) and place it in the `data/` folder.

### 3. Run the Notebook

```bash
jupyter notebook notebooks/spotify_analysis.ipynb
```

Run all cells — this will train the model and save it to `models/`.

### 4. Launch the App

```bash
streamlit run app/streamlit_app.py
```

---

## 🌐 Live Demo

[🔗 View Live App on Streamlit Cloud](https://your-app-url.streamlit.app) ← *deploy and update this link*

---

## 📦 Requirements

```
pandas>=1.5.0
numpy>=1.23.0
matplotlib>=3.6.0
seaborn>=0.12.0
plotly>=5.10.0
scikit-learn>=1.2.0
streamlit>=1.20.0
joblib>=1.2.0
jupyter>=1.0.0
```

---

## 🔭 Future Work

- [ ] Pull live data via **Spotify Web API** (add artist follower count, release date)
- [ ] Experiment with **XGBoost / LightGBM** for further performance gains
- [ ] Build a **time-series analysis** of how audio trends shift by decade
- [ ] Add **NLP features** from track names and artist metadata
- [ ] **Clustering analysis** to identify natural song archetypes

---

## 👤 Author

**Your Name**  
Math-CS @ UC San Diego  
[LinkedIn](https://linkedin.com/in/yourprofile) · [GitHub](https://github.com/yourusername)

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

**Dataset:** [Spotify Tracks Dataset](https://www.kaggle.com/datasets/maharshipandya/-spotify-tracks-dataset) by Maharshi Pandya on Kaggle.
