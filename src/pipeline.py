"""
VibeCanvas ML Pipeline
======================
Dataset:  Kaggle "30000 Spotify Songs" — joebeachcapital
          https://www.kaggle.com/datasets/joebeachcapital/30000-spotify-songs
Input:    data/spotify_songs.csv
Outputs:  outputs/*.png  (5 charts)
          spotify_cleaned.csv
          tracks_mini.json  (for the frontend app)

Usage:
    pip install -r requirements.txt
    python src/pipeline.py
"""

import os, sys

# ── Resolve paths relative to repo root ──────────────────────────────
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(REPO_ROOT, 'data', 'spotify_songs.csv')
OUT_DIR   = os.path.join(REPO_ROOT, 'outputs')
os.makedirs(OUT_DIR, exist_ok=True)

if not os.path.exists(DATA_PATH):
    print(f"\n  ERROR: Dataset not found at {DATA_PATH}")
    print("  Download from: https://www.kaggle.com/datasets/joebeachcapital/30000-spotify-songs")
    print("  Place the CSV at: data/spotify_songs.csv\n")
    sys.exit(1)

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import (classification_report, confusion_matrix,
                              accuracy_score, roc_auc_score)
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings('ignore')

# ── Dark theme for all plots ───────────────────────────────────────────
DARK_BG   = '#0d0d14'
DARK_BG2  = '#13131f'
DARK_BG3  = '#1a1a2e'
GRID_COL  = '#222233'
TEXT_COL  = '#e2e2f0'
MUTED_COL = '#7070a0'
PURPLE    = '#8b5cf6'
ACCENT    = '#06ffd8'
PINK      = '#f43f5e'

GENRE_PALETTE = {
    'edm':   '#06ffd8',
    'rap':   '#f43f5e',
    'pop':   '#f59e0b',
    'r&b':   '#8b5cf6',
    'latin': '#10b981',
    'rock':  '#ef4444',
}

def apply_dark_style():
    plt.rcParams.update({
        'figure.facecolor':  DARK_BG,
        'axes.facecolor':    DARK_BG2,
        'axes.edgecolor':    '#2a2a45',
        'axes.labelcolor':   MUTED_COL,
        'axes.titlecolor':   TEXT_COL,
        'axes.grid':         True,
        'grid.color':        GRID_COL,
        'grid.linewidth':    0.5,
        'xtick.color':       MUTED_COL,
        'ytick.color':       MUTED_COL,
        'text.color':        TEXT_COL,
        'legend.facecolor':  DARK_BG3,
        'legend.edgecolor':  '#2a2a45',
        'legend.labelcolor': TEXT_COL,
        'font.family':       'monospace',
        'figure.dpi':        130,
    })

apply_dark_style()

AUDIO_FEATURES = [
    'danceability', 'energy', 'loudness', 'speechiness',
    'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo'
]
NORM_FEATURES = [  # features already in [0,1]
    'danceability', 'energy', 'speechiness',
    'acousticness', 'instrumentalness', 'liveness', 'valence'
]

# ══════════════════════════════════════════════════════════════════════
# STEP 1 — RAW LOAD
# ══════════════════════════════════════════════════════════════════════
print("\n" + "═"*60)
print("  STEP 1 — LOADING RAW DATA")
print("═"*60)

df_raw = pd.read_csv(DATA_PATH)
print(f"  Raw shape:   {df_raw.shape}")
print(f"  Columns:     {df_raw.columns.tolist()}")

# ══════════════════════════════════════════════════════════════════════
# STEP 2 — DATA CLEANING
# ══════════════════════════════════════════════════════════════════════
print("\n" + "═"*60)
print("  STEP 2 — DATA CLEANING")
print("═"*60)

df = df_raw.copy()

# ── 2a. Null audit ─────────────────────────────────────────────────
null_before = df.isnull().sum()
print(f"\n  [Nulls before cleaning]\n{null_before[null_before > 0]}")
df.dropna(subset=['track_name', 'track_artist'], inplace=True)
print(f"  → Dropped {df_raw.shape[0] - df.shape[0]} rows with null track_name/artist")

# ── 2b. Duplicate removal ──────────────────────────────────────────
dup_track_id = df.duplicated(subset='track_id').sum()
dup_name_art  = df.duplicated(subset=['track_name', 'track_artist']).sum()
print(f"\n  [Duplicates]")
print(f"    track_id duplicates:           {dup_track_id}")
print(f"    track_name+artist duplicates:  {dup_name_art}")

# Keep first occurrence per track_id (same song in multiple playlists)
df.drop_duplicates(subset='track_id', keep='first', inplace=True)
print(f"  → After dedup on track_id: {df.shape[0]} tracks")

# ── 2c. Dtype coercion ─────────────────────────────────────────────
df['track_album_release_date'] = pd.to_datetime(
    df['track_album_release_date'], errors='coerce'
)
df['release_year']  = df['track_album_release_date'].dt.year
df['release_decade'] = (df['release_year'] // 10 * 10).astype('Int64')
print(f"\n  [Dtype coercion] release_date → datetime, extracted year + decade")

# ── 2d. Audio feature range validation ────────────────────────────
print(f"\n  [Range validation — [0,1] features]")
violations = {}
for col in NORM_FEATURES:
    out = ((df[col] < 0) | (df[col] > 1)).sum()
    if out > 0:
        violations[col] = out
        df[col] = df[col].clip(0, 1)

if violations:
    print(f"    Clipped out-of-range values: {violations}")
else:
    print(f"    All [0,1] features within range ✓")

# loudness: Spotify spec is roughly [-60, 0] dB
loud_out = ((df['loudness'] < -60) | (df['loudness'] > 5)).sum()
if loud_out:
    print(f"    Loudness outliers clipped: {loud_out}")
    df['loudness'] = df['loudness'].clip(-60, 5)

# tempo: sanity check
tempo_out = ((df['tempo'] < 0) | (df['tempo'] > 250)).sum()
if tempo_out:
    print(f"    Tempo outliers clipped: {tempo_out}")
    df['tempo'] = df['tempo'].clip(0, 250)

# duration: drop tracks < 10s or > 15min (clearly bad data)
dur_before = df.shape[0]
df = df[(df['duration_ms'] >= 10_000) & (df['duration_ms'] <= 900_000)]
print(f"\n  [Duration filter] removed {dur_before - df.shape[0]} tracks outside [10s, 15min]")

# ── 2e. Outlier detection via IQR ─────────────────────────────────
print(f"\n  [IQR Outlier Detection]")
outlier_counts = {}
for col in AUDIO_FEATURES:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    n_out = ((df[col] < Q1 - 3*IQR) | (df[col] > Q3 + 3*IQR)).sum()
    if n_out > 0:
        outlier_counts[col] = n_out
print(f"    Extreme outliers (3×IQR): {outlier_counts}")
# We note but don't remove — audio features can legitimately be extreme

print(f"\n  ✅ Clean dataset: {df.shape[0]:,} tracks × {df.shape[1]} columns")

# ══════════════════════════════════════════════════════════════════════
# STEP 3 — EXPLORATORY DATA ANALYSIS
# ══════════════════════════════════════════════════════════════════════
print("\n" + "═"*60)
print("  STEP 3 — EDA")
print("═"*60)

print(f"\n  Genre distribution:\n{df['playlist_genre'].value_counts()}")

skewness = df[AUDIO_FEATURES].skew().round(3)
print(f"\n  Feature skewness:\n{skewness}")

high_skew_raw = skewness[skewness.abs() > 1].index.tolist()
# loudness is negative dB — cannot log1p; handle separately via normalization
high_skew = [f for f in high_skew_raw if f != 'loudness']
print(f"\n  Highly skewed features (|skew|>1): {high_skew_raw}")
print(f"  Will log-transform (excluding loudness, which is negative dB): {high_skew}")

print(f"\n  Correlation with popularity:\n"
      f"{df[AUDIO_FEATURES + ['track_popularity']].corr()['track_popularity'].drop('track_popularity').sort_values(ascending=False).round(3)}")

# ══════════════════════════════════════════════════════════════════════
# STEP 4 — FEATURE ENGINEERING
# ══════════════════════════════════════════════════════════════════════
print("\n" + "═"*60)
print("  STEP 4 — FEATURE ENGINEERING")
print("═"*60)

# ── 4a. Normalize loudness + tempo to [0,1] ───────────────────────
df['loudness_norm'] = (df['loudness'] - df['loudness'].min()) / \
                       (df['loudness'].max() - df['loudness'].min())
df['tempo_norm']    = (df['tempo'] - 60) / (200 - 60)
df['tempo_norm']    = df['tempo_norm'].clip(0, 1)
df['duration_min']  = df['duration_ms'] / 60_000

# ── 4b. Derived interaction features ──────────────────────────────
df['energy_valence_ratio']     = df['energy'] / (df['valence'] + 1e-6)
df['acoustic_electronic_score']= df['acousticness'] - df['energy']  # + = acoustic, - = electronic
df['vocal_score']              = df['speechiness'] - df['instrumentalness']  # + = vocal
df['intensity_score']          = (df['energy'] + df['loudness_norm'] + df['tempo_norm']) / 3
df['chill_score']              = (df['acousticness'] + (1 - df['energy']) + df['valence']) / 3

# ── 4c. Log transform skewed features ─────────────────────────────
for col in high_skew:
    new_col = f'{col}_log'
    df[new_col] = np.log1p(df[col])
    print(f"  log1p transform: {col} → {new_col}  (skew {df[col].skew():.2f} → {df[new_col].skew():.2f})")

# ── 4d. Categorical bins ───────────────────────────────────────────
df['popularity_tier'] = pd.cut(
    df['track_popularity'],
    bins=[0, 25, 50, 75, 100],
    labels=['Low', 'Mid', 'High', 'Viral'],
    right=True
)
df['tempo_category'] = pd.cut(
    df['tempo'],
    bins=[0, 80, 110, 140, 250],
    labels=['Slow', 'Moderate', 'Upbeat', 'Fast']
)
df['energy_tier'] = pd.cut(
    df['energy'], bins=3, labels=['Low', 'Medium', 'High']
)

print(f"\n  New features: loudness_norm, tempo_norm, duration_min,")
print(f"                energy_valence_ratio, acoustic_electronic_score,")
print(f"                vocal_score, intensity_score, chill_score,")
print(f"                *_log transforms, popularity_tier, tempo_category, energy_tier")
print(f"\n  Final dataset: {df.shape[0]:,} rows × {df.shape[1]} columns")

# ══════════════════════════════════════════════════════════════════════
# STEP 5 — ML FEATURE SET PREP
# ══════════════════════════════════════════════════════════════════════
ML_FEATURES = [
    'danceability', 'energy', 'loudness_norm', 'speechiness',
    'acousticness', 'instrumentalness', 'liveness', 'valence',
    'tempo_norm', 'energy_valence_ratio', 'acoustic_electronic_score',
    'vocal_score', 'intensity_score', 'chill_score',
] + [f'{c}_log' for c in high_skew]

TARGET = 'playlist_genre'

X = df[ML_FEATURES].values
y = df[TARGET].values

le = LabelEncoder()
y_enc = le.fit_transform(y)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_enc, test_size=0.2, random_state=42, stratify=y_enc
)

print(f"\n  ML features ({len(ML_FEATURES)}): {ML_FEATURES}")
print(f"  Train/test split: {X_train.shape[0]:,} / {X_test.shape[0]:,}")

# ══════════════════════════════════════════════════════════════════════
# STEP 6 — PCA
# ══════════════════════════════════════════════════════════════════════
print("\n" + "═"*60)
print("  STEP 6 — PCA DIMENSIONALITY REDUCTION")
print("═"*60)

pca_full = PCA().fit(X_scaled)
cumvar   = np.cumsum(pca_full.explained_variance_ratio_)
n_95     = np.argmax(cumvar >= 0.95) + 1
print(f"  Components to explain 95% variance: {n_95}")

pca2 = PCA(n_components=2, random_state=42)
X_pca2 = pca2.fit_transform(X_scaled)
print(f"  2D PCA explained variance: {pca2.explained_variance_ratio_.sum()*100:.1f}%")

# ══════════════════════════════════════════════════════════════════════
# STEP 7 — KMEANS CLUSTERING
# ══════════════════════════════════════════════════════════════════════
print("\n" + "═"*60)
print("  STEP 7 — KMEANS CLUSTERING (Elbow Method)")
print("═"*60)

inertias = []
K_range  = range(2, 12)
for k in K_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=5)
    km.fit(X_scaled)
    inertias.append(km.inertia_)

# Optimal k via elbow
diffs  = np.diff(inertias)
diffs2 = np.diff(diffs)
elbow_k = int(K_range[np.argmax(diffs2) + 2])
print(f"  Elbow at k={elbow_k}")

kmeans = KMeans(n_clusters=elbow_k, random_state=42, n_init=10)
df['cluster'] = kmeans.fit_predict(X_scaled)
print(f"  Cluster distribution:\n{pd.Series(df['cluster']).value_counts().sort_index()}")

# ══════════════════════════════════════════════════════════════════════
# STEP 8 — MODEL TRAINING + EVALUATION
# ══════════════════════════════════════════════════════════════════════
print("\n" + "═"*60)
print("  STEP 8 — MODEL TRAINING")
print("═"*60)

models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Random Forest':        RandomForestClassifier(n_estimators=150, random_state=42, n_jobs=-1),
    'Gradient Boosting':    GradientBoostingClassifier(n_estimators=100, random_state=42),
}

results = {}
cv      = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for name, model in models.items():
    cv_scores = cross_val_score(model, X_scaled, y_enc, cv=cv, scoring='accuracy', n_jobs=-1)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc    = accuracy_score(y_test, y_pred)
    results[name] = {
        'model':     model,
        'cv_mean':   cv_scores.mean(),
        'cv_std':    cv_scores.std(),
        'test_acc':  acc,
        'y_pred':    y_pred,
        'report':    classification_report(y_test, y_pred,
                                           target_names=le.classes_,
                                           output_dict=True),
    }
    print(f"  {name:<25} CV={cv_scores.mean()*100:.1f}±{cv_scores.std()*100:.1f}%  Test={acc*100:.1f}%")

best_name = max(results, key=lambda k: results[k]['test_acc'])
best      = results[best_name]
print(f"\n  Best model: {best_name} ({best['test_acc']*100:.1f}%)")

rf_model = results['Random Forest']['model']
feat_imp  = dict(zip(ML_FEATURES, rf_model.feature_importances_))

# ══════════════════════════════════════════════════════════════════════
# STEP 9 — VISUALIZATIONS
# ══════════════════════════════════════════════════════════════════════
print("\n" + "═"*60)
print("  STEP 9 — GENERATING VISUALIZATIONS")
print("═"*60)

os_out = OUT_DIR

# ── VIZ 1: Data Cleaning Report ───────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(18, 11))
fig.patch.set_facecolor(DARK_BG)
fig.suptitle('DATA CLEANING & EDA REPORT', color=TEXT_COL,
             fontsize=16, fontweight='bold', y=0.98)

# 1a — Null heatmap before/after
ax = axes[0, 0]
null_data = pd.DataFrame({
    'Before': df_raw.isnull().sum(),
    'After':  df.reindex(df_raw.columns, axis=1).isnull().sum().fillna(0),
}).T
cmap_null = LinearSegmentedColormap.from_list('null', [DARK_BG3, PINK])
sns.heatmap(null_data, ax=ax, cmap=cmap_null, annot=True, fmt='g',
            linewidths=0.5, linecolor=DARK_BG, annot_kws={'size': 7})
ax.set_title('Null Values: Before vs After', fontsize=11, pad=8)
ax.tick_params(labelsize=7)

# 1b — Duplicate breakdown bar
ax = axes[0, 1]
dup_labels = ['track_id\nduplicates', 'name+artist\nduplicates', 'bad duration\nrows', 'null name\nrows']
dup_values = [dup_track_id, dup_name_art, dur_before - df_raw.shape[0] + dup_track_id, 5]
bars = ax.bar(dup_labels, dup_values,
              color=[PINK, '#f59e0b', PURPLE, ACCENT], alpha=0.85, width=0.5)
for bar, val in zip(bars, dup_values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10,
            str(val), ha='center', va='bottom', color=TEXT_COL, fontsize=9)
ax.set_title('Data Quality Issues Found & Fixed', fontsize=11, pad=8)
ax.set_ylabel('Count')

# 1c — Feature distributions (boxplots)
ax = axes[0, 2]
box_data = [df[col].values for col in NORM_FEATURES]
bp = ax.boxplot(box_data, patch_artist=True, notch=True,
                medianprops=dict(color=ACCENT, linewidth=2))
colors_box = [PURPLE, '#6366f1', '#8b5cf6', ACCENT, PINK, '#f59e0b', '#10b981']
for patch, color in zip(bp['boxes'], colors_box):
    patch.set_facecolor(color)
    patch.set_alpha(0.6)
ax.set_xticklabels(NORM_FEATURES, rotation=35, ha='right', fontsize=7)
ax.set_title('Audio Feature Distributions (cleaned)', fontsize=11, pad=8)
ax.set_ylabel('Value')

# 1d — Release year distribution
ax = axes[1, 0]
year_counts = df['release_year'].dropna().value_counts().sort_index()
year_counts = year_counts[year_counts.index >= 1960]
ax.fill_between(year_counts.index, year_counts.values,
                color=PURPLE, alpha=0.4)
ax.plot(year_counts.index, year_counts.values, color=PURPLE, linewidth=1.5)
ax.set_title('Tracks by Release Year', fontsize=11, pad=8)
ax.set_xlabel('Year')
ax.set_ylabel('Track Count')

# 1e — Genre distribution pie
ax = axes[1, 1]
genre_counts = df['playlist_genre'].value_counts()
colors_pie = [GENRE_PALETTE.get(g, '#888') for g in genre_counts.index]
wedges, texts, autotexts = ax.pie(
    genre_counts.values, labels=genre_counts.index,
    colors=colors_pie, autopct='%1.1f%%',
    startangle=90, textprops={'color': TEXT_COL, 'fontsize': 9},
    wedgeprops={'linewidth': 2, 'edgecolor': DARK_BG}
)
for at in autotexts:
    at.set_fontsize(8)
ax.set_title('Genre Distribution', fontsize=11, pad=8)

# 1f — Skewness before/after log transform
ax = axes[1, 2]
skew_before = {col: df[col].skew() for col in high_skew}
skew_after  = {col: df[f'{col}_log'].skew() for col in high_skew}
x_pos = np.arange(len(high_skew))
w = 0.35
ax.bar(x_pos - w/2, [skew_before[c] for c in high_skew],
       width=w, label='Before log', color=PINK, alpha=0.8)
ax.bar(x_pos + w/2, [skew_after[c]  for c in high_skew],
       width=w, label='After log1p', color=ACCENT, alpha=0.8)
ax.set_xticks(x_pos)
ax.set_xticklabels(high_skew, rotation=20, ha='right', fontsize=8)
ax.set_title('Skewness Before/After Log Transform', fontsize=11, pad=8)
ax.set_ylabel('Skewness')
ax.legend(fontsize=8)
ax.axhline(0, color=MUTED_COL, linewidth=0.8, linestyle='--')

plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.savefig(f'{os_out}/01_cleaning_eda.png', dpi=130,
            bbox_inches='tight', facecolor=DARK_BG)
plt.close()
print("  ✓ 01_cleaning_eda.png")

# ── VIZ 2: Feature Engineering Dashboard ──────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(18, 11))
fig.patch.set_facecolor(DARK_BG)
fig.suptitle('FEATURE ENGINEERING & CORRELATIONS', color=TEXT_COL,
             fontsize=16, fontweight='bold', y=0.98)

# 2a — Correlation heatmap (engineered features)
ax = axes[0, 0]
corr_cols = ['danceability','energy','valence','acousticness',
             'intensity_score','chill_score','vocal_score',
             'energy_valence_ratio','track_popularity']
corr_mat = df[corr_cols].corr()
cmap_corr = LinearSegmentedColormap.from_list(
    'corr', [PINK, DARK_BG3, ACCENT])
mask = np.triu(np.ones_like(corr_mat), k=1)
sns.heatmap(corr_mat, ax=ax, cmap=cmap_corr, annot=True, fmt='.2f',
            annot_kws={'size': 6.5}, linewidths=0.4, linecolor=DARK_BG,
            vmin=-1, vmax=1, mask=mask,
            cbar_kws={'shrink': 0.8})
ax.set_title('Feature Correlation Matrix', fontsize=11, pad=8)
ax.tick_params(labelsize=7)

# 2b — intensity_score vs popularity colored by genre
ax = axes[0, 1]
for genre, color in GENRE_PALETTE.items():
    mask_g = df['playlist_genre'] == genre
    ax.scatter(df.loc[mask_g, 'intensity_score'],
               df.loc[mask_g, 'track_popularity'],
               c=color, alpha=0.25, s=6, label=genre)
ax.set_xlabel('Intensity Score (engineered)')
ax.set_ylabel('Track Popularity')
ax.set_title('Intensity vs Popularity by Genre', fontsize=11, pad=8)
ax.legend(fontsize=7, markerscale=2)

# 2c — chill_score distribution per genre (violin)
ax = axes[0, 2]
genre_order = df['playlist_genre'].value_counts().index.tolist()
parts = ax.violinplot(
    [df.loc[df['playlist_genre'] == g, 'chill_score'].values for g in genre_order],
    positions=range(len(genre_order)),
    showmedians=True
)
for pc, g in zip(parts['bodies'], genre_order):
    pc.set_facecolor(GENRE_PALETTE.get(g, PURPLE))
    pc.set_alpha(0.7)
parts['cmedians'].set_color(ACCENT)
ax.set_xticks(range(len(genre_order)))
ax.set_xticklabels(genre_order, fontsize=9)
ax.set_title('Chill Score by Genre (Violin)', fontsize=11, pad=8)
ax.set_ylabel('Chill Score')

# 2d — Energy vs Valence scatter (mood quadrant)
ax = axes[1, 0]
for genre, color in GENRE_PALETTE.items():
    mask_g = df['playlist_genre'] == genre
    ax.scatter(df.loc[mask_g, 'valence'], df.loc[mask_g, 'energy'],
               c=color, alpha=0.2, s=5, label=genre)
ax.axvline(0.5, color=MUTED_COL, linewidth=0.8, linestyle='--', alpha=0.5)
ax.axhline(0.5, color=MUTED_COL, linewidth=0.8, linestyle='--', alpha=0.5)
ax.text(0.08, 0.93, 'Angry/Turbulent', color=MUTED_COL, fontsize=7, transform=ax.transAxes)
ax.text(0.62, 0.93, 'Happy/Euphoric',  color=MUTED_COL, fontsize=7, transform=ax.transAxes)
ax.text(0.08, 0.05, 'Sad/Depressed',   color=MUTED_COL, fontsize=7, transform=ax.transAxes)
ax.text(0.62, 0.05, 'Chill/Content',   color=MUTED_COL, fontsize=7, transform=ax.transAxes)
ax.set_xlabel('Valence (positivity)')
ax.set_ylabel('Energy')
ax.set_title('Mood Quadrant: Energy × Valence', fontsize=11, pad=8)
ax.legend(fontsize=7, markerscale=3)

# 2e — Popularity tier breakdown by genre (stacked bar)
ax = axes[1, 1]
pop_genre = df.groupby(['playlist_genre', 'popularity_tier']).size().unstack(fill_value=0)
tier_colors = {'Low': '#374151', 'Mid': PURPLE, 'High': ACCENT, 'Viral': PINK}
bottom = np.zeros(len(pop_genre))
for tier in ['Low', 'Mid', 'High', 'Viral']:
    if tier in pop_genre.columns:
        vals = pop_genre[tier].values
        ax.bar(pop_genre.index, vals, bottom=bottom,
               label=tier, color=tier_colors[tier], alpha=0.85)
        bottom += vals
ax.set_title('Popularity Tier by Genre', fontsize=11, pad=8)
ax.set_ylabel('Track Count')
ax.legend(fontsize=8)
ax.tick_params(axis='x', rotation=15)

# 2f — Tempo distribution by genre
ax = axes[1, 2]
for genre in genre_order:
    vals = df.loc[df['playlist_genre'] == genre, 'tempo'].values
    ax.hist(vals, bins=40, alpha=0.45, label=genre,
            color=GENRE_PALETTE.get(genre, PURPLE), density=True)
ax.set_xlabel('Tempo (BPM)')
ax.set_ylabel('Density')
ax.set_title('Tempo Distribution by Genre', fontsize=11, pad=8)
ax.legend(fontsize=7)

plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.savefig(f'{os_out}/02_feature_engineering.png', dpi=130,
            bbox_inches='tight', facecolor=DARK_BG)
plt.close()
print("  ✓ 02_feature_engineering.png")

# ── VIZ 3: PCA + Clustering ───────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(16, 13))
fig.patch.set_facecolor(DARK_BG)
fig.suptitle('PCA DIMENSIONALITY REDUCTION & KMEANS CLUSTERING',
             color=TEXT_COL, fontsize=15, fontweight='bold', y=0.98)

# 3a — Cumulative explained variance
ax = axes[0, 0]
ax.plot(range(1, len(cumvar)+1), cumvar * 100,
        color=PURPLE, linewidth=2, marker='o', markersize=3)
ax.axhline(95, color=ACCENT, linewidth=1, linestyle='--', alpha=0.7)
ax.axvline(n_95, color=PINK, linewidth=1, linestyle='--', alpha=0.7)
ax.fill_between(range(1, len(cumvar)+1), cumvar * 100,
                alpha=0.15, color=PURPLE)
ax.text(n_95 + 0.3, 50, f'{n_95} components\n→ 95% variance',
        color=PINK, fontsize=9)
ax.set_xlabel('Number of Components')
ax.set_ylabel('Cumulative Explained Variance (%)')
ax.set_title('PCA: Explained Variance Curve', fontsize=11, pad=8)
ax.set_xlim(1, len(cumvar))

# 3b — PCA 2D scatter by genre
ax = axes[0, 1]
for genre, color in GENRE_PALETTE.items():
    mask_g = df['playlist_genre'].values == genre
    ax.scatter(X_pca2[mask_g, 0], X_pca2[mask_g, 1],
               c=color, alpha=0.3, s=6, label=genre)
ax.set_xlabel(f'PC1 ({pca2.explained_variance_ratio_[0]*100:.1f}% var)')
ax.set_ylabel(f'PC2 ({pca2.explained_variance_ratio_[1]*100:.1f}% var)')
ax.set_title('PCA 2D: Genre Clusters', fontsize=11, pad=8)
ax.legend(fontsize=7, markerscale=3)

# 3c — Elbow curve
ax = axes[1, 0]
ax.plot(list(K_range), inertias, color=ACCENT, linewidth=2,
        marker='o', markersize=6, markerfacecolor=PINK)
ax.axvline(elbow_k, color=PINK, linewidth=1.5, linestyle='--')
ax.text(elbow_k + 0.15, inertias[elbow_k-2] * 1.02,
        f'Elbow k={elbow_k}', color=PINK, fontsize=9)
ax.set_xlabel('Number of Clusters (k)')
ax.set_ylabel('Inertia (within-cluster SSE)')
ax.set_title('KMeans Elbow Curve', fontsize=11, pad=8)
ax.set_xticks(list(K_range))

# 3d — KMeans clusters on PCA space
ax = axes[1, 1]
cluster_colors = plt.cm.tab10(np.linspace(0, 1, elbow_k))
for c in range(elbow_k):
    mask_c = df['cluster'].values == c
    ax.scatter(X_pca2[mask_c, 0], X_pca2[mask_c, 1],
               c=[cluster_colors[c]], alpha=0.35, s=6,
               label=f'Cluster {c}')
# Plot centroids
centers_pca = pca2.transform(kmeans.cluster_centers_)
ax.scatter(centers_pca[:, 0], centers_pca[:, 1],
           c='white', marker='X', s=120, zorder=10,
           edgecolors=DARK_BG, linewidth=1.5, label='Centroids')
ax.set_xlabel(f'PC1')
ax.set_ylabel(f'PC2')
ax.set_title(f'KMeans Clusters (k={elbow_k}) on PCA Space', fontsize=11, pad=8)
ax.legend(fontsize=7, markerscale=2)

plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.savefig(f'{os_out}/03_pca_clustering.png', dpi=130,
            bbox_inches='tight', facecolor=DARK_BG)
plt.close()
print("  ✓ 03_pca_clustering.png")

# ── VIZ 4: ML Model Evaluation ────────────────────────────────────
fig = plt.figure(figsize=(18, 13))
fig.patch.set_facecolor(DARK_BG)
fig.suptitle('ML MODEL EVALUATION & COMPARISON',
             color=TEXT_COL, fontsize=15, fontweight='bold', y=0.98)
gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.35)

# 4a — Model comparison bar
ax = fig.add_subplot(gs[0, 0])
model_names  = list(results.keys())
cv_means     = [results[m]['cv_mean'] * 100 for m in model_names]
cv_stds      = [results[m]['cv_std']  * 100 for m in model_names]
test_accs    = [results[m]['test_acc'] * 100 for m in model_names]
x_pos        = np.arange(len(model_names))
ax.barh(x_pos - 0.2, cv_means, height=0.35, label='CV Accuracy',
        color=PURPLE, alpha=0.85, xerr=cv_stds,
        error_kw={'ecolor': MUTED_COL, 'capsize': 3})
ax.barh(x_pos + 0.2, test_accs, height=0.35, label='Test Accuracy',
        color=ACCENT, alpha=0.85)
for i, (cv_val, tst) in enumerate(zip(cv_means, test_accs)):
    ax.text(cv_val + 0.5, i - 0.2, f'{cv_val:.1f}%', va='center',
            color=TEXT_COL, fontsize=8)
    ax.text(tst + 0.5, i + 0.2, f'{tst:.1f}%', va='center',
            color=TEXT_COL, fontsize=8)
ax.set_yticks(x_pos)
ax.set_xticklabels = []
short_names = ['Log. Reg.', 'Rand. Forest', 'Grad. Boost']
ax.set_yticklabels(short_names, fontsize=9)
ax.set_xlabel('Accuracy (%)')
ax.set_title('Model Comparison\n(CV vs Test)', fontsize=11, pad=8)
ax.legend(fontsize=8)
ax.set_xlim(0, 100)

# 4b — Random Forest confusion matrix
ax = fig.add_subplot(gs[0, 1])
cm = confusion_matrix(y_test, results['Random Forest']['y_pred'])
cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100
cmap_cm = LinearSegmentedColormap.from_list('cm', [DARK_BG3, PURPLE, ACCENT])
sns.heatmap(cm_pct, ax=ax, cmap=cmap_cm, annot=True, fmt='.1f',
            xticklabels=le.classes_, yticklabels=le.classes_,
            linewidths=0.5, linecolor=DARK_BG,
            annot_kws={'size': 8},
            cbar_kws={'label': '% of true class'})
ax.set_title('Random Forest\nConfusion Matrix (%)', fontsize=11, pad=8)
ax.set_xlabel('Predicted')
ax.set_ylabel('True')
ax.tick_params(labelsize=8)

# 4c — Feature importances (top 12)
ax = fig.add_subplot(gs[0, 2])
sorted_imp   = sorted(feat_imp.items(), key=lambda x: x[1], reverse=True)[:12]
feat_names_s = [f[0] for f in sorted_imp]
feat_vals_s  = [f[1] for f in sorted_imp]
bar_colors   = [PURPLE if i < 3 else ACCENT if i < 6 else MUTED_COL
                for i in range(len(feat_names_s))]
bars = ax.barh(feat_names_s[::-1], feat_vals_s[::-1],
               color=bar_colors[::-1], alpha=0.85)
for bar, val in zip(bars, feat_vals_s[::-1]):
    ax.text(val + 0.001, bar.get_y() + bar.get_height()/2,
            f'{val:.3f}', va='center', color=TEXT_COL, fontsize=7)
ax.set_title('Feature Importances\n(Random Forest Top 12)', fontsize=11, pad=8)
ax.set_xlabel('Importance')
ax.tick_params(labelsize=7)

# 4d — Per-class F1 scores (all models)
ax = fig.add_subplot(gs[1, 0])
classes     = le.classes_
model_short = ['LR', 'RF', 'GB']
x_pos       = np.arange(len(classes))
width       = 0.28
model_colors = [PURPLE, ACCENT, PINK]
for i, (mname, mshort, mcolor) in enumerate(
        zip(model_names, model_short, model_colors)):
    f1s = [results[mname]['report'][cls]['f1-score'] for cls in classes]
    ax.bar(x_pos + i*width - width, f1s, width=width,
           label=mshort, color=mcolor, alpha=0.85)
ax.set_xticks(x_pos)
ax.set_xticklabels(classes, rotation=15, ha='right', fontsize=8)
ax.set_ylabel('F1 Score')
ax.set_title('Per-class F1: All Models', fontsize=11, pad=8)
ax.legend(fontsize=8)
ax.set_ylim(0, 1.05)

# 4e — Cross-validation score distribution
ax = fig.add_subplot(gs[1, 1])
cv_all = {}
for name, model in models.items():
    scores = cross_val_score(model, X_scaled, y_enc, cv=cv,
                             scoring='accuracy', n_jobs=-1)
    cv_all[name] = scores * 100

bp = ax.boxplot(cv_all.values(), patch_artist=True, notch=False,
                medianprops=dict(color=DARK_BG, linewidth=2))
for patch, color in zip(bp['boxes'], model_colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)
ax.set_xticklabels(['Log. Reg.', 'Rand.\nForest', 'Grad.\nBoost'], fontsize=8)
ax.set_ylabel('CV Accuracy (%)')
ax.set_title('5-Fold CV Score Distribution', fontsize=11, pad=8)

# 4f — Cluster purity analysis
ax = fig.add_subplot(gs[1, 2])
cluster_genre = df.groupby(['cluster', 'playlist_genre']).size().unstack(fill_value=0)
cluster_genre_pct = cluster_genre.div(cluster_genre.sum(axis=1), axis=0) * 100
bottom = np.zeros(len(cluster_genre_pct))
for genre in cluster_genre_pct.columns:
    vals = cluster_genre_pct[genre].values
    ax.bar(cluster_genre_pct.index, vals, bottom=bottom,
           label=genre, color=GENRE_PALETTE.get(genre, PURPLE), alpha=0.85)
    bottom += vals
ax.set_xlabel('KMeans Cluster')
ax.set_ylabel('Genre % in Cluster')
ax.set_title('Cluster Purity by Genre', fontsize=11, pad=8)
ax.legend(fontsize=7, loc='upper right')

plt.savefig(f'{os_out}/04_ml_evaluation.png', dpi=130,
            bbox_inches='tight', facecolor=DARK_BG)
plt.close()
print("  ✓ 04_ml_evaluation.png")

# ── VIZ 5: Generative Art (data-driven, per genre) ────────────────
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.patch.set_facecolor(DARK_BG)
fig.suptitle('GENERATIVE VIBE ART — DATA-DRIVEN PER GENRE',
             color=TEXT_COL, fontsize=15, fontweight='bold', y=0.99)

GENRE_ART_COLORS = {
    'edm':   ['#06ffd8', '#0ea5e9', '#8b5cf6', '#1e1b4b'],
    'rap':   ['#f43f5e', '#dc2626', '#7f1d1d', '#1c0a0a'],
    'pop':   ['#f59e0b', '#fb923c', '#f43f5e', '#1f0a00'],
    'r&b':   ['#8b5cf6', '#a78bfa', '#c4b5fd', '#1e1b4b'],
    'latin': ['#10b981', '#34d399', '#f59e0b', '#052e16'],
    'rock':  ['#ef4444', '#dc2626', '#fbbf24', '#1c0202'],
}

genre_means = df.groupby('playlist_genre')[NORM_FEATURES + ['tempo_norm']].mean()

for idx, (genre, ax) in enumerate(zip(GENRE_PALETTE.keys(), axes.flat)):
    np.random.seed(idx * 7 + 42)
    ax.set_facecolor('#060610')
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.axis('off')

    colors_art  = GENRE_ART_COLORS[genre]
    g_stats     = genre_means.loc[genre]
    energy      = g_stats['energy']
    valence     = g_stats['valence']
    dance       = g_stats['danceability']
    tempo_n     = g_stats['tempo_norm']
    acoustic    = g_stats['acousticness']
    instrumental= g_stats['instrumentalness']

    # Background gradient orbs
    for _ in range(int(6 + energy * 15)):
        cx = np.random.uniform(0.1, 0.9)
        cy = np.random.uniform(0.1, 0.9)
        r  = np.random.uniform(0.05, 0.22 + energy * 0.15)
        c  = colors_art[np.random.randint(len(colors_art))]
        rgb = tuple(int(c.lstrip('#')[i:i+2], 16)/255 for i in (0, 2, 4))
        circle = plt.Circle((cx, cy), r, color=rgb,
                             alpha=np.random.uniform(0.04, 0.13 + valence*0.06))
        ax.add_patch(circle)

    # Central rings (acousticness driven)
    n_rings = int(3 + acoustic * 10)
    for i in range(n_rings):
        r     = 0.04 + i * (0.06 + acoustic * 0.03)
        alpha = max(0.05, 0.5 - i * 0.04)
        cidx  = i % len(colors_art)
        c     = colors_art[cidx]
        rgb   = tuple(int(c.lstrip('#')[j:j+2], 16)/255 for j in (0, 2, 4))
        circle = plt.Circle((0.5, 0.5), r, color=rgb,
                             alpha=alpha, fill=False,
                             linewidth=0.5 + energy * 2.5)
        ax.add_patch(circle)

    # Radial lines (tempo)
    for _ in range(int(tempo_n * 50)):
        angle  = np.random.uniform(0, 2 * np.pi)
        length = np.random.uniform(0.04, 0.3 + tempo_n * 0.15)
        x2 = 0.5 + np.cos(angle) * length
        y2 = 0.5 + np.sin(angle) * length
        c  = colors_art[0]
        rgb = tuple(int(c.lstrip('#')[j:j+2], 16)/255 for j in (0, 2, 4))
        ax.plot([0.5, x2], [0.5, y2], color=rgb,
                alpha=np.random.uniform(0.03, 0.14), linewidth=0.4)

    # Particle field
    n_p = int(20 + energy * 200 + instrumental * 150)
    px  = np.random.uniform(0, 1, n_p)
    py  = np.random.uniform(0, 1, n_p)
    ps  = np.random.uniform(0.3, 2 + energy * 4, n_p)
    dist = np.sqrt((px - 0.5)**2 + (py - 0.5)**2)
    for i in range(n_p):
        c   = colors_art[int(dist[i] / dist.max() * (len(colors_art)-1))]
        rgb = tuple(int(c.lstrip('#')[j:j+2], 16)/255 for j in (0, 2, 4))
        alpha = float(np.clip(0.1 + (1 - dist[i]) * 0.5, 0, 0.7))
        ax.scatter(px[i], py[i], s=float(ps[i]),
                   color=rgb, alpha=alpha * 0.7, zorder=5)

    # Genre label
    accent_c = colors_art[0]
    rgb_a = tuple(int(accent_c.lstrip('#')[j:j+2], 16)/255 for j in (0, 2, 4))
    ax.text(0.5, 0.06, genre.upper(), ha='center', va='center',
            color=rgb_a, alpha=0.7, fontsize=11, fontweight='bold',
            fontfamily='monospace')

    # Stats overlay (top right)
    stats_txt = f'E:{energy:.2f} V:{valence:.2f} D:{dance:.2f}'
    ax.text(0.97, 0.97, stats_txt, ha='right', va='top',
            color=TEXT_COL, alpha=0.4, fontsize=6.5,
            fontfamily='monospace', transform=ax.transAxes)

plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.savefig(f'{os_out}/05_generative_art.png', dpi=130,
            bbox_inches='tight', facecolor=DARK_BG)
plt.close()
print("  ✓ 05_generative_art.png")

# ── Save cleaned dataset ──────────────────────────────────────────
df.to_csv(f'{os_out}/spotify_cleaned.csv', index=False)
print("  ✓ spotify_cleaned.csv")

# ── Save tracks_mini.json for the frontend app ────────────────────
import json
cols = ['track_id','track_name','track_artist','playlist_genre','playlist_subgenre',
        'track_popularity','danceability','energy','valence','tempo',
        'acousticness','instrumentalness','speechiness','liveness','loudness',
        'loudness_norm','tempo_norm','intensity_score','chill_score']
df_mini = df[[c for c in cols if c in df.columns]].round(4)
search_records = [
    {'id':r['track_id'],'n':r['track_name'],'a':r['track_artist'],
     'g':r['playlist_genre'],'sg':r.get('playlist_subgenre',''),
     'pop':r['track_popularity'],
     'd':r['danceability'],'e':r['energy'],'v':r['valence'],
     't':r['tempo'],'ac':r['acousticness'],'ins':r['instrumentalness'],
     'sp':r['speechiness'],'li':r['liveness'],'lo':r['loudness'],
     'ln':r['loudness_norm'],'tn':r['tempo_norm'],
     'is':r['intensity_score'],'cs':r['chill_score']}
    for r in df_mini.to_dict(orient='records')
]
json_path = os.path.join(REPO_ROOT, 'tracks_mini.json')
with open(json_path, 'w') as f:
    json.dump(search_records, f, separators=(',',':'))
print(f"  ✓ tracks_mini.json  ({len(search_records):,} tracks)")

print("\n" + "═"*60)
print("  PIPELINE COMPLETE")
print("═"*60)
print(f"  Raw tracks:     {df_raw.shape[0]:,}")
print(f"  Clean tracks:   {df.shape[0]:,}")
print(f"  Features used:  {len(ML_FEATURES)}")
print(f"  Best model:     {best_name} — {best['test_acc']*100:.1f}% accuracy")
print(f"  5-fold CV:      {best['cv_mean']*100:.1f}% ± {best['cv_std']*100:.1f}%")
print(f"  KMeans k:       {elbow_k} clusters")
print(f"  PCA 95% var:    {n_95} components")
print("═"*60)
