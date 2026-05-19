"""
nlp_utils.py
============
NLP utilities for COSC 2671 Assignment 2 — Sudden Fuel Price Hike.

Design decisions:
    VADER            — designed for social media text; handles slang, caps, emphasis
    sklearn LDA      — pure Python, no C++ compiler required (replaces gensim)
    Lemmatisation    — produces readable real-word tokens (vs stemming)
    Domain stopwords — removes generic fuel-price terms that mask topic differences

Dependencies (no compiler needed):
    pip install vaderSentiment nltk scikit-learn pandas numpy matplotlib seaborn wordcloud
"""

import re
import ast
import logging
import warnings
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

warnings.filterwarnings("ignore", category=DeprecationWarning)
logger = logging.getLogger(__name__)

for _r in ["stopwords", "punkt", "wordnet", "omw-1.4"]:
    nltk.download(_r, quiet=True)

# ── Singletons ─────────────────────────────────────────────────────────────────
_STOP_WORDS = set(stopwords.words("english"))
_LEMMATIZER = WordNetLemmatizer()
_VADER      = SentimentIntensityAnalyzer()

# Domain stopwords: terms that appear in almost every fuel-price comment
# and provide no topic discrimination — removing them reveals meaningful themes.
FUEL_DOMAIN_STOPWORDS = {
    "fuel", "gas", "petrol", "price", "prices", "hike", "oil", "cost",
    "costs", "pump", "station", "litre", "liter", "gallon", "barrel",
    "crude", "energy", "per", "pay", "paying", "paid", "going", "get",
    "got", "one", "would", "could", "really", "much", "lot", "still",
    "even", "back", "know", "think", "people", "said", "say", "also",
    "like", "just", "good", "make", "well", "need", "right", "come",
    "keep", "way", "thing", "use", "used",
}

# Combined stop list as a Python list (CountVectorizer / TfidfVectorizer format)
_COMBINED_STOP_LIST = list(_STOP_WORDS | FUEL_DOMAIN_STOPWORDS)


# ══════════════════════════════════════════════════════════════════════════════
# 1. TEXT CLEANING
# ══════════════════════════════════════════════════════════════════════════════

def clean_text_basic(text: str) -> str:
    """
    Light cleaning preserving punctuation for VADER.
    Removes URLs, @mentions, #hashtags, non-ASCII.
    Normalises repeated characters ('noooo' -> 'noo').
    """
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+|https\S+", " ", text)
    text = re.sub(r"@\w+",  " ", text)
    text = re.sub(r"#\w+",  " ", text)
    text = re.sub(r"[^\x00-\x7F]", " ", text)
    text = re.sub(r"(.)\1{3,}", r"\1\1", text)   # normalise emphasis
    text = re.sub(r"\s+", " ", text).strip()
    return text


def clean_text_for_nlp(
    text: str,
    remove_stopwords: bool = True,
    lemmatize: bool = True,
    extra_stopwords: set = None,
) -> list:
    """
    Deep cleaning for topic modelling — returns a list of tokens.

    Steps:
        1. clean_text_basic  (URLs, mentions, non-ASCII removed)
        2. Strip punctuation
        3. Word tokenise
        4. Remove English + fuel domain stopwords
        5. WordNet lemmatise
        6. Filter tokens shorter than 3 characters
    """
    stop = _STOP_WORDS | FUEL_DOMAIN_STOPWORDS
    if extra_stopwords:
        stop |= extra_stopwords

    text = clean_text_basic(text)
    text = re.sub(r"[^a-z\s]", " ", text)
    tokens = word_tokenize(text)

    if remove_stopwords:
        tokens = [t for t in tokens if t not in stop]
    if lemmatize:
        tokens = [_LEMMATIZER.lemmatize(t) for t in tokens]
    return [t for t in tokens if len(t) > 2]


def clean_texts_for_vectorizer(texts: list) -> list:
    """
    Clean each text and rejoin tokens as a string.
    Required because CountVectorizer/TfidfVectorizer expects strings, not lists.
    """
    return [" ".join(clean_text_for_nlp(t)) for t in texts]


def parse_token_column(token_str: str) -> list:
    """Convert stored token string from CSV back to a Python list."""
    try:
        return ast.literal_eval(str(token_str))
    except (ValueError, SyntaxError):
        return []


# ══════════════════════════════════════════════════════════════════════════════
# 2. SENTIMENT ANALYSIS (VADER)
# ══════════════════════════════════════════════════════════════════════════════

def get_vader_sentiment(text: str) -> dict:
    """
    VADER sentiment scores for a single text string.

    VADER is chosen because it:
        - Was designed specifically for social media short text
        - Handles slang, capitalisation, and punctuation emphasis
        - Requires no training data or model download
        - Provides a granular compound score plus pos/neg/neu breakdown

    Thresholds (VADER paper, Hutto & Gilbert 2014):
        compound >=  0.05 -> Positive
        compound <= -0.05 -> Negative
        otherwise         -> Neutral

    Returns
    -------
    dict: compound, label, pos, neg, neu
    """
    scores   = _VADER.polarity_scores(str(text))
    compound = scores["compound"]
    label    = ("Positive" if compound >= 0.05
                else "Negative" if compound <= -0.05
                else "Neutral")
    return {
        "compound": round(compound, 4),
        "label":    label,
        "pos":      round(scores["pos"], 4),
        "neg":      round(scores["neg"], 4),
        "neu":      round(scores["neu"], 4),
    }


def apply_sentiment(df: pd.DataFrame, text_col: str = "text") -> pd.DataFrame:
    """Apply VADER to every row. Adds columns: compound, label, pos, neg, neu."""
    logger.info(f"Applying VADER to {len(df)} rows...")
    sent_df = pd.DataFrame(df[text_col].apply(get_vader_sentiment).tolist())
    return pd.concat([df.reset_index(drop=True), sent_df], axis=1)


def get_sentiment_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Return a count + percentage breakdown table for sentiment labels."""
    total   = len(df)
    summary = df["label"].value_counts().reset_index()
    summary.columns = ["label", "count"]
    summary["percentage"] = (summary["count"] / total * 100).round(1)
    return summary


def sentiment_by_group(
    df: pd.DataFrame,
    group_col: str,
    label_col: str = "label",
) -> pd.DataFrame:
    """
    Pivot table of sentiment counts grouped by a column.
    Use group_col='year_month' for time trends or
        group_col='topic_id'  for topic-sentiment cross-analysis.
    """
    return (df.groupby([group_col, label_col])
              .size()
              .unstack(fill_value=0))


# ══════════════════════════════════════════════════════════════════════════════
# 3. TOPIC MODELLING — sklearn LDA  (replaces gensim, no C++ needed)
# ══════════════════════════════════════════════════════════════════════════════

def build_lda_model(
    texts: list,
    num_topics: int = 6,
    max_iter: int = 30,
    max_features: int = 2000,
    min_df: int = 5,
    max_df: float = 0.6,
    ngram_range: tuple = (1, 1),
    random_state: int = 42,
) -> tuple:
    """
    Train a scikit-learn LDA topic model on raw comment text.

    Why sklearn LDA instead of gensim?
        - Pure Python — no Microsoft Visual C++ 14.0 compiler required
        - Identical LDA algorithm (Blei et al. 2003)
        - Well-maintained, ships with scikit-learn (no extra install)
        - Uses CountVectorizer (bag-of-words) instead of gensim corpora

    Parameters
    ----------
    texts        : List of raw comment strings.
    num_topics   : k — number of latent topics to discover.
    max_iter     : EM algorithm iterations (more = better convergence).
    max_features : Vocabulary cap for CountVectorizer.
    min_df       : Min document frequency for a term to be included.
    max_df       : Max document fraction (removes over-common terms).
    ngram_range  : (1,1) unigrams; (1,2) includes bigrams.
    random_state : Reproducibility seed.

    Returns
    -------
    tuple: (lda_model, dtm, vectorizer, cleaned_texts)
        lda_model     : Fitted LatentDirichletAllocation
        dtm           : Document-term matrix (sparse, n_docs x vocab)
        vectorizer    : Fitted CountVectorizer (holds vocabulary)
        cleaned_texts : List of cleaned joined strings (for inspection)
    """
    logger.info(f"Cleaning {len(texts)} documents...")
    cleaned = clean_texts_for_vectorizer(texts)

    vectorizer = CountVectorizer(
        max_features=max_features,
        stop_words=_COMBINED_STOP_LIST,
        min_df=min_df,
        max_df=max_df,
        ngram_range=ngram_range,
        token_pattern=r"(?u)\b[a-z][a-z]{2,}\b",   # min 3-char alpha tokens
    )
    dtm = vectorizer.fit_transform(cleaned)
    logger.info(f"Vocabulary size: {len(vectorizer.get_feature_names_out())} terms")

    logger.info(f"Training LDA: {num_topics} topics, {max_iter} iterations...")
    lda_model = LatentDirichletAllocation(
        n_components=num_topics,
        max_iter=max_iter,
        random_state=random_state,
        learning_method="batch",   # more stable for datasets under ~50k docs
        evaluate_every=5,          # log perplexity every 5 iters
        n_jobs=-1,                 # use all CPU cores
    )
    lda_model.fit(dtm)

    perplexity = lda_model.perplexity(dtm)
    logger.info(f"Training complete. Perplexity: {perplexity:.2f} (lower = better fit)")

    return lda_model, dtm, vectorizer, cleaned


def get_topic_labels(lda_model, vectorizer, num_words: int = 10) -> dict:
    """
    Extract top words per topic from a fitted sklearn LDA model.

    Parameters
    ----------
    lda_model  : Fitted LatentDirichletAllocation.
    vectorizer : Fitted CountVectorizer (provides vocabulary mapping).
    num_words  : Number of top words to return per topic.

    Returns
    -------
    dict: {topic_id (int): [word1, word2, ...]}
    """
    feature_names = vectorizer.get_feature_names_out()
    topics = {}
    for i, component in enumerate(lda_model.components_):
        top_indices = component.argsort()[-num_words:][::-1]
        topics[i]   = [feature_names[j] for j in top_indices]
    return topics


def get_dominant_topic(lda_model, dtm) -> list:
    """
    Assign dominant topic to each document.

    Parameters
    ----------
    lda_model : Fitted LatentDirichletAllocation.
    dtm       : Document-term matrix from the same vectorizer used in training.

    Returns
    -------
    list of dicts: [{"topic_id": int, "prob": float}, ...]
    """
    topic_distributions = lda_model.transform(dtm)
    results = []
    for dist in topic_distributions:
        best_topic = int(dist.argmax())
        results.append({
            "topic_id": best_topic,
            "prob":     round(float(dist[best_topic]), 4),
        })
    return results


def assign_topics_to_df(
    df: pd.DataFrame,
    lda_model,
    dtm,
    vectorizer,
    topic_labels: dict = None,
) -> pd.DataFrame:
    """
    Add topic_id, prob, and optional topic_label columns to a DataFrame.

    Parameters
    ----------
    df           : Original comments DataFrame (must align row-for-row with dtm).
    lda_model    : Fitted LDA model.
    dtm          : Document-term matrix aligned with df.
    vectorizer   : CountVectorizer (used if dtm needs to be regenerated).
    topic_labels : Optional dict from get_topic_labels() for human-readable labels.
    """
    dom_df = pd.DataFrame(get_dominant_topic(lda_model, dtm))
    result = pd.concat([df.reset_index(drop=True), dom_df], axis=1)
    if topic_labels:
        result["topic_label"] = result["topic_id"].map(
            {k: ", ".join(v[:3]) for k, v in topic_labels.items()}
        )
    return result


def get_topic_document_matrix(lda_model, dtm) -> pd.DataFrame:
    """
    Full topic probability distribution for each document.
    Useful for heatmaps and cross-analysis with sentiment.

    Returns pd.DataFrame shape (n_docs, num_topics)
    """
    dist = lda_model.transform(dtm)
    cols = [f"topic_{i}" for i in range(lda_model.n_components)]
    return pd.DataFrame(dist, columns=cols)


# ══════════════════════════════════════════════════════════════════════════════
# 4. OPTIMAL TOPIC COUNT — perplexity-based (replaces gensim c_v coherence)
# ══════════════════════════════════════════════════════════════════════════════

def find_optimal_num_topics(
    texts: list,
    topic_range: range = range(3, 12),
    max_iter: int = 15,
) -> pd.DataFrame:
    """
    Train LDA for each k in topic_range; record perplexity + log-likelihood.

    Perplexity  : lower  = better model fit.
    Log-likelihood: higher = better model fit.
    Pick k at the 'elbow' of the perplexity curve.

    This replaces gensim's c_v CoherenceModel which required
    Cython compilation. Perplexity is sufficient to justify topic
    count selection in the report.

    Returns
    -------
    pd.DataFrame with columns: num_topics, perplexity, log_likelihood
    """
    cleaned    = clean_texts_for_vectorizer(texts)
    vectorizer = CountVectorizer(
        max_features=2000,
        stop_words=_COMBINED_STOP_LIST,
        min_df=5, max_df=0.6,
        token_pattern=r"(?u)\b[a-z][a-z]{2,}\b",
    )
    dtm = vectorizer.fit_transform(cleaned)

    results = []
    for k in topic_range:
        model = LatentDirichletAllocation(
            n_components=k, max_iter=max_iter,
            random_state=42, learning_method="batch",
        )
        model.fit(dtm)
        perp = model.perplexity(dtm)
        ll   = model.score(dtm)
        results.append({
            "num_topics":     k,
            "perplexity":     round(perp, 2),
            "log_likelihood": round(ll,   2),
        })
        logger.info(f"  k={k}: perplexity={perp:.2f}, log_likelihood={ll:.2f}")

    df   = pd.DataFrame(results)
    best = df.loc[df["perplexity"].idxmin()]
    logger.info(f"\nLowest perplexity at k={int(best['num_topics'])} "
                f"(perplexity={best['perplexity']:.2f})")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# 5. TF-IDF ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════

def compute_tfidf_top_terms(
    texts: list,
    top_n: int = 25,
    ngram_range: tuple = (1, 2),
) -> pd.DataFrame:
    """
    Top TF-IDF terms across corpus (includes bigrams by default).
    Bigrams capture 'price increase', 'fuel crisis', 'high cost'.

    Returns pd.DataFrame with columns: term, tfidf_mean
    """
    vec  = TfidfVectorizer(
        max_features=1000,
        stop_words=_COMBINED_STOP_LIST,
        ngram_range=ngram_range,
        token_pattern=r"(?u)\b[a-z][a-z]{2,}\b",
    )
    X    = vec.fit_transform(texts)
    mean = X.mean(axis=0).A1
    return (pd.DataFrame({"term": vec.get_feature_names_out(), "tfidf_mean": mean})
              .sort_values("tfidf_mean", ascending=False)
              .head(top_n)
              .reset_index(drop=True))


def tfidf_by_sentiment(
    df: pd.DataFrame,
    text_col: str = "text_clean",
    label_col: str = "label",
    top_n: int = 15,
) -> dict:
    """
    Top TF-IDF terms per sentiment class.
    Reveals vocabulary differences between Positive, Negative, Neutral comments.

    Returns dict: {"Positive": DataFrame, "Negative": DataFrame, "Neutral": DataFrame}
    """
    results = {}
    for label in ["Positive", "Negative", "Neutral"]:
        subset = df[df[label_col] == label][text_col].dropna().tolist()
        if len(subset) >= 5:
            results[label] = compute_tfidf_top_terms(
                subset, top_n=top_n, ngram_range=(1, 2)
            )
    return results


# ══════════════════════════════════════════════════════════════════════════════
# 6. VISUALISATION
# ══════════════════════════════════════════════════════════════════════════════

def plot_sentiment_distribution(
    df, label_col="label", compound_col="compound", save_path=None
):
    """
    Two-panel figure:
        Left  — Sentiment label bar chart with counts and percentages
        Right — VADER compound score histogram with threshold markers
    """
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    counts = df[label_col].value_counts()
    colors = {"Positive": "#2ecc71", "Negative": "#e74c3c", "Neutral": "#95a5a6"}
    bar_colors = [colors.get(l, "steelblue") for l in counts.index]

    counts.plot(kind="bar", ax=axes[0], color=bar_colors, edgecolor="white")
    axes[0].set_title("Sentiment Distribution")
    axes[0].set_xlabel("Sentiment")
    axes[0].set_ylabel("Number of Comments")
    axes[0].tick_params(axis="x", rotation=0)
    for bar, count in zip(axes[0].patches, counts.values):
        axes[0].text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 10,
            f"{count:,}\n({count/len(df)*100:.1f}%)",
            ha="center", va="bottom", fontsize=9,
        )

    axes[1].hist(df[compound_col], bins=40, color="steelblue",
                 edgecolor="white", alpha=0.85)
    axes[1].axvline( 0.05, color="#2ecc71", linestyle="--",
                    linewidth=1.5, label="Positive threshold (0.05)")
    axes[1].axvline(-0.05, color="#e74c3c", linestyle="--",
                    linewidth=1.5, label="Negative threshold (-0.05)")
    axes[1].set_title("VADER Compound Score Distribution")
    axes[1].set_xlabel("Compound Score")
    axes[1].set_ylabel("Frequency")
    axes[1].legend(fontsize=8)

    plt.suptitle("Sentiment Analysis — YouTube Comments: Fuel Price Hike",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Saved: {save_path}")
    plt.show()


def plot_sentiment_over_time(
    df, time_col="year_month", label_col="label", save_path=None
):
    """Stacked bar chart — sentiment counts per month."""
    pivot = (df.groupby([time_col, label_col])
               .size().unstack(fill_value=0))
    for col in ["Positive", "Neutral", "Negative"]:
        if col not in pivot.columns:
            pivot[col] = 0
    ax = pivot[["Positive", "Neutral", "Negative"]].plot(
        kind="bar", stacked=True, figsize=(14, 5),
        color=["#2ecc71", "#95a5a6", "#e74c3c"], edgecolor="white",
    )
    ax.set_title("Sentiment Trend Over Time — Fuel Price Hike YouTube Comments")
    ax.set_xlabel("Month")
    ax.set_ylabel("Number of Comments")
    plt.xticks(rotation=45, ha="right")
    plt.legend(title="Sentiment")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Saved: {save_path}")
    plt.show()


def plot_topic_perplexity(perplexity_df: pd.DataFrame, save_path=None):
    """
    Two-panel line plot of perplexity + log-likelihood vs number of topics.
    Use the elbow point to justify your chosen k in the report.
    Replaces gensim's c_v coherence plot.
    """
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))

    # Panel 1: Perplexity (lower = better)
    axes[0].plot(perplexity_df["num_topics"], perplexity_df["perplexity"],
                 marker="o", color="coral", linewidth=2)
    best_p = perplexity_df.loc[perplexity_df["perplexity"].idxmin()]
    axes[0].axvline(best_p["num_topics"], color="red", linestyle="--", alpha=0.6,
                    label=f"Lowest k={int(best_p['num_topics'])} "
                          f"({best_p['perplexity']:.1f})")
    axes[0].set_title("Perplexity vs Number of Topics\n(Lower = Better Fit)")
    axes[0].set_xlabel("Number of Topics (k)")
    axes[0].set_ylabel("Perplexity")
    axes[0].set_xticks(perplexity_df["num_topics"])
    axes[0].legend()

    # Panel 2: Log-likelihood (higher = better)
    axes[1].plot(perplexity_df["num_topics"], perplexity_df["log_likelihood"],
                 marker="s", color="teal", linewidth=2)
    best_ll = perplexity_df.loc[perplexity_df["log_likelihood"].idxmax()]
    axes[1].axvline(best_ll["num_topics"], color="teal", linestyle="--", alpha=0.6,
                    label=f"Best k={int(best_ll['num_topics'])} "
                          f"({best_ll['log_likelihood']:.1f})")
    axes[1].set_title("Log-Likelihood vs Number of Topics\n(Higher = Better Fit)")
    axes[1].set_xlabel("Number of Topics (k)")
    axes[1].set_ylabel("Log-Likelihood")
    axes[1].set_xticks(perplexity_df["num_topics"])
    axes[1].legend()

    plt.suptitle("Optimal LDA Topic Count — Fuel Price Hike YouTube Comments",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Saved: {save_path}")
    plt.show()


def plot_topic_top_words(
    lda_model, vectorizer, num_words: int = 8, save_path: str = None
):
    """
    Horizontal bar charts — one panel per topic showing top words by weight.
    More precise than word clouds for identifying distinct topics.
    """
    feature_names = vectorizer.get_feature_names_out()
    topic_labels  = get_topic_labels(lda_model, vectorizer, num_words=num_words)
    n_topics      = lda_model.n_components
    cols          = min(3, n_topics)
    rows          = (n_topics + cols - 1) // cols
    fig, axes     = plt.subplots(rows, cols, figsize=(cols * 5, rows * 3.5))
    axes          = axes.flatten() if n_topics > 1 else [axes]
    colors        = plt.cm.tab10.colors

    for i, ax in enumerate(axes):
        if i >= n_topics:
            ax.axis("off")
            continue
        component   = lda_model.components_[i]
        top_idx     = component.argsort()[-num_words:]
        top_words   = [feature_names[j] for j in top_idx]
        top_weights = component[top_idx]
        ax.barh(top_words, top_weights / top_weights.sum(),
                color=colors[i % len(colors)], edgecolor="white")
        ax.set_title(f"Topic {i}: {', '.join(topic_labels[i][:3])}",
                     fontsize=10, fontweight="bold")
        ax.set_xlabel("Relative Weight")
        ax.tick_params(axis="y", labelsize=8)

    plt.suptitle("LDA Topic Top Words — Fuel Price Hike YouTube Comments",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Saved: {save_path}")
    plt.show()


def plot_topic_sentiment_heatmap(
    df, topic_col="topic_id", label_col="label",
    topic_labels=None, save_path=None
):
    """
    Heatmap: comment counts by topic x sentiment.
    Reveals which topics drive negative or positive discourse.
    """
    pivot = (df[df[topic_col] >= 0]
             .groupby([topic_col, label_col])
             .size().unstack(fill_value=0))
    if topic_labels:
        pivot.index = [
            f"T{i}: {', '.join(topic_labels.get(i, [])[:2])}"
            for i in pivot.index
        ]
    plt.figure(figsize=(9, max(4, len(pivot) * 0.65)))
    sns.heatmap(pivot, annot=True, fmt="d", cmap="YlOrRd",
                linewidths=0.5, cbar_kws={"label": "Comment Count"})
    plt.title("Topic x Sentiment Heatmap — Fuel Price Hike YouTube Comments",
              fontsize=11)
    plt.xlabel("Sentiment")
    plt.ylabel("LDA Topic")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Saved: {save_path}")
    plt.show()


def plot_wordcloud_per_topic(lda_model, vectorizer, save_path: str = None):
    """
    Word cloud for each topic using topic-word weights as frequencies.
    Requires: pip install wordcloud
    """
    try:
        from wordcloud import WordCloud
    except ImportError:
        logger.warning("Install wordcloud: pip install wordcloud")
        return

    feature_names = vectorizer.get_feature_names_out()
    n_topics      = lda_model.n_components
    cols          = min(3, n_topics)
    rows          = (n_topics + cols - 1) // cols
    fig, axes     = plt.subplots(rows, cols, figsize=(cols * 5, rows * 3.5))
    axes          = axes.flatten() if n_topics > 1 else [axes]

    for i, ax in enumerate(axes):
        if i >= n_topics:
            ax.axis("off")
            continue
        component = lda_model.components_[i]
        freq_dict = {feature_names[j]: float(component[j])
                     for j in component.argsort()[-50:]}
        wc = WordCloud(width=400, height=250, background_color="white",
                       colormap="viridis").generate_from_frequencies(freq_dict)
        ax.imshow(wc, interpolation="bilinear")
        ax.set_title(f"Topic {i}", fontsize=10)
        ax.axis("off")

    plt.suptitle("Word Clouds per LDA Topic — Fuel Price Hike",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Saved: {save_path}")
    plt.show()
