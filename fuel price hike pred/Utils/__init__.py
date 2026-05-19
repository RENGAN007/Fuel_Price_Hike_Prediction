"""
utils/
======
Utility modules for COSC 2671 Assignment 2 — Sudden Fuel Price Hike.

Modules:
    youtube_collector  : YouTube Data API v3 data collection
    network_builder    : NetworkX graph construction and analysis
    nlp_utils          : VADER sentiment, sklearn LDA topic modelling, TF-IDF
"""

# ── YouTube Data Collection ────────────────────────────────────────────────────
from .youtube_collector import (
    get_youtube_client,
    search_videos,
    get_video_stats,
    get_comments,
    get_channel_info,
    collect_full_dataset,
    estimate_quota,
)

# ── Network Construction & Analysis ───────────────────────────────────────────
from .network_builder import (
    build_user_cocomment_network,
    build_channel_interaction_network,
    compute_network_metrics,
    detect_communities,
    get_community_summary,
    get_network_stats,
    plot_network,
    plot_degree_distribution,
    plot_centrality_comparison,
)

# ── NLP: Sentiment + Topic Modelling (sklearn — no gensim) ────────────────────
from .nlp_utils import (
    # Text cleaning
    clean_text_basic,
    clean_text_for_nlp,
    clean_texts_for_vectorizer,
    parse_token_column,
    # Sentiment (VADER)
    get_vader_sentiment,
    apply_sentiment,
    get_sentiment_summary,
    sentiment_by_group,
    # Topic modelling (sklearn LDA)
    build_lda_model,
    get_topic_labels,
    get_dominant_topic,
    assign_topics_to_df,
    get_topic_document_matrix,
    find_optimal_num_topics,
    # TF-IDF
    compute_tfidf_top_terms,
    tfidf_by_sentiment,
    # Visualisation
    plot_sentiment_distribution,
    plot_sentiment_over_time,
    plot_topic_perplexity,          # replaces plot_topic_coherence (gensim removed)
    plot_topic_top_words,
    plot_topic_sentiment_heatmap,
    plot_wordcloud_per_topic,
    # Constants
    FUEL_DOMAIN_STOPWORDS,
)