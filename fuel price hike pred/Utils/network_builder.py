"""
network_builder.py
==================
Network construction and analysis utilities for COSC 2671 Assignment 2.
Sudden Fuel Price Hike — YouTube Social Media Analysis.

Design decisions:
    Co-comment network  : Undirected, weighted — co-occurrence has no direction
    Channel network     : Directed, weighted   — engagement flows from commenter to channel
    Louvain communities : Fast, scalable, maximises modularity on weighted graphs
    PageRank + Betweenness : Captures global influence AND bridge/broker roles

Dependencies:
    pip install networkx python-louvain matplotlib pandas numpy
"""

import logging
import warnings
from collections import defaultdict, Counter
from itertools import combinations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import networkx as nx

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)

# ── Safe Louvain import ───────────────────────────────────────────────────────
# python-louvain installs as 'community' but is imported as 'community'
# networkx also ships community detection — we fall back to greedy modularity
# so the code never crashes even if python-louvain is missing.
try:
    import community as community_louvain          # python-louvain package
    _LOUVAIN_AVAILABLE = True
    logger.info("python-louvain loaded successfully.")
except ImportError:
    try:
        from community import community_louvain    # alternate import path
        _LOUVAIN_AVAILABLE = True
    except ImportError:
        community_louvain  = None
        _LOUVAIN_AVAILABLE = False
        logger.warning(
            "python-louvain not found. Falling back to NetworkX greedy modularity.\n"
            "For best results: pip install python-louvain"
        )


# ══════════════════════════════════════════════════════════════════════════════
# 1. NETWORK CONSTRUCTION
# ══════════════════════════════════════════════════════════════════════════════

def build_user_cocomment_network(
    comments_df: pd.DataFrame,
    user_col: str = "author_id",
    video_col: str = "video_id",
    min_shared_videos: int = 1,
    max_users_per_video: int = 500,
) -> nx.Graph:
    """
    Build an undirected weighted co-comment network.

    Nodes  : YouTube commenters (author_id)
    Edges  : Two users both commented on the same video
    Weight : Number of shared videos commented on

    Why undirected? Co-occurrence has no inherent direction.
    Why weighted?   More shared videos = stronger latent relationship.

    Parameters
    ----------
    comments_df        : DataFrame with at least user_col and video_col.
    user_col           : Column name for commenter identifier.
    video_col          : Column name for video identifier.
    min_shared_videos  : Minimum shared videos to create an edge (noise filter).
    max_users_per_video: Skip videos with more than this many commenters
                         (avoids dense cliques from viral videos distorting structure).

    Returns
    -------
    nx.Graph (undirected, weighted)
    """
    df = comments_df[[user_col, video_col]].dropna()
    df = df.drop_duplicates()

    # Group commenters by video
    video_users = df.groupby(video_col)[user_col].apply(list).to_dict()

    edge_weights = defaultdict(int)
    skipped = 0

    for vid, users in video_users.items():
        if len(users) > max_users_per_video:
            skipped += 1
            continue
        for u1, u2 in combinations(sorted(set(users)), 2):
            edge_weights[(u1, u2)] += 1

    if skipped:
        logger.info(f"Skipped {skipped} high-traffic videos (>{max_users_per_video} commenters)")

    G = nx.Graph()
    for (u1, u2), weight in edge_weights.items():
        if weight >= min_shared_videos:
            G.add_edge(u1, u2, weight=weight)

    # Add isolated nodes (commenters with no co-commenter)
    all_users = df[user_col].unique()
    G.add_nodes_from(all_users)

    logger.info(
        f"Co-comment network: {G.number_of_nodes()} nodes, "
        f"{G.number_of_edges()} edges (min_shared={min_shared_videos})"
    )
    return G


def build_channel_interaction_network(
    comments_df: pd.DataFrame,
    videos_df: pd.DataFrame,
    commenter_col: str = "author_channel_id",
    video_col: str = "video_id",
    channel_col: str = "channel_id",
    include_self_loops: bool = False,
) -> nx.DiGraph:
    """
    Build a directed weighted channel interaction network.

    Nodes  : YouTube channels
    Edges  : Channel A's subscriber commented on Channel B's video
             (directed: A → B, i.e. engagement flows TO B)
    Weight : Number of such cross-channel interactions

    Why directed? Engagement flows FROM the commenter's channel TO the content channel.

    Parameters
    ----------
    comments_df      : DataFrame with commenter channel id and video_id.
    videos_df        : DataFrame with video_id and channel_id.
    commenter_col    : Column in comments_df for commenter's channel.
    video_col        : Join key present in both DataFrames.
    channel_col      : Column in videos_df for content channel.
    include_self_loops: Whether to include edges where source == target.

    Returns
    -------
    nx.DiGraph (directed, weighted)
    """
    merged = comments_df[[commenter_col, video_col]].merge(
        videos_df[[video_col, channel_col]], on=video_col, how="inner"
    ).dropna()

    edge_weights = defaultdict(int)
    for _, row in merged.iterrows():
        src = row[commenter_col]
        dst = row[channel_col]
        if src == dst and not include_self_loops:
            continue
        edge_weights[(src, dst)] += 1

    G = nx.DiGraph()
    for (src, dst), weight in edge_weights.items():
        G.add_edge(src, dst, weight=weight)

    logger.info(
        f"Channel interaction network: {G.number_of_nodes()} nodes, "
        f"{G.number_of_edges()} edges"
    )
    return G


# ══════════════════════════════════════════════════════════════════════════════
# 2. COMMUNITY DETECTION
# ══════════════════════════════════════════════════════════════════════════════

def detect_communities(G: nx.Graph, weight: str = "weight") -> dict:
    """
    Louvain community detection on an undirected weighted graph.

    Uses python-louvain if available; falls back to NetworkX greedy modularity
    so the function never raises ImportError.

    Why Louvain?
        - Fast and scalable (near-linear time)
        - Handles weighted graphs natively
        - Maximises modularity — a standard measure of community quality
        - Widely used in social network research

    Parameters
    ----------
    G      : Undirected NetworkX graph.
    weight : Edge attribute to use as weight.

    Returns
    -------
    dict mapping node → community_id (int)
    """
    if G.number_of_nodes() == 0:
        logger.warning("Empty graph passed to detect_communities.")
        return {}

    # Convert to undirected if directed graph passed
    G_und = G.to_undirected() if isinstance(G, nx.DiGraph) else G

    if _LOUVAIN_AVAILABLE:
        logger.info("Running Louvain community detection (python-louvain)...")
        partition = community_louvain.best_partition(G_und, weight=weight)
        method = "Louvain"
    else:
        # Fallback: NetworkX greedy modularity (no extra install needed)
        logger.info("Running greedy modularity community detection (NetworkX fallback)...")
        communities_generator = nx.community.greedy_modularity_communities(
            G_und, weight=weight
        )
        partition = {}
        for cid, community_set in enumerate(communities_generator):
            for node in community_set:
                partition[node] = cid
        method = "Greedy Modularity (fallback)"

    n_communities = len(set(partition.values()))
    logger.info(f"{method}: {n_communities} communities detected")
    return partition


def compute_modularity(G: nx.Graph, partition: dict, weight: str = "weight") -> float:
    """
    Compute modularity score for a given partition.
    Modularity > 0.3 generally indicates meaningful community structure.

    Returns
    -------
    float: modularity score (-1 to 1, higher = better community structure)
    """
    if _LOUVAIN_AVAILABLE:
        return community_louvain.modularity(partition, G, weight=weight)
    else:
        # NetworkX modularity
        community_sets = defaultdict(set)
        for node, cid in partition.items():
            community_sets[cid].add(node)
        return nx.community.modularity(G, list(community_sets.values()), weight=weight)


def get_community_summary(
    G: nx.Graph,
    partition: dict,
    metrics_df: pd.DataFrame = None,
) -> pd.DataFrame:
    """
    Summary table: community size, density, and top node per community.

    Parameters
    ----------
    G           : The original graph.
    partition   : Node → community_id dict from detect_communities().
    metrics_df  : Optional DataFrame from compute_network_metrics() to get top node.

    Returns
    -------
    pd.DataFrame with columns: community_id, size, density, top_node (if metrics provided)
    """
    community_nodes = defaultdict(list)
    for node, cid in partition.items():
        community_nodes[cid].append(node)

    rows = []
    for cid, nodes in sorted(community_nodes.items(), key=lambda x: -len(x[1])):
        subgraph = G.subgraph(nodes)
        row = {
            "community_id": cid,
            "size":         len(nodes),
            "density":      round(nx.density(subgraph), 5),
            "internal_edges": subgraph.number_of_edges(),
        }
        if metrics_df is not None and "node" in metrics_df.columns:
            community_metrics = metrics_df[metrics_df["node"].isin(nodes)]
            if not community_metrics.empty:
                row["top_node_pagerank"] = community_metrics.iloc[0]["node"]
        rows.append(row)

    return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════════════════════════
# 3. CENTRALITY & NETWORK METRICS
# ══════════════════════════════════════════════════════════════════════════════

def compute_network_metrics(
    G: nx.Graph,
    weight: str = "weight",
    top_n_betweenness: int = 500,
) -> pd.DataFrame:
    """
    Compute centrality measures for all nodes and return a ranked DataFrame.

    Measures computed:
        degree_c     : Degree centrality — how many direct connections
        pagerank     : PageRank — global influence via network walks
        betweenness_c: Betweenness centrality — broker/bridge role
        closeness_c  : Closeness centrality — how quickly info reaches all others

    Why these measures?
        - PageRank captures recursive influence (a user connected to influencers scores higher)
        - Betweenness identifies users who bridge otherwise-disconnected communities
        - Together they distinguish popular users from structurally important ones

    Parameters
    ----------
    G                   : NetworkX graph (directed or undirected).
    weight              : Edge weight attribute name.
    top_n_betweenness   : Betweenness approximation k (full = None, slow on large graphs).

    Returns
    -------
    pd.DataFrame sorted by PageRank descending.
    """
    if G.number_of_nodes() == 0:
        return pd.DataFrame()

    logger.info(f"Computing centrality for {G.number_of_nodes()} nodes...")

    # Work on largest connected component for undirected graphs
    # to avoid division-by-zero in closeness on disconnected graphs
    if isinstance(G, nx.Graph) and not isinstance(G, nx.DiGraph):
        largest_cc = max(nx.connected_components(G), key=len)
        G_lcc = G.subgraph(largest_cc).copy()
    else:
        G_lcc = G

    degree_c     = nx.degree_centrality(G_lcc)
    pagerank     = nx.pagerank(G_lcc, weight=weight, max_iter=200, tol=1e-6)
    closeness_c  = nx.closeness_centrality(G_lcc, distance=None)

    # Betweenness: approximate with k samples on large graphs for speed
    n = G_lcc.number_of_nodes()
    k = min(top_n_betweenness, n) if n > top_n_betweenness else None
    betweenness_c = nx.betweenness_centrality(G_lcc, weight=weight, k=k, normalized=True)

    rows = []
    for node in G_lcc.nodes():
        rows.append({
            "node":          node,
            "degree":        G_lcc.degree(node, weight=weight),
            "degree_c":      round(degree_c.get(node, 0), 6),
            "pagerank":      round(pagerank.get(node, 0), 6),
            "betweenness_c": round(betweenness_c.get(node, 0), 6),
            "closeness_c":   round(closeness_c.get(node, 0), 6),
        })

    df = pd.DataFrame(rows).sort_values("pagerank", ascending=False).reset_index(drop=True)
    logger.info("Centrality computation complete.")
    return df


def get_network_stats(G: nx.Graph) -> dict:
    """
    Return a summary statistics dictionary for reporting.

    Returns
    -------
    dict suitable for printing in a report or converting to a table.
    """
    stats = {
        "Nodes":               G.number_of_nodes(),
        "Edges":               G.number_of_edges(),
        "Density":             round(nx.density(G), 6),
        "Is directed":         isinstance(G, nx.DiGraph),
    }

    if isinstance(G, nx.Graph) and not isinstance(G, nx.DiGraph):
        components = list(nx.connected_components(G))
        stats["Connected components"]  = len(components)
        stats["Largest component size"] = max(len(c) for c in components)
        stats["Is connected"]          = nx.is_connected(G)

        degrees = [d for _, d in G.degree()]
        stats["Average degree"]  = round(np.mean(degrees), 3)
        stats["Max degree"]      = max(degrees)
        stats["Min degree"]      = min(degrees)

        # Transitivity (global clustering coefficient)
        stats["Transitivity (clustering)"] = round(nx.transitivity(G), 4)

    return stats


# ══════════════════════════════════════════════════════════════════════════════
# 4. VISUALISATION
# ══════════════════════════════════════════════════════════════════════════════

def plot_network(
    G: nx.Graph,
    partition: dict = None,
    title: str = "Network Graph",
    top_n: int = 100,
    layout: str = "spring",
    save_path: str = None,
    figsize: tuple = (14, 10),
) -> None:
    """
    Visualise the network, colouring nodes by community if partition provided.

    Parameters
    ----------
    G         : NetworkX graph to visualise.
    partition : Optional node → community_id dict (colours nodes).
    title     : Plot title.
    top_n     : Only visualise the top_n nodes by degree (avoids overcrowding).
    layout    : 'spring', 'kamada_kawai', or 'circular'.
    save_path : If provided, save figure to this path.
    figsize   : Figure dimensions.
    """
    if G.number_of_nodes() == 0:
        logger.warning("Empty graph — nothing to plot.")
        return

    # Restrict to top_n nodes by degree for readability
    if G.number_of_nodes() > top_n:
        top_nodes = sorted(G.nodes(), key=lambda n: G.degree(n, weight="weight"),
                           reverse=True)[:top_n]
        G_sub = G.subgraph(top_nodes).copy()
    else:
        G_sub = G.copy()

    # Layout
    logger.info(f"Computing {layout} layout for {G_sub.number_of_nodes()} nodes...")
    if layout == "kamada_kawai":
        pos = nx.kamada_kawai_layout(G_sub)
    elif layout == "circular":
        pos = nx.circular_layout(G_sub)
    else:
        pos = nx.spring_layout(G_sub, seed=42, k=1.5 / (G_sub.number_of_nodes() ** 0.5))

    # Node colours by community
    if partition:
        community_ids = [partition.get(n, 0) for n in G_sub.nodes()]
        cmap    = cm.get_cmap("tab20", max(community_ids) + 1)
        colors  = [cmap(cid) for cid in community_ids]
    else:
        colors = "steelblue"

    # Node sizes by degree
    degrees    = dict(G_sub.degree(weight="weight"))
    max_degree = max(degrees.values()) if degrees else 1
    node_sizes = [100 + 900 * (degrees.get(n, 0) / max_degree) for n in G_sub.nodes()]

    # Edge widths by weight
    weights    = [G_sub[u][v].get("weight", 1) for u, v in G_sub.edges()]
    max_w      = max(weights) if weights else 1
    edge_widths = [0.3 + 2.5 * (w / max_w) for w in weights]

    fig, ax = plt.subplots(figsize=figsize)
    nx.draw_networkx_edges(G_sub, pos, ax=ax, width=edge_widths,
                           alpha=0.25, edge_color="grey")
    nx.draw_networkx_nodes(G_sub, pos, ax=ax, node_color=colors,
                           node_size=node_sizes, alpha=0.85)

    # Label only the top 10 highest-degree nodes
    top10 = sorted(degrees, key=degrees.get, reverse=True)[:10]
    labels = {n: str(n)[:15] for n in top10}
    nx.draw_networkx_labels(G_sub, pos, labels=labels, ax=ax,
                            font_size=7, font_color="black")

    ax.set_title(title, fontsize=12, fontweight="bold", pad=15)
    ax.axis("off")
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Saved: {save_path}")
    plt.show()


def plot_degree_distribution(
    G: nx.Graph,
    title: str = "Degree Distribution",
    log_scale: bool = True,
    save_path: str = None,
) -> None:
    """
    Degree distribution histogram with optional log-log scale.
    Power-law distributions are characteristic of real social networks.
    """
    degrees = [d for _, d in G.degree()]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))

    # Linear scale
    axes[0].hist(degrees, bins=30, color="steelblue", edgecolor="white")
    axes[0].set_title(f"{title} (Linear)")
    axes[0].set_xlabel("Degree")
    axes[0].set_ylabel("Frequency")

    # Log-log scale (reveals power-law behaviour)
    if log_scale and max(degrees) > 1:
        degree_counts = Counter(degrees)
        x = sorted(degree_counts.keys())
        y = [degree_counts[d] for d in x]
        axes[1].loglog(x, y, "o", markersize=4, color="coral", alpha=0.7)
        axes[1].set_title(f"{title} (Log-Log)")
        axes[1].set_xlabel("Degree (log)")
        axes[1].set_ylabel("Frequency (log)")
        axes[1].grid(True, alpha=0.3)
    else:
        axes[1].axis("off")

    plt.suptitle("Network Degree Distribution — Fuel Price Hike Co-Comment Network",
                 fontsize=11, fontweight="bold")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Saved: {save_path}")
    plt.show()


def plot_centrality_comparison(
    metrics_df: pd.DataFrame,
    top_n: int = 20,
    save_path: str = None,
) -> None:
    """
    Three-panel bar chart comparing PageRank, Betweenness, and Degree
    for the top_n nodes. Helps identify users who score high on all three
    (globally influential AND structurally important bridges).
    """
    top = metrics_df.head(top_n).copy()
    top["node_short"] = top["node"].astype(str).str[:18]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    metrics   = [("pagerank", "PageRank",      "steelblue"),
                 ("betweenness_c", "Betweenness", "coral"),
                 ("degree_c",      "Degree",      "teal")]

    for ax, (col, label, color) in zip(axes, metrics):
        if col not in top.columns:
            ax.set_title(f"{label} (not available)")
            ax.axis("off")
            continue
        sorted_top = top.sort_values(col, ascending=True)
        ax.barh(sorted_top["node_short"], sorted_top[col],
                color=color, edgecolor="white")
        ax.set_title(f"Top {top_n} by {label}", fontsize=11)
        ax.set_xlabel(label)
        ax.tick_params(axis="y", labelsize=8)

    plt.suptitle("Centrality Comparison — Fuel Price Hike YouTube Co-Comment Network",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Saved: {save_path}")
    plt.show()


# ── make os available for save_path makedirs ──────────────────────────────────
import os