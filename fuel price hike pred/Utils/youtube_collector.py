"""
youtube_collector.py
====================
YouTube Data API v3 helper module for the "Sudden Fuel Price Hike"
Social Media & Network Analysis project (COSC 2671 Assignment 2).

Quota cost reference:
    search.list          → 100 units per call
    videos.list          → 1 unit per call (up to 50 IDs)
    commentThreads.list  → 1 unit per call (up to 100 comments)
    Daily quota limit    → 10,000 units (default)
"""

import os
import time
import logging
import pandas as pd
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from dotenv import load_dotenv
from tqdm import tqdm

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

load_dotenv()
_API_KEY = os.getenv("YOUTUBE_API_KEY")


# ── Client ─────────────────────────────────────────────────────────────────────

def get_youtube_client(api_key: str = None):
    """Build and return a YouTube Data API v3 service client."""
    key = api_key or _API_KEY
    if not key:
        raise EnvironmentError(
            "YOUTUBE_API_KEY not set. Add it to your .env file:\n"
            "  YOUTUBE_API_KEY=your_key_here"
        )
    return build("youtube", "v3", developerKey=key)


# ── Video Search ───────────────────────────────────────────────────────────────

def search_videos(
    query: str,
    max_results: int = 50,
    published_after: str = None,
    published_before: str = None,
    order: str = "relevance",
    region_code: str = None,
    api_key: str = None,
) -> pd.DataFrame:
    """
    Search YouTube for videos matching a keyword query.

    Parameters
    ----------
    query           : Keyword search string.
    max_results     : Max videos (≤50 per call; costs 100 quota units/call).
    published_after : RFC 3339 lower bound, e.g. '2023-01-01T00:00:00Z'.
    published_before: RFC 3339 upper bound.
    order           : 'relevance' | 'date' | 'viewCount' | 'rating'.
    region_code     : ISO 3166-1 alpha-2, e.g. 'AU' for Australia.
    api_key         : Optional key override.

    Returns
    -------
    pd.DataFrame : video_id, title, channel_id, channel_title,
                   published_at, description, thumbnail_url, query_used
    """
    youtube = get_youtube_client(api_key)
    params = {
        "part":              "snippet",
        "q":                 query,
        "type":              "video",
        "maxResults":        min(max_results, 50),
        "relevanceLanguage": "en",
        "order":             order,
        "safeSearch":        "none",
    }
    if published_after:  params["publishedAfter"]  = published_after
    if published_before: params["publishedBefore"] = published_before
    if region_code:      params["regionCode"]      = region_code

    try:
        response = youtube.search().list(**params).execute()
    except HttpError as e:
        logger.error(f"search.list failed for '{query}': {e}")
        return pd.DataFrame()

    records = []
    for item in response.get("items", []):
        vid_id = item.get("id", {}).get("videoId")
        if not vid_id:
            continue
        snippet = item.get("snippet", {})
        records.append({
            "video_id":      vid_id,
            "title":         snippet.get("title", ""),
            "channel_id":    snippet.get("channelId", ""),
            "channel_title": snippet.get("channelTitle", ""),
            "published_at":  snippet.get("publishedAt", ""),
            "description":   snippet.get("description", "")[:500],
            "thumbnail_url": (snippet.get("thumbnails", {})
                                     .get("high", {}).get("url", "")),
            "query_used":    query,
        })

    logger.info(f"search_videos('{query}'): {len(records)} results")
    return pd.DataFrame(records)


# ── Video Statistics ───────────────────────────────────────────────────────────

def get_video_stats(video_ids: list, api_key: str = None) -> pd.DataFrame:
    """
    Fetch statistics and content details for a list of video IDs.
    Batches into groups of 50 (1 quota unit per batch).

    Returns
    -------
    pd.DataFrame : video_id, view_count, like_count, comment_count,
                   duration, category_id, tags, default_language
    """
    youtube = get_youtube_client(api_key)
    records = []

    for i in range(0, len(video_ids), 50):
        batch = video_ids[i:i+50]
        try:
            response = youtube.videos().list(
                part="statistics,snippet,contentDetails",
                id=",".join(batch),
            ).execute()
        except HttpError as e:
            logger.warning(f"videos.list failed for batch {i//50}: {e}")
            continue

        for item in response.get("items", []):
            stats   = item.get("statistics", {})
            snippet = item.get("snippet", {})
            content = item.get("contentDetails", {})
            records.append({
                "video_id":         item["id"],
                "view_count":       int(stats.get("viewCount",    0)),
                "like_count":       int(stats.get("likeCount",    0)),
                "comment_count":    int(stats.get("commentCount", 0)),
                "duration":         content.get("duration", ""),
                "category_id":      snippet.get("categoryId", ""),
                "tags":             "|".join(snippet.get("tags", [])),
                "default_language": snippet.get(
                    "defaultAudioLanguage",
                    snippet.get("defaultLanguage", "")
                ),
            })

    logger.info(f"get_video_stats: fetched for {len(records)} videos")
    return pd.DataFrame(records)


# ── Comment Collection ─────────────────────────────────────────────────────────

def get_comments(
    video_id: str,
    max_comments: int = 200,
    order: str = "relevance",
    api_key: str = None,
) -> pd.DataFrame:
    """
    Fetch top-level comments for a single YouTube video.

    Parameters
    ----------
    video_id     : YouTube video ID.
    max_comments : Max top-level comments to retrieve.
    order        : 'relevance' | 'time'
    api_key      : Optional key override.

    Returns
    -------
    pd.DataFrame : comment_id, video_id, author_id, author_name,
                   text, like_count, reply_count, published_at, updated_at

    Notes
    -----
    - Videos with disabled comments return empty DataFrame silently.
    - Pagination handled via nextPageToken.
    - Each call fetches up to 100 comments (1 quota unit/call).
    """
    youtube    = get_youtube_client(api_key)
    comments   = []
    page_token = None

    while len(comments) < max_comments:
        params = {
            "part":       "snippet",
            "videoId":    video_id,
            "maxResults": min(100, max_comments - len(comments)),
            "textFormat": "plainText",
            "order":      order,
        }
        if page_token:
            params["pageToken"] = page_token

        try:
            response = youtube.commentThreads().list(**params).execute()
        except HttpError as e:
            if any(kw in str(e) for kw in
                   ["commentsDisabled", "403", "disabled", "forbidden"]):
                pass  # Silently skip
            else:
                logger.warning(f"commentThreads.list failed for {video_id}: {e}")
            break

        for item in response.get("items", []):
            top = item["snippet"]["topLevelComment"]["snippet"]
            comments.append({
                "comment_id":  item["id"],
                "video_id":    video_id,
                "author_id":   (top.get("authorChannelId", {})
                                   .get("value", "anonymous")),
                "author_name": top.get("authorDisplayName", ""),
                "text":        top.get("textDisplay", ""),
                "like_count":  int(top.get("likeCount",  0)),
                "reply_count": int(item["snippet"].get("totalReplyCount", 0)),
                "published_at":top.get("publishedAt", ""),
                "updated_at":  top.get("updatedAt", ""),
            })

        page_token = response.get("nextPageToken")
        if not page_token:
            break

    return pd.DataFrame(comments)


# ── Channel Info ───────────────────────────────────────────────────────────────

def get_channel_info(channel_ids: list, api_key: str = None) -> pd.DataFrame:
    """
    Retrieve subscriber counts and basic metadata for channel IDs.
    Enriches channel network nodes with real-world attributes.

    Returns
    -------
    pd.DataFrame : channel_id, channel_name, subscriber_count,
                   video_count, total_view_count, country, description
    """
    youtube = get_youtube_client(api_key)
    records = []

    for i in range(0, len(channel_ids), 50):
        batch = channel_ids[i:i+50]
        try:
            response = youtube.channels().list(
                part="snippet,statistics",
                id=",".join(batch),
            ).execute()
        except HttpError as e:
            logger.warning(f"channels.list failed: {e}")
            continue

        for item in response.get("items", []):
            stats   = item.get("statistics", {})
            snippet = item.get("snippet", {})
            records.append({
                "channel_id":       item["id"],
                "channel_name":     snippet.get("title", ""),
                "subscriber_count": int(stats.get("subscriberCount", 0)),
                "video_count":      int(stats.get("videoCount", 0)),
                "total_view_count": int(stats.get("viewCount", 0)),
                "country":          snippet.get("country", ""),
                "description":      snippet.get("description", "")[:300],
            })

    return pd.DataFrame(records)


# ── End-to-End Pipeline ────────────────────────────────────────────────────────

def collect_full_dataset(
    queries: list,
    max_videos_per_query: int = 30,
    max_comments_per_video: int = 150,
    published_after: str = "2023-01-01T00:00:00Z",
    published_before: str = None,
    order: str = "relevance",
    region_code: str = None,
    rate_limit_sleep: float = 0.4,
    api_key: str = None,
) -> dict:
    """
    Full collection pipeline: search → stats → comments → channel info.

    Returns
    -------
    dict with keys:
        'videos'   → pd.DataFrame (unique videos + stats)
        'comments' → pd.DataFrame (all comments)
        'channels' → pd.DataFrame (channel metadata)
    """
    all_videos   = []
    all_comments = []

    for q_idx, query in enumerate(queries):
        logger.info(f"\n[{q_idx+1}/{len(queries)}] Query: '{query}'")

        # Step 1: Search
        df_search = search_videos(
            query, max_results=max_videos_per_query,
            published_after=published_after,
            published_before=published_before,
            order=order, region_code=region_code, api_key=api_key,
        )
        if df_search.empty:
            logger.warning("  No videos found — skipping.")
            continue

        # Step 2: Stats
        video_ids = df_search["video_id"].tolist()
        df_stats  = get_video_stats(video_ids, api_key=api_key)
        df_videos = df_search.merge(df_stats, on="video_id", how="left")
        all_videos.append(df_videos)

        # Step 3: Comments
        query_comments = []
        for vid_id in tqdm(video_ids, desc=f"  Comments [{query[:30]}]"):
            cdf = get_comments(vid_id, max_comments=max_comments_per_video,
                               api_key=api_key)
            if not cdf.empty:
                query_comments.append(cdf)
            time.sleep(rate_limit_sleep)

        if query_comments:
            q_cdf = pd.concat(query_comments, ignore_index=True)
            all_comments.append(q_cdf)
            logger.info(f"  Comments collected: {len(q_cdf)}")

    if not all_videos:
        raise RuntimeError("No data collected. Check API key and queries.")

    videos_df   = (pd.concat(all_videos, ignore_index=True)
                     .drop_duplicates("video_id"))
    comments_df = (pd.concat(all_comments, ignore_index=True)
                     .drop_duplicates("comment_id")
                   if all_comments else pd.DataFrame())

    # Step 4: Channel info
    unique_channels = videos_df["channel_id"].dropna().unique().tolist()
    channels_df = get_channel_info(unique_channels, api_key=api_key)

    logger.info(f"\n✅ Done. Videos={len(videos_df)}, "
                f"Comments={len(comments_df)}, "
                f"Channels={len(channels_df)}")

    return {"videos": videos_df, "comments": comments_df, "channels": channels_df}


# ── Quota Estimator ────────────────────────────────────────────────────────────

def estimate_quota(
    num_queries: int,
    videos_per_query: int,
    comments_per_video: int,
) -> dict:
    """
    Estimate API quota usage before collection.
    Daily limit: 10,000 units.
    """
    search_cost  = num_queries * 100
    stats_cost   = num_queries * max(1, videos_per_query // 50)
    comment_cost = num_queries * videos_per_query * max(1, comments_per_video // 100)
    total        = search_cost + stats_cost + comment_cost
    return {
        "search_units":  search_cost,
        "stats_units":   stats_cost,
        "comment_units": comment_cost,
        "total_units":   total,
        "daily_limit":   10_000,
        "within_limit":  total <= 10_000,
        "pct_of_daily":  round(total / 10_000 * 100, 1),
    }