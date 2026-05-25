COSC2671 Social Media and Network Analytics
Assignment 2

**Sudden Fuel Price Hike Discussions on YouTube: A Combined Network and NLP Analysis**

## **Project Overview**

This project explores how people discuss and react to sudden fuel price hikes on YouTube. The analysis focuses on understanding how YouTube users engage with fuel price related video content, which channels and commenters are most influential, what topics are commonly discussed, and how the public emotionally responds to price increase events.

YouTube was selected as the main data source because it contains large amounts of public video content and comment data related to news and economic events, with visible engagement relationships between users and channels. By combining Social Network Analysis (SNA) and Natural Language Processing (NLP), the project provides insight into both user behaviour and discussion content.

The project investigates:

* how YouTube users interact during fuel price hike discussions,
* which users and channels are most influential in conversations,
* how discussion communities are formed around this topic,
* and what sentiments and themes appear most frequently in comments.

# **Research Question**

How do YouTube users interact and express sentiment in discussions related to sudden fuel price hike events, and who are the most influential voices shaping public discourse?

# **Dataset Information**

The dataset was collected from YouTube using the YouTube Data API v3 via Python scripts.

To focus specifically on fuel price hike discussions, videos were searched using keywords associated with fuel costs, price increases, and energy pricing.

Some keywords used include:

* fuel price hike
* petrol price increase
* gas price rise
* oil price surge
* fuel cost increase
* petrol pump price

The final dataset contained:

* YouTube videos related to fuel price hike discussions
* YouTube comments collected from those videos

The collected data included:

* video titles
* video descriptions
* channel IDs and names
* comment text
* commenter channel IDs (author_id)
* comment timestamps
* comment like counts
* reply counts

# **Project Structure**

```
Fuel Price Hike/
│
├── Data/
│   ├── raw_videos.csv
│   ├── raw_comments.csv
│   ├── videos_processed.csv
│   ├── comments_processed.csv
│   ├── comments_with_sentiment_topics.csv
│   ├── network_metrics.csv
│   └── channel_indegree.csv
│
├── Utils/
│   ├── __init__.py
│   ├── youtube_collector.py
│   ├── network_builder.py
│   └── nlp_utils.py
│
├── notebooks/
│   ├── 01_data_collection.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_network_analysis.ipynb
│   ├── 04_nlp_analysis.ipynb
│   └── 05_visualisation.ipynb
│
├── outputs/
│   ├── 02_cocomment_network.png
│   ├── 03_degree_distribution.png
│   ├── 04_centrality_scatter.png
│   ├── 05_sentiment_distribution.png
│   ├── 06_sentiment_over_time.png
│   ├── 07_influence_vs_sentiment.png
│   ├── 08_tfidf_by_sentiment.png
│   ├── 09_topic_perplexity.png
│   ├── 10_topic_top_words.png
│   ├── 11_topic_wordclouds.png
│   ├── 12_topic_distribution.png
│   ├── 13_topic_sentiment_heatmap.png
│   ├── 14_summary_dashboard.png
│   ├── 15_topic_sentiment_heatmap.png
│   ├── 16_centrality_scatter.png
│   ├── 17_video_sentiment_scatter.png
│   ├── 18_community_sizes.png
│   └── 19_community_sentiment.png
│
├── README.md
└── requirements.txt
```

# **Technologies Used**

The project was developed using Python and several data science libraries.

## **Main Libraries**

* pandas
* numpy
* google-api-python-client
* python-dotenv
* networkx
* python-louvain
* matplotlib
* seaborn
* nltk
* vaderSentiment
* scikit-learn
* wordcloud

# **Network Analysis**

The network analysis section focused on understanding how YouTube users and channels interact with one another during fuel price hike discussions.

Two networks were constructed:

* A **User Co-Comment Network** (undirected, weighted) using shared video commenting relationships between users
* A **Channel Interaction Network** (directed, weighted) using engagement flows from commenter channels to content creator channels

## **Network Statistics**

* 10,624 nodes (unique YouTube channels)
* 10,762 edges (interaction relationships)
* 11,380 total cross-channel interactions
* Communities detected using the Louvain algorithm

## **Analysis Performed**

* Degree Centrality
* Betweenness Centrality
* PageRank (influence ranking)
* Community Detection (Louvain)
* Network Visualisation

The analysis identified several highly influential commenters who engaged frequently across multiple fuel price hike videos. It also revealed that discussions were clustered into distinct communities, likely grouped around different news channels or regional audiences.

# **NLP Analysis**

The NLP section focused on analysing the textual content of YouTube comments.

This part of the project included:

* text preprocessing and cleaning
* VADER sentiment analysis
* LDA topic modelling (sklearn)
* TF-IDF keyword analysis
* topic-sentiment cross analysis

The NLP analysis helped identify:

* dominant public sentiments toward fuel price increases,
* common discussion topics such as government policy, daily impact, and alternatives,
* and vocabulary differences between positive, negative, and neutral commenters.

# **How to Run the Project**

## **Step 1 — Set Up Virtual Environment**

### **Windows**

```
python -m venv venv
venv\Scripts\activate
```

### **Mac / Linux**

```
python3 -m venv venv
source venv/bin/activate
```

## **Step 2 — Install Required Libraries**

```
pip install -r requirements.txt
```

## **Step 3 — Add Your YouTube API Key**

Copy `.env.example` to a new file named `.env` and paste your YouTube Data API v3 key:

```
YOUTUBE_API_KEY=your_actual_api_key_here
```

> Do NOT commit your `.env` file. It contains private credentials.

## **Step 4 — Collect YouTube Videos and Comments**

Open and run:

```
notebooks/01_data_collection.ipynb
```

## **Step 5 — Preprocess the Data**

Open and run:

```
notebooks/02_preprocessing.ipynb
```

## **Step 6 — Run Network Analysis**

Open and run:

```
notebooks/03_network_analysis.ipynb
```

## **Step 7 — Run NLP Analysis**

Open and run:

```
notebooks/04_nlp_analysis.ipynb
```

## **Step 8 — Generate Summary Visualisations**

Open and run:

```
notebooks/05_visualisation.ipynb
```

> Run all notebooks top to bottom, in order. Each notebook depends on output files from the previous one.

# **Output Files**

The project generates:

* user co-comment network graphs,
* channel interaction network graphs,
* influential user and channel charts,
* sentiment distribution visualisations,
* topic modelling results and word clouds,
* and integrated summary dashboards.

Generated outputs are stored inside:

* outputs/

# **Key Findings**

The analysis showed that YouTube discussions about sudden fuel price hikes are largely negative in sentiment, with users expressing frustration, concern, and criticism toward governments and oil companies.

Several YouTube channels attracted disproportionately high levels of engagement, with the top channel receiving over 1,300 cross-channel interactions — indicating strong influence in shaping the public narrative.

The topic modelling revealed distinct discussion clusters including government policy criticism, daily life impact, calls for alternative energy, and international oil market commentary.

The network analysis showed that influential commenters tended to have slightly more negative average sentiment scores than the broader community, suggesting that emotionally charged users drive a higher share of discussion activity.

# **Limitations**

Some limitations of this project include:

* YouTube API daily quota limits (10,000 units) may restrict collection to a subset of available videos,
* keyword filtering may exclude relevant discussions that use informal language or regional terms,
* VADER sentiment analysis may misclassify sarcastic or ironic comments,
* and the dataset reflects a specific collection period which may not capture long-term trends.

Additionally, YouTube commenters may not fully represent the broader public opinion on fuel prices.

# **Authors**

Postgraduate Group — COSC 2671

* Member 1 — Data Collection, Network Analysis, Community Detection, Visualisations
* Member 2 — NLP Analysis, Sentiment Analysis, Topic Modelling, Reporting
* Member 3 — Preprocessing, Integration, Report Writing, Presentation
