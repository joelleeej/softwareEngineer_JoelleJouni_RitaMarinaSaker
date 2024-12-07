import pandas as pd
import numpy as np
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import json
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
matplotlib.use('Agg')  # Use a non-GUI backend

# Function to load data from the JSON file
def load_data(channel_id):
    filename = f'channel_data_{channel_id}.json'
    with open(filename, 'r') as f:
        data = json.load(f)
    return data

# Function to process and clean data into dataframes
def process_data(data):
    # Extract channel stats
    channel_data = data.get('channel', {})
    channel_df = pd.DataFrame([channel_data])

    # Initialize list for comments data
    comments_data = []

    # Check if videoComments exist in the loaded JSON
    if 'videoComments' in data:
        video_comments = data['videoComments']
        
        # Loop through video comments to structure them
        for video_id, comments in video_comments.items():
            for comment in comments.get('comments', []):  # Ensure 'comments' is accessed correctly
                comments_data.append({'videoId': video_id, 'Comment': comment})

    # Create DataFrame for comments
    comments_df = pd.DataFrame(comments_data)

    # Extract top videos stats
    top_videos_data = data.get('topVideos', [])
    videos_df = pd.DataFrame(top_videos_data)

    return channel_df, comments_df, videos_df

# Function to clean and prepare video data
def clean_video_data(videos_df):
    # Convert relevant columns to numeric, forcing errors to NaN (useful if there are any non-numeric values)
    videos_df['viewCount'] = pd.to_numeric(videos_df['viewCount'], errors='coerce')
    videos_df['likeCount'] = pd.to_numeric(videos_df['likeCount'], errors='coerce')
    videos_df['commentCount'] = pd.to_numeric(videos_df['commentCount'], errors='coerce')

    # Fill missing values (if any) with zeros or suitable default values
    videos_df.fillna({
        'viewCount': 0,
        'likeCount': 0,
        'commentCount': 0
    }, inplace=True)

    # Add a 'total_engagement' column combining views, likes, and comments
    videos_df['total_engagement'] = videos_df['viewCount'] + videos_df['likeCount'] + videos_df['commentCount']

    # Calculate Engagement Rate (Total Engagement / Total Views)
    videos_df['engagement_rate'] = videos_df['total_engagement'] / videos_df['viewCount']

    # Calculate Like-to-View Ratio
    videos_df['like_to_view_ratio'] = videos_df['likeCount'] / videos_df['viewCount']

    return videos_df

# Function to apply sentiment analysis and topic modeling to the video data
def apply_nlp(videos_df, comments_df):
    # Sentiment Analysis with VADER
    sia = SentimentIntensityAnalyzer()

    # Apply sentiment analysis on video descriptions
    videos_df['sentiment'] = videos_df['description'].apply(lambda x: sia.polarity_scores(x)['compound'])

    # Apply sentiment analysis on comments
    comments_df['sentiment'] = comments_df['Comment'].apply(lambda x: sia.polarity_scores(x)['compound'])

    # Topic Modeling with LDA (Latent Dirichlet Allocation) for video descriptions
    vectorizer = CountVectorizer(stop_words='english', max_features=1000)
    dtm = vectorizer.fit_transform(videos_df['description'].fillna(''))

    lda_model = LatentDirichletAllocation(n_components=5, random_state=42)
    lda_topics = lda_model.fit_transform(dtm)

    # Add the dominant topic for each video
    videos_df['dominant_topic'] = lda_topics.argmax(axis=1)

    # Visualizations
    plot_sentiment_distribution(comments_df)
    plot_topic_distribution(videos_df)
    plot_engagement_statistics(videos_df)
    plot_top_keywords(videos_df['description'], 'Top Keywords in Video Descriptions', 'static/keywords_descriptions.png')
    plot_top_keywords(comments_df['Comment'], 'Top Keywords in Comments', 'static/keywords_comments.png')

    return videos_df, comments_df

# Visualization of sentiment distribution in comments
def plot_sentiment_distribution(comments_df):
    plt.figure(figsize=(8, 6))
    sns.histplot(comments_df['sentiment'], kde=True, bins=20, color='blue')
    plt.title('Sentiment Distribution of Comments')
    plt.xlabel('Sentiment Score')
    plt.ylabel('Frequency')
    plt.savefig('static/sentiment_distribution.png')
    plt.close()

# Visualization of topic distribution in video descriptions
def plot_topic_distribution(videos_df):
    plt.figure(figsize=(8, 6))
    sns.countplot(x='dominant_topic', data=videos_df, palette='Set2')
    plt.title('Topic Distribution in Video Descriptions')
    plt.xlabel('Dominant Topic')
    plt.ylabel('Frequency')
    plt.savefig('static/topic_distribution.png')
    plt.close()

# Visualization of video engagement statistics
def plot_engagement_statistics(videos_df):
    plt.figure(figsize=(10, 6))
    sns.barplot(x='title', y='engagement_rate', data=videos_df.sort_values(by='engagement_rate', ascending=False).head(10))
    plt.title('Top 10 Videos by Engagement Rate')
    plt.xticks(rotation=45, ha='right')
    plt.xlabel('Video Title')
    plt.ylabel('Engagement Rate')
    plt.tight_layout()
    plt.savefig('static/engagement_statistics.png')
    plt.close()

# New function to extract and visualize keywords from text data
def plot_top_keywords(text_series, title, output_path, num_keywords=10):
    vectorizer = CountVectorizer(stop_words='english', max_features=1000)
    dtm = vectorizer.fit_transform(text_series.fillna(''))
    word_counts = np.asarray(dtm.sum(axis=0)).flatten()
    keywords_freq = pd.DataFrame({'keyword': vectorizer.get_feature_names_out(), 'count': word_counts})
    top_keywords = keywords_freq.sort_values(by='count', ascending=False).head(num_keywords)

    plt.figure(figsize=(10, 6))
    sns.barplot(x='count', y='keyword', data=top_keywords, palette='viridis')
    plt.title(title)
    plt.xlabel('Frequency')
    plt.ylabel('Keyword')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

# Function to combine and extract features from both the videos and comments data
def extract_features(channel_df, videos_df, comments_df):
    # Aggregate comment sentiment scores by VideoId
    comment_sentiment_avg = comments_df.groupby('videoId')['sentiment'].mean().reset_index()
    videos_df = pd.merge(videos_df, comment_sentiment_avg, how='left', on='videoId')

    # Feature Extraction: Combine Channel Stats with Video Stats
    # Remove 'videoId' from channel_df since it doesn't exist in channel_df
    channel_engagement = channel_df[['viewCount', 'subscriberCount', 'videoCount']]  # Example features
    # If necessary, repeat the same channel-level stats for each video (based on how channel_df is structured)
    channel_engagement = pd.concat([channel_engagement] * len(videos_df), ignore_index=True)

    # Merge with videos_df (without 'videoId' in channel_df)
    combined_df = pd.merge(videos_df, channel_engagement, how='left', left_index=True, right_index=True)

    # Additional example feature: Average Sentiment Score per Channel (optional)
    average_channel_sentiment = comments_df['sentiment'].mean()  # Sentiment across all comments
    channel_df['average_sentiment'] = average_channel_sentiment

    return combined_df, channel_df





