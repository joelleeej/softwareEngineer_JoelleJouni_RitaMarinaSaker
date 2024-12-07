import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Function to visualize engagement rate
def visualize_engagement_rate(videos_df):
    # Create a bar plot for engagement rate per video
    fig = px.bar(videos_df, x='videoId', y='engagement_rate',
                 title="Engagement Rate per Video", labels={'videoId': 'Video ID', 'engagement_rate': 'Engagement Rate (%)'})
    return fig

# Function to visualize sentiment analysis
def visualize_sentiment_analysis(videos_df, comments_df):
    # Visualize the sentiment of videos using a distribution plot
    plt.figure(figsize=(10, 6))
    sns.histplot(videos_df['sentiment'], kde=True, color='blue', bins=30)
    plt.title('Sentiment Distribution of Video Descriptions')
    plt.xlabel('Sentiment Score')
    plt.ylabel('Frequency')
    sentiment_fig = plt.gcf()  # Get the current figure
    plt.close()  # Close to prevent it from displaying automatically in Jupyter
    
    # Visualize sentiment for comments
    plt.figure(figsize=(10, 6))
    sns.histplot(comments_df['sentiment'], kde=True, color='green', bins=30)
    plt.title('Sentiment Distribution of Comments')
    plt.xlabel('Sentiment Score')
    plt.ylabel('Frequency')
    comments_sentiment_fig = plt.gcf()
    plt.close()

    return sentiment_fig, comments_sentiment_fig

# Function to visualize topic distribution from LDA
def visualize_topics(videos_df):
    # Plot the topic distribution per video (if it's a good fit)
    topics = [f"Topic {i}" for i in range(videos_df.shape[1] - 6)]  # Assuming 6 initial columns before topic columns
    topic_data = videos_df[topics]
    
    topic_data_mean = topic_data.mean()
    fig = px.bar(topic_data_mean, x=topic_data_mean.index, y=topic_data_mean.values,
                 title="Average Topic Distribution per Video", labels={'x': 'Topic', 'y': 'Average Distribution'})
    return fig
