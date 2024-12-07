import seaborn as sns
import matplotlib.pyplot as plt
import os

class VisualizationProcessor:
    def __init__(self, static_dir='static/'):
        self.static_dir = static_dir
        os.makedirs(self.static_dir, exist_ok=True)

    def visualize_engagement_rate(self, videos_df):
        plt.figure(figsize=(10, 6))
        sns.barplot(x='videoId', y='engagement_rate', data=videos_df, palette='viridis')
        plt.title("Engagement Rate per Video")
        plt.xlabel("Video ID")
        plt.ylabel("Engagement Rate (%)")
        output_path = os.path.join(self.static_dir, "engagement_statistics.png")
        plt.savefig(output_path)
        plt.close()
        return output_path

    def visualize_sentiment_analysis(self, videos_df, comments_df):
        # Video sentiment distribution
        plt.figure(figsize=(10, 6))
        sns.histplot(videos_df['sentiment'], kde=True, color='blue', bins=30)
        plt.title('Sentiment Distribution of Video Descriptions')
        plt.xlabel('Sentiment Score')
        plt.ylabel('Frequency')
        video_sentiment_path = os.path.join(self.static_dir, "video_sentiment_distribution.png")
        plt.savefig(video_sentiment_path)
        plt.close()

        # Comments sentiment distribution
        plt.figure(figsize=(10, 6))
        sns.histplot(comments_df['sentiment'], kde=True, color='green', bins=30)
        plt.title('Sentiment Distribution of Comments')
        plt.xlabel('Sentiment Score')
        plt.ylabel('Frequency')
        comments_sentiment_path = os.path.join(self.static_dir, "comments_sentiment_distribution.png")
        plt.savefig(comments_sentiment_path)
        plt.close()

        return video_sentiment_path, comments_sentiment_path

    def visualize_topics(self, videos_df):
        topics = [f"topic_{i}" for i in range(5)]  # Adjust based on actual topic columns
        topic_data = videos_df[topics]
        topic_data_mean = topic_data.mean()

        plt.figure(figsize=(10, 6))
        sns.barplot(x=topic_data_mean.index, y=topic_data_mean.values, palette='muted')
        plt.title("Average Topic Distribution per Video")
        plt.xlabel("Topic")
        plt.ylabel("Average Distribution")
        topic_distribution_path = os.path.join(self.static_dir, "topic_distribution.png")
        plt.savefig(topic_distribution_path)
        plt.close()
        return topic_distribution_path


