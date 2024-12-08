import pandas as pd
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import json
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib

matplotlib.use("Agg")  # Use a non-GUI backend


class DataCleaner:
    def __init__(self, channel_id=None):
        self.channel_id = channel_id

    def load_json(self):
        filename = f"channel_data_{self.channel_id}.json"
        with open(filename, "r") as f:
            data = json.load(f)
        return data

    def process_data(self, data):
        channel_data = data.get("channel", {})
        channel_df = pd.DataFrame([channel_data])

        comments_data = []
        if "videoComments" in data:
            video_comments = data["videoComments"]
            for video_id, comments in video_comments.items():
                for comment in comments.get("comments", []):
                    comments_data.append({"videoId": video_id, "Comment": comment})

        comments_df = pd.DataFrame(comments_data)
        top_videos_data = data.get("topVideos", [])
        videos_df = pd.DataFrame(top_videos_data)
        return channel_df, comments_df, videos_df

    def clean_video_data(self, videos_df):
        videos_df["viewCount"] = pd.to_numeric(videos_df["viewCount"], errors="coerce")
        videos_df["likeCount"] = pd.to_numeric(videos_df["likeCount"], errors="coerce")
        videos_df["commentCount"] = pd.to_numeric(
            videos_df["commentCount"], errors="coerce"
        )
        videos_df.fillna(
            {"viewCount": 0, "likeCount": 0, "commentCount": 0}, inplace=True
        )
        videos_df["total_engagement"] = (
            videos_df["viewCount"] + videos_df["likeCount"] + videos_df["commentCount"]
        )
        videos_df["engagement_rate"] = videos_df["total_engagement"] / videos_df[
            "viewCount"
        ].replace(0, 1)
        videos_df["like_to_view_ratio"] = videos_df["likeCount"] / videos_df[
            "viewCount"
        ].replace(0, 1)
        return videos_df

    def apply_nlp(self, videos_df, comments_df):
        sia = SentimentIntensityAnalyzer()
        videos_df["sentiment"] = videos_df["description"].apply(
            lambda x: sia.polarity_scores(x)["compound"]
        )
        comments_df["sentiment"] = comments_df["Comment"].apply(
            lambda x: sia.polarity_scores(x)["compound"]
        )
        vectorizer = CountVectorizer(stop_words="english", max_features=1000)
        dtm = vectorizer.fit_transform(videos_df["description"].fillna(""))
        lda_model = LatentDirichletAllocation(n_components=5, random_state=42)
        lda_topics = lda_model.fit_transform(dtm)
        videos_df["dominant_topic"] = lda_topics.argmax(axis=1)
        return videos_df, comments_df

    def plot_sentiment_distribution(self, comments_df):
        plt.figure(figsize=(8, 6))
        sns.histplot(comments_df["sentiment"], kde=True, bins=20, color="blue")
        plt.title("Sentiment Distribution of Comments")
        plt.xlabel("Sentiment Score")
        plt.ylabel("Frequency")
        plt.savefig("static/sentiment_distribution.png")
        plt.close()

    def plot_topic_distribution(self, videos_df):
        plt.figure(figsize=(8, 6))
        sns.countplot(
            x="dominant_topic",
            hue="dominant_topic",
            data=videos_df,
            palette="Set2",
            legend=False,
        )  # Fixed warning
        plt.title("Topic Distribution in Video Descriptions")
        plt.xlabel("Dominant Topic")
        plt.ylabel("Frequency")
        plt.savefig("static/topic_distribution.png")
        plt.close()

    def extract_features(self, channel_df, videos_df, comments_df):
        comment_sentiment_avg = (
            comments_df.groupby("videoId")["sentiment"].mean().reset_index()
        )
        videos_df = pd.merge(videos_df, comment_sentiment_avg, how="left", on="videoId")
        channel_engagement = channel_df[["viewCount", "subscriberCount", "videoCount"]]
        channel_engagement = pd.concat(
            [channel_engagement] * len(videos_df), ignore_index=True
        )
        combined_df = pd.merge(
            videos_df, channel_engagement, how="left", left_index=True, right_index=True
        )
        average_channel_sentiment = comments_df["sentiment"].mean()
        channel_df["average_sentiment"] = average_channel_sentiment
        return combined_df, channel_df
