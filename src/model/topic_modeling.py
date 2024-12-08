import joblib
import re
import matplotlib.pyplot as plt
import seaborn as sns
import os
import mlflow
import mlflow.sklearn  # Import if you're logging sklearn models


class TopicModelingProcessor:
    def __init__(self, model_dir="channel_analysis_models/", static_dir="static/"):
        self.model_dir = model_dir
        self.static_dir = static_dir

        # Ensure directories exist
        os.makedirs(self.static_dir, exist_ok=True)

        # Load the saved model and TF-IDF vectorizer
        self.loaded_model = joblib.load(os.path.join(self.model_dir, "model.pkl"))
        self.tfidf_vectorizer = joblib.load(
            os.path.join(self.model_dir, "tfidf_vectorizer.pkl")
        )

    @staticmethod
    def preprocess_text(text):
        text = text.lower()
        text = re.sub(r"[^\w\s]", "", text)  # Remove punctuation
        text = re.sub(r"\d+", "", text)  # Remove digits
        return text

    def preprocess_data(self, videos_df, comments_df):
        videos_df["Cleaned_Title"] = videos_df["title"].apply(self.preprocess_text)
        videos_df["Cleaned_Description"] = videos_df["description"].apply(
            self.preprocess_text
        )
        comments_df["Cleaned_Comment"] = comments_df["Comment"].apply(
            self.preprocess_text
        )
        return videos_df, comments_df

    def apply_model_predictions(self, videos_df, comments_df):
        video_columns = ["Cleaned_Title", "Cleaned_Description"]
        for column in video_columns:
            X_tfidf = self.tfidf_vectorizer.transform(videos_df[column])
            predictions = self.loaded_model.predict(X_tfidf)
            videos_df[f"{column}_Dominant_Topic"] = predictions

        X_tfidf_comments = self.tfidf_vectorizer.transform(
            comments_df["Cleaned_Comment"]
        )
        comments_df["Comment_Dominant_Topic"] = self.loaded_model.predict(
            X_tfidf_comments
        )

        return videos_df, comments_df

    def visualize_topics(self, df, column_name, title, filename):
        plt.figure(figsize=(12, 6))
        sns.countplot(x=column_name, data=df)
        plt.title(title)
        plt.xlabel("Dominant Topic")
        plt.ylabel("Frequency")
        plt.xticks(rotation=45)
        output_path = os.path.join(self.static_dir, f"{filename}.png")
        plt.savefig(output_path, format="png")
        plt.close()
        return output_path

    def generate_summary(self, videos_df, comments_df):
        topic_names = {0: "World", 1: "Sports", 2: "Business", 3: "Sci/Tech"}

        video_title_dominant_topic = topic_names[
            videos_df["Cleaned_Title_Dominant_Topic"].mode()[0]
        ]
        video_description_dominant_topic = topic_names[
            videos_df["Cleaned_Description_Dominant_Topic"].mode()[0]
        ]
        comment_dominant_topic = topic_names[
            comments_df["Comment_Dominant_Topic"].mode()[0]
        ]

        video_summary = f"Dominant topics in video titles: {video_title_dominant_topic}"
        video_summary += (
            f"Dominant topics in video descriptions: {video_description_dominant_topic}"
        )
        comment_summary = f"Dominant topic in comments: {comment_dominant_topic}"

        summary = (
            f"The dominant topic in these columns are:{video_summary}{comment_summary}"
        )
        dominant_topics = {
            "video_title": video_title_dominant_topic,
            "video_description": video_description_dominant_topic,
            "comment": comment_dominant_topic,
        }

        return summary, dominant_topics

    def process_and_visualize(self, videos_df, comments_df):
        # Preprocess data
        videos_df, comments_df = self.preprocess_data(videos_df, comments_df)

        # Apply model predictions
        videos_df, comments_df = self.apply_model_predictions(videos_df, comments_df)

        # Visualize dominant topics
        video_title_img = self.visualize_topics(
            videos_df, "Cleaned_Title_Dominant_Topic", "Video Titles", "video_title_img"
        )
        video_description_img = self.visualize_topics(
            videos_df,
            "Cleaned_Description_Dominant_Topic",
            "Video Descriptions",
            "video_description_img",
        )
        comment_img = self.visualize_topics(
            comments_df, "Comment_Dominant_Topic", "Comments", "comment_img"
        )

        # Generate summary
        summary, dominant_topics = self.generate_summary(videos_df, comments_df)

        # Log metrics to MLflow
        mlflow.log_metric(
            "video_title_topic", len(videos_df["Cleaned_Title_Dominant_Topic"])
        )
        mlflow.log_metric("comment_topic", len(comments_df["Comment_Dominant_Topic"]))

        return (
            summary,
            video_title_img,
            video_description_img,
            comment_img,
            dominant_topics,
        )
