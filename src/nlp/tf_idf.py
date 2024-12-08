import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import matplotlib
import os
matplotlib.use('Agg')

class TFIDFProcessor:
    def __init__(self, static_dir='static/', max_features=20):
        self.static_dir = static_dir
        self.max_features = max_features
        self.tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_features=max_features)

        # Ensure the static directory exists
        os.makedirs(self.static_dir, exist_ok=True)

    @staticmethod
    def preprocess_text(text):
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
        text = re.sub(r'\d+', '', text)  # Remove digits
        return text

    def clean_data(self, videos_df, comments_df):
        videos_df['Cleaned_Description'] = videos_df['description'].apply(self.preprocess_text)
        comments_df['Cleaned_Comment'] = comments_df['Comment'].apply(self.preprocess_text)
        return videos_df, comments_df

    def apply_tfidf(self, videos_df, comments_df):
        combined_text = list(videos_df['Cleaned_Description']) + list(comments_df['Cleaned_Comment'])
        tfidf_matrix = self.tfidf_vectorizer.fit_transform(combined_text)
        feature_names = self.tfidf_vectorizer.get_feature_names_out()
        dense_matrix = tfidf_matrix.todense()
        tfidf_df = pd.DataFrame(dense_matrix, columns=feature_names)
        return tfidf_df, feature_names, tfidf_matrix

    def visualize_tfidf_keywords(self, tfidf_df, num_words=10, output_filename='tfidf_keywords_descriptions.png'):
        sum_tfidf_scores = tfidf_df.sum(axis=0).sort_values(ascending=False)
        top_words = sum_tfidf_scores.iloc[:num_words]

        # Visualization
        plt.figure(figsize=(10, 6))
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(sum_tfidf_scores)
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title('Top TF-IDF Keywords')

        output_path = os.path.join(self.static_dir, output_filename)
        plt.savefig(output_path)
        plt.close()

        return top_words.index.tolist(), top_words.values.tolist()

    def visualize_video_description_keywords(self, videos_df, feature_names, tfidf_matrix, output_filename='tfidf_keywords_comments.png'):
        tfidf_description = tfidf_matrix[:len(videos_df), :]
        description_keywords = tfidf_description.sum(axis=0).A1
        description_keywords = list(zip(feature_names, description_keywords))
        description_keywords = sorted(description_keywords, key=lambda x: x[1], reverse=True)

        top_keywords = description_keywords[:10]
        words, scores = zip(*top_keywords)

        plt.figure(figsize=(10, 6))
        plt.barh(words, scores, color='blue')
        plt.xlabel('TF-IDF Score')
        plt.title('Top Keywords from Video Descriptions')
        plt.gca().invert_yaxis()

        output_path = os.path.join(self.static_dir, output_filename)
        plt.savefig(output_path)
        plt.close()

