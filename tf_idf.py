import json
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
from wordcloud import WordCloud


# Function to preprocess text by removing punctuation, digits, and converting to lowercase
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    text = re.sub(r'\d+', '', text)  # Remove digits
    return text


# Function to clean video descriptions and comments
def clean_data(videos_df, comments_df):
    # Apply preprocessing to descriptions and comments
    videos_df['Cleaned_Description'] = videos_df['description'].apply(preprocess_text)
    comments_df['Cleaned_Comment'] = comments_df['Comment'].apply(preprocess_text)
    return videos_df, comments_df


# Function to apply TF-IDF vectorization and return the results
def apply_tfidf(videos_df, comments_df, max_features=20):
    # Combine both video descriptions and comments into a single list for TF-IDF processing
    combined_text = list(videos_df['Cleaned_Description']) + list(comments_df['Cleaned_Comment'])

    # Initialize the TF-IDF Vectorizer
    tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_features=max_features)

    # Fit and transform the combined text data
    tfidf_matrix = tfidf_vectorizer.fit_transform(combined_text)

    # Extract the feature names (words) and their corresponding TF-IDF scores
    feature_names = tfidf_vectorizer.get_feature_names_out()
    dense_matrix = tfidf_matrix.todense()
    denselist = dense_matrix.tolist()

    # Create a DataFrame for the TF-IDF scores
    tfidf_df = pd.DataFrame(denselist, columns=feature_names)

    return tfidf_df, feature_names, tfidf_matrix


# Function to visualize the top TF-IDF keywords
def visualize_tfidf_keywords(tfidf_df, feature_names, output_path, num_words=10):
    # Aggregate the TF-IDF scores and sort the keywords by score
    sum_tfidf_scores = tfidf_df.sum(axis=0).sort_values(ascending=False)
    top_words = sum_tfidf_scores.head(num_words)
    # Create a word cloud to visualize the top keywords
    plt.figure(figsize=(10, 6))
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(sum_tfidf_scores)
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title('Top TF-IDF Keywords')
    
    # Save the plot to the output path
    plt.savefig(output_path)
    plt.close()  # Close the plot to avoid display in non-interactive environments
    return top_words.index.tolist(), top_words.values.tolist()

# Function to visualize the top keywords for video descriptions
def visualize_video_description_keywords(videos_df, feature_names, tfidf_matrix, output_path):
    # Get the TF-IDF scores for video descriptions
    tfidf_description = tfidf_matrix[:len(videos_df), :]

    # Sum the TF-IDF scores for each word across all video descriptions
    description_keywords = tfidf_description.sum(axis=0).A1
    description_keywords = list(zip(feature_names, description_keywords))
    description_keywords = sorted(description_keywords, key=lambda x: x[1], reverse=True)

    # Visualize top keywords from video descriptions
    top_keywords = description_keywords[:10]
    words, scores = zip(*top_keywords)
    
    plt.figure(figsize=(10, 6))
    plt.barh(words, scores, color='blue')
    plt.xlabel('TF-IDF Score')
    plt.title('Top Keywords from Video Descriptions')
    plt.gca().invert_yaxis()  # Invert y-axis to display the highest score at the top
    
    # Save the plot to the output path
    plt.savefig(output_path)
    plt.close()  # Close the plot to avoid display in non-interactive environments

