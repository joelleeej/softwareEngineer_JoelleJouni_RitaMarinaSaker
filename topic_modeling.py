import pandas as pd
import joblib
import re
import matplotlib.pyplot as plt
import seaborn as sns

# Load the saved model and TF-IDF vectorizer
loaded_model = joblib.load('channel_analysis_models/model.pkl')
tfidf_vectorizer = joblib.load('channel_analysis_models/tfidf_vectorizer.pkl')

# Preprocess function to clean text data
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    text = re.sub(r'\d+', '', text)  # Remove digits
    return text

# Apply preprocessing to the video_df and comments_df columns
def preprocess_data(videos_df, comments_df):
    videos_df['Cleaned_Title'] = videos_df['title'].apply(preprocess_text)
    videos_df['Cleaned_Description'] = videos_df['description'].apply(preprocess_text)
    comments_df['Cleaned_Comment'] = comments_df['Comment'].apply(preprocess_text)
    return videos_df, comments_df

# Function to apply model predictions
def apply_model_predictions(videos_df, comments_df):
    video_columns = ['Cleaned_Title', 'Cleaned_Description']
    for column in video_columns:
        X_tfidf = tfidf_vectorizer.transform(videos_df[column])
        predictions = loaded_model.predict(X_tfidf)
        videos_df[f"{column}_Dominant_Topic"] = predictions

    X_tfidf_comments = tfidf_vectorizer.transform(comments_df['Cleaned_Comment'])
    comments_df['Comment_Dominant_Topic'] = loaded_model.predict(X_tfidf_comments)

    return videos_df, comments_df

# Function to visualize the dominant topics
def visualize_topics(df, column_name, title, image_filename):
    plt.figure(figsize=(12, 6))
    sns.countplot(x=column_name, data=df)
    plt.title(title)
    plt.xlabel('Dominant Topic')
    plt.ylabel('Frequency')
    plt.xticks(rotation=45)
    image_path = f"static/{image_filename}.png"
    plt.savefig(image_path, format='png')
    plt.close()
    return image_path

# Function to generate summary
def generate_summary(videos_df, comments_df):
    topic_names = {0: 'World', 1: 'Sports', 2: 'Business', 3: 'Sci/Tech'}
    
    video_title_dominant_topic = topic_names[videos_df['Cleaned_Title_Dominant_Topic'].mode()[0]]
    video_description_dominant_topic = topic_names[videos_df['Cleaned_Description_Dominant_Topic'].mode()[0]]
    
    comment_dominant_topic = topic_names[comments_df['Comment_Dominant_Topic'].mode()[0]]
    
    video_summary = f"Dominant topics in video titles: {video_title_dominant_topic}\n"
    video_summary += f"Dominant topics in video descriptions: {video_description_dominant_topic}\n"
    dominant_topics = {
        'video_title': video_title_dominant_topic,
        'video_description': video_description_dominant_topic,
        'comment': comment_dominant_topic
    }
    comment_summary = f"Dominant topic in comments: {comment_dominant_topic}\n"
    
    summary = f"The dominant topic in these columns are:\n\n{video_summary}{comment_summary}"
    
    return summary, dominant_topics

# Main function to process data and visualize topics
def process_and_visualize(videos_df, comments_df):
    # Preprocess data
    videos_df, comments_df = preprocess_data(videos_df, comments_df)
    
    # Apply model predictions
    videos_df, comments_df = apply_model_predictions(videos_df, comments_df)
    
    # Visualize dominant topics for videos_df and comments_df with dynamic filenames
    video_title_img = visualize_topics(videos_df, 'Cleaned_Title_Dominant_Topic', 'Dominant Topics in Video Titles', 'video_title_img')
    video_description_img = visualize_topics(videos_df, 'Cleaned_Description_Dominant_Topic', 'Dominant Topics in Video Descriptions', 'video_description_img')
    comment_img = visualize_topics(comments_df, 'Comment_Dominant_Topic', 'Dominant Topics in Comments', 'comment_img')
    
    # Generate summary text
    summary = generate_summary(videos_df, comments_df)
    
    return summary, video_title_img, video_description_img, comment_img, videos_df, comments_df





