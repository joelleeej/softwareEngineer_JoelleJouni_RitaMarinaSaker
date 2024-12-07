import os
import atexit
import json
from sqlite3 import DataError
from google_auth_oauthlib.flow import Flow
from googleapiclient.discovery import build
from google.oauth2.credentials import Credentials
from src.api.youtube_api import YouTubeAPI  # Updated import for class-based YouTubeAPI
from src.data.cleaning import DataCleaner
from tf_idf import visualize_tfidf_keywords, visualize_video_description_keywords, apply_tfidf, clean_data
from flask import Flask, render_template, send_from_directory, redirect, request, session, url_for
from bert import analyze_bert, plot_keywords
import matplotlib.pyplot as plt
from topic_modeling import process_and_visualize, generate_summary
from youtube_api_2 import search_channels_by_keywords

app = Flask(__name__)
app.secret_key = os.urandom(24)

CLIENT_SECRETS_FILE = "client_secret_986605252946-ihc6sj61e8o858cqcm2fpu585igddmdf.apps.googleusercontent.com (1).json"
API_NAME = 'youtube'
API_VERSION = 'v3'
SCOPES = ['https://www.googleapis.com/auth/youtube.readonly', 'https://www.googleapis.com/auth/youtube.force-ssl']

# Set up the OAuth flow
flow = Flow.from_client_secrets_file(
    CLIENT_SECRETS_FILE,
    scopes=SCOPES,
    redirect_uri='https://127.0.0.1:5000/callback'
)

@app.route('/')
def index():
    if 'credentials' not in session:
        return redirect(url_for('authorize'))
    return render_template('index.html')

@app.route('/authorize')
def authorize():
    authorization_url, state = flow.authorization_url(access_type='offline', prompt='consent')
    session['state'] = state
    return redirect(authorization_url)

@app.route('/callback')
def callback():
    flow.fetch_token(authorization_response=request.url)
    credentials = flow.credentials
    session['credentials'] = credentials_to_dict(credentials)
    return redirect(url_for('index'))

@app.route('/analyze', methods=['POST'])
def analyze():
    channel_id = request.form['channel_id']
    if 'credentials' not in session:
        return redirect(url_for('authorize'))

    credentials = Credentials.from_authorized_user_info(session['credentials'])
    yt_api = YouTubeAPI(credentials)  # Instantiate YouTubeAPI

    # Fetch channel data
    channel_data = yt_api.get_channel_details(channel_id)
    if "error" in channel_data:
        return f"Error: {channel_data['error']}"
    cleaner = DataCleaner(channel_id)
    # Process and clean the data
    channel_df, comments_df, videos_df = cleaner.process_data(channel_data)
    videos_df = cleaner.clean_video_data(videos_df)
    videos_df, comments_df = cleaner.apply_nlp(videos_df, comments_df)
    videos_df, comments_df = clean_data(videos_df, comments_df)

    # Apply TF-IDF and extract features
    tfidf_df, feature_names, tfidf_matrix = apply_tfidf(videos_df, comments_df)
    top_words, top_scores = visualize_tfidf_keywords(tfidf_df, feature_names, 'static/tfidf_keywords_descriptions.png')
    top_word_scores = list(zip(top_words, top_scores))
    combined_df, updated_channel_df = cleaner.extract_features(channel_df, videos_df, comments_df)

    average_sentiment = updated_channel_df['average_sentiment'].iloc[0]

    # Apply BERT-based keyword extraction
    top_video_indices, video_keyword_scores, top_comment_indices, comment_keyword_scores = analyze_bert(videos_df, comments_df)

    # Prepare data for plotting
    video_keywords = [videos_df['Cleaned_Description'].iloc[i] for i in top_video_indices]
    comment_keywords = [comments_df['Cleaned_Comment'].iloc[i] for i in top_comment_indices]

    # Visualize and save results
    plot_keywords(video_keywords, video_keyword_scores, "Top Keywords in Video Descriptions (BERT)", "static/bert_keywords_descriptions.png")
    plot_keywords(comment_keywords, comment_keyword_scores, "Top Keywords in Comments (BERT)", "static/bert_keywords_comments.png")
    cleaner.plot_sentiment_distribution(comments_df)
    cleaner.plot_topic_distribution(videos_df)
    visualize_tfidf_keywords(tfidf_df, feature_names, 'static/tfidf_keywords_descriptions.png')
    visualize_video_description_keywords(videos_df, feature_names, tfidf_matrix, 'static/tfidf_keywords_comments.png')

    # Process and visualize topics using the topic_modeling.py functions
    summary, video_title_img, video_description_img, comment_img, videos_df, comments_df = process_and_visualize(videos_df, comments_df)
    dominant_topics = generate_summary(videos_df, comments_df)

    # Render the dashboard with results
    return render_template(
        'dashboard.html',
        summary=summary,
        channel_data=updated_channel_df.to_dict(orient='records'),
        videos_data=videos_df.to_dict(orient='records'),
        comments_data=comments_df.to_dict(orient='records'),
        video_titles=[video['title'] for video in videos_df.to_dict(orient='records')],
        engagement_rates=[video['engagement_rate'] for video in videos_df.to_dict(orient='records')],
        video_title_img=video_title_img,
        video_description_img=video_description_img,
        comment_img=comment_img,
        average_sentiment=average_sentiment,
        top_word_scores=top_word_scores,
        dominant_topics=dominant_topics
    )

@app.route('/search_results', methods=['POST'])
def search_results():
    if 'credentials' not in session:
        return redirect(url_for('authorize'))

    credentials = Credentials.from_authorized_user_info(session['credentials'])
    youtube = build(API_NAME, API_VERSION, credentials=credentials)

    keywords = request.form.get('keywords').split()
    extracted_data = search_channels_by_keywords(youtube, keywords)

    return render_template(
        'search_results.html',
        extracted_data=extracted_data
    )

@app.route('/static/<filename>')
def serve_image(filename):
    return send_from_directory(os.path.join(app.root_path, 'static'), filename)

def credentials_to_dict(credentials):
    return {
        'token': credentials.token,
        'refresh_token': credentials.refresh_token if credentials.refresh_token else None,
        'token_uri': credentials.token_uri,
        'client_id': credentials.client_id,
        'client_secret': credentials.client_secret,
        'scopes': credentials.scopes
    }

if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.1', port=5000, ssl_context=('certs/certificate.crt', 'certs/private.key'))














