import os
from google_auth_oauthlib.flow import Flow
from googleapiclient.discovery import build
from google.oauth2.credentials import Credentials
from src.api.youtube_api import YouTubeAPI  # Updated import for class-based YouTubeAPI
from src.data.cleaning import DataCleaner
from src.nlp.tf_idf import TFIDFProcessor
from flask import (
    Flask,
    render_template,
    send_from_directory,
    redirect,
    request,
    session,
    url_for,
)
from src.nlp.bert import BERTProcessor
from src.model.topic_modeling import TopicModelingProcessor
from src.api.youtube_api_2 import YouTubeAPIProcessor
import mlflow
import mlflow.sklearn
from prometheus_client import Counter, Gauge, generate_latest
from prometheus_flask_exporter import PrometheusMetrics



app = Flask(__name__)
app.secret_key = os.urandom(24)

metrics = PrometheusMetrics(app)
metrics.info('app_info', 'YouTube Channel Analysis Application', version='1.0.0')

CLIENT_SECRETS_FILE = os.getenv(
    "CLIENT_SECRETS_FILE",
    "client_secret_986605252946-1pcv2saf5bb5dri7vdo59hid58r2kn7e.apps.googleusercontent.com.json",
)
API_NAME = "youtube"
API_VERSION = "v3"
SCOPES = [
    "https://www.googleapis.com/auth/youtube.readonly",
    "https://www.googleapis.com/auth/youtube.force-ssl",
]

# Set up the OAuth flow
flow = Flow.from_client_secrets_file(
    CLIENT_SECRETS_FILE, scopes=SCOPES, redirect_uri="https://127.0.0.1:5000/callback"
)


@app.route("/")
def index():
    if "credentials" not in session:
        return redirect(url_for("authorize"))
    return render_template("index.html")


@app.route("/authorize")
def authorize():
    authorization_url, state = flow.authorization_url(
        access_type="offline", prompt="consent"
    )
    session["state"] = state
    return redirect(authorization_url)


@app.route("/callback")
def callback():
    flow.fetch_token(authorization_response=request.url)
    credentials = flow.credentials
    session["credentials"] = credentials_to_dict(credentials)
    return redirect(url_for("index"))


@app.route("/analyze", methods=["POST"])
def analyze():
    channel_id = request.form["channel_id"]
    if "credentials" not in session:
        return redirect(url_for("authorize"))

    credentials = Credentials.from_authorized_user_info(session["credentials"])
    yt_api = YouTubeAPI(credentials)  # Instantiate YouTubeAPI

    # Start an MLflow experiment
    mlflow.set_experiment("YouTube Channel Analysis")
    with mlflow.start_run():
        # Fetch channel data
        channel_data = yt_api.get_channel_details(channel_id)
        if "error" in channel_data:
            return f"Error: {channel_data['error']}"

        cleaner = DataCleaner(channel_id)
        tfidf_processor = TFIDFProcessor()
        bert_processor = BERTProcessor(static_dir="static/")
        topic_modeling_processor = TopicModelingProcessor(
            model_dir="channel_analysis_models/", static_dir="static/"
        )

        # Process and clean the data
        channel_df, comments_df, videos_df = cleaner.process_data(channel_data)
        videos_df = cleaner.clean_video_data(videos_df)
        videos_df, comments_df = cleaner.apply_nlp(videos_df, comments_df)
        videos_df, comments_df = tfidf_processor.clean_data(videos_df, comments_df)

        mlflow.log_param("channel_id", channel_id)
        mlflow.log_metric("video_count", len(videos_df))
        mlflow.log_metric("comment_count", len(comments_df))

        # Apply TF-IDF and extract features
        tfidf_df, feature_names, tfidf_matrix = tfidf_processor.apply_tfidf(
            videos_df, comments_df
        )
        mlflow.log_metric("top_words", len(tfidf_df))

        top_words, top_scores = tfidf_processor.visualize_tfidf_keywords(tfidf_df)
        top_word_scores = list(zip(top_words, top_scores))
        combined_df, updated_channel_df = cleaner.extract_features(
            channel_df, videos_df, comments_df
        )

        average_sentiment = updated_channel_df["average_sentiment"].iloc[0]

        # Apply BERT-based keyword extraction
        bert_processor.analyze(videos_df, comments_df)

        # Visualize and save results

        cleaner.plot_sentiment_distribution(comments_df)
        cleaner.plot_topic_distribution(videos_df)
        tfidf_processor.visualize_tfidf_keywords(tfidf_df)
        tfidf_processor.visualize_video_description_keywords(
            videos_df, feature_names, tfidf_matrix
        )

        # Process and visualize topics using the topic_modeling.py functions
        (
            summary,
            video_title_img,
            video_description_img,
            comment_img,
            dominant_topics,
        ) = topic_modeling_processor.process_and_visualize(videos_df, comments_df)

        # Log artifacts
        mlflow.log_artifact(video_title_img)
        mlflow.log_artifact(video_description_img)
        mlflow.log_artifact(comment_img)

        # Render the dashboard with results
        return render_template(
            "dashboard.html",
            summary=summary,
            channel_data=updated_channel_df.to_dict(orient="records"),
            videos_data=videos_df.to_dict(orient="records"),
            comments_data=comments_df.to_dict(orient="records"),
            video_titles=[
                video["title"] for video in videos_df.to_dict(orient="records")
            ],
            engagement_rates=[
                video["engagement_rate"]
                for video in videos_df.to_dict(orient="records")
            ],
            video_title_img=video_title_img,
            video_description_img=video_description_img,
            comment_img=comment_img,
            average_sentiment=average_sentiment,
            top_word_scores=top_word_scores,
            dominant_topics=dominant_topics,
        )


@app.route("/search_results", methods=["POST"])
def search_results():
    if "credentials" not in session:
        return redirect(url_for("authorize"))

    credentials = Credentials.from_authorized_user_info(session["credentials"])
    youtube = build(API_NAME, API_VERSION, credentials=credentials)

    # Instantiate YouTubeAPIProcessor with the authenticated YouTube client
    yt_processor = YouTubeAPIProcessor(youtube)

    # Fetch relevant channels based on keywords from form input
    keywords = request.form.get("keywords").split()
    extracted_data = yt_processor.search_channels_by_keywords(keywords)

    return render_template("search_results.html", extracted_data=extracted_data)


@app.route("/static/<filename>")
def serve_image(filename):
    return send_from_directory(os.path.join(app.root_path, "static"), filename)


def credentials_to_dict(credentials):
    return {
        "token": credentials.token,
        "refresh_token": (
            credentials.refresh_token if credentials.refresh_token else None
        ),
        "token_uri": credentials.token_uri,
        "client_id": credentials.client_id,
        "client_secret": credentials.client_secret,
        "scopes": credentials.scopes,
    }
# Metrics definitions
request_counter = Counter('request_count', 'Total number of requests made to the application')
active_requests = Gauge('active_requests', 'Number of active requests currently being processed')

@app.before_request
def before_request():
    request_counter.inc()
    active_requests.inc()

@app.after_request
def after_request(response):
    active_requests.dec()
    return response

@app.route('/metrics')
def metrics():
    return generate_latest(), 200, {'Content-Type': 'text/plain; charset=utf-8'}


if __name__ == "__main__":
    app.run(
        debug=True,
        host="0.0.0.0",
        port=5000,
        ssl_context=("certs/certificate.crt", "certs/private.key"),
    )
