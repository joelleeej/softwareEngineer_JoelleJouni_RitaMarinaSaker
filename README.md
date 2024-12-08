# YouTube Channel Analysis Dashboard

## Overview
A Flask-based web application that leverages the YouTube Data API to analyze YouTube channels. This project allows users to:
- Analyze channel metrics like subscriber count, video count, and overall engagement.
- Extract insights from top-performing videos and their comments.
- Perform NLP-based keyword extraction and sentiment analysis using TF-IDF and BERT.
- Visualize engagement statistics, sentiment distributions, and keyword importance.
- Recommend similar channels for collaboration based on keywords and topics.

---

## Features
- **YouTube API Integration**: The `YouTubeAPI` class, located in `src/api/youtube_api.py`, fetches detailed metrics, videos, and comments for a given channel.
- **Data Cleaning**: The `DataCleaner` class, located in `src/data/cleaning.py`, handles:
  - Loading raw data from `.json` files.
  - Cleaning and preprocessing video and comment data.
  - Applying Natural Language Processing (NLP) techniques like sentiment analysis and topic modeling.
  - Extracting features like engagement rates and sentiment scores.
- **NLP Analytics**:
  - **TF-IDF**: The `TFIDFProcessor` class, located in `src/nlp/tf_idf.py`, extracts keywords from video descriptions and comments. Results are visualized in the static/ directory.
  - **BERT**:  The `BERTProcessor` class, located in `src/nlp/bert.py`, performs advanced keyword extraction and contextual analysis. Visualization results are saved in static/.
  - **Topic Modeling**: -the `TopicModelingProcessor` class, located in `src/model/topic_modeling.py`, identifies dominant topics in video content.
- **Visualizations**:
  - The `VisualizationProcessor` class, located in `src/visualization/visualization_channel.py`, generates:
  - **Engagement Rate Chart**: Bar plot of engagement rate per video.
  - **Video and Comment Sentiment Distribution**: Histograms for sentiment analysis.
  - **Topic Distribution**: Bar plot for average topic distribution.
- **Collaboration Recommendations**: - The `YouTubeAPIProcessor` class, located in `src/api/youtube_api_2.py`, recommends similar channels based on top keywords and topics.

---

## Installation

1. **Clone this repository**:
   ```bash
   git clone https://github.com/joelleeej/softwareEngineer_JoelleJouni_RitaMarinaSaker
   cd softwareEngineer_JoelleJouni_RitaMarinaSaker
   ```

2. **Set up a virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate   # For Linux/Mac
   venv\Scripts\activate    # For Windows
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Add YouTube API credentials**:
   - Download your `client_secret.json` file from the [Google Cloud Console](https://console.cloud.google.com/).
   - Place the `client_secret.json` file in the project root directory.

5. **Run the application**:
   ```bash
   python app.py
   ```

---

## Usage
1. Visit `https://127.0.0.1:5000/` in your browser.
2. Log in using your Google account to authorize the app.
3. Enter a YouTube channel ID to analyze its data.
4. View the dashboard with:
   - Channel insights and metrics.
   - Visualizations for engagement, keywords, and sentiment.
   - Collaboration recommendations.

---
## Testing
### Running Tests:
To run unit tests for the implemented features:

  ```pytest tests/
  ```
### Coverage Report:
Generate and view code coverage:

  ```pytest --cov=src --cov-report=html
  ```
Open ```htmlcov/index.html``` for detailed insights.

## Project Structure
```plaintext
marketing_project/
│
├── app.py                       # Main Flask application
├── README.md                    # Project documentation
├── requirements.txt             # Python dependencies
├── static/                      # Directory for generated visualizations
│   ├── engagement_statistics.png
│   ├── topic_distribution.png
│   ├── video_sentiment_distribution.png
│   ├── comments_sentiment_distribution.png
│   └── ...
├── templates/                   # HTML templates
│   ├── dashboard.html
│   └── ...
├── src/                         # Source code
│   ├── api/
│   │   └── youtube_api_2.py     # YouTube API functions
│   ├── data/
│   │   └── cleaning.py          # Data cleaning functions
│   ├── nlp/
│   │   ├── bert.py              # BERT-based NLP
│   │   └── tf_idf.py            # TF-IDF-based NLP
│   ├── visualization/
│   │   └── visualization_channel.py  # Visualization functions
│   └── model/topic_modeling.py               # (Optional future directory for ML models)
├── datasets/                    # (Optional directory for storing datasets)
└── ...
```

---
## ScreenShots:

## Roadmap
-  **Create Application**: Flask WebInterface, Ml/DL models, visualizations, dashboards.
- **OOP Refactoring**: Transition all modules to object-oriented programming.
- **Unit Testing**: Add comprehensive unit tests using `pytest`.
- **Enhanced Visualizations**: Improve dashboard interactivity.
- **Containerization**: Create a `Dockerfile` for easy deployment.
- **Deployment**: Deploy the app on Heroku, AWS, or similar platforms.

---

## License
This project is licensed under the MIT License.

