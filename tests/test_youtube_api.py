import pytest
from src.api.youtube_api import YouTubeAPI  # Import your real YouTubeAPI class

@pytest.fixture
def youtube_api():
    """
    Fixture to initialize the YouTubeAPI class with your real API key.
    Replace 'YOUR_API_KEY' with your actual YouTube API key.
    """
    api_key = "AIzaSyAqwn9cK_etVlcJ8f0L8q5g2ZVg-dJJ2U8"  # Your API Key
    return YouTubeAPI(api_key=api_key)

def test_fetch_video_comments(youtube_api):
    """
    Test fetching the first 5 comments from a specific YouTube video.
    """
    video_id = "QmZrUAPmcds"  # Example video ID
    comments = youtube_api.get_comments(video_id)

    # Ensure we get at least 5 comments
    assert len(comments) >= 5

    # Print the comments for debugging purposes
    print("Fetched comments successfully:", comments[:5])

    # Perform assertions to check the content
    assert isinstance(comments, list)
    assert all(isinstance(comment, str) for comment in comments)









