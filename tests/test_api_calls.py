import pytest
from unittest.mock import patch
from src.api.youtube_api_2 import YouTubeAPIProcessor

@pytest.fixture
def youtube_api():
    return YouTubeAPIProcessor()

@patch('googleapiclient.discovery.build')
def test_search_channels(mock_build, youtube_api):
    mock_service = mock_build.return_value
    mock_service.search().list().execute.return_value = {
        "items": [{"id": {"channelId": "test_channel"}}]
    }
    channels = youtube_api.search_channels_by_keywords(["test", "keyword"])
    assert len(channels) == 1
    assert channels[0]["channelId"] == "test_channel"
