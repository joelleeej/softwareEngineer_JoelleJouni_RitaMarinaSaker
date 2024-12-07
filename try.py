import json
import re
from langdetect import detect, LangDetectException
from collections import Counter
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

# Function to remove emojis
def remove_emojis(text):
    emoji_pattern = re.compile(
        "[" 
        "\U0001F600-\U0001F64F"  # emoticons
        "\U0001F300-\U0001F5FF"  # symbols & pictographs
        "\U0001F680-\U0001F6FF"  # transport & map symbols
        "\U0001F1E0-\U0001F1FF"  # flags (iOS)
        "\U00002702-\U000027B0"
        "\U000024C2-\U0001F251"
        "]+",
        flags=re.UNICODE
    )
    return emoji_pattern.sub(r'', text)

# Function to detect the most common language in a list of comments
def detect_most_common_language(comments):
    languages = []
    for comment in comments:
        try:
            lang = detect(comment)
            languages.append(lang)
        except LangDetectException:
            continue
    if languages:
        most_common_lang = Counter(languages).most_common(1)[0][0]
        return most_common_lang
    return None

def search_channels_by_keywords(youtube, keywords):
    try:
        # Limit search results to top 3 channels based on the combined keywords
        search_request = youtube.search().list(
            part="snippet",
            q=' '.join(keywords),  # Keywords derived from dominant topics and top words
            type="channel",
            maxResults=3  # Fetch top 3 relevant channels
        )
        search_response = search_request.execute()

        channel_dataa = []

        # Fetch channel details for each channel found
        for item in search_response['items']:
            channel_id = item['snippet']['channelId']
            
            # Get detailed channel info
            channel_request = youtube.channels().list(
                part="snippet,statistics",
                id=channel_id
            )
            channel_response = channel_request.execute()

            if channel_response.get("items"):
                channel_info = channel_response['items'][0]
                channel_dataa.append({
                    "title": channel_info['snippet']['title'],
                    "subscriberCount": channel_info['statistics']['subscriberCount'],
                    "description": channel_info['snippet']['description']
                })

        # Save the channel data to a JSON file
        filename = 'relevant_channels.json'
        with open(filename, 'w') as f:
            json.dump(channel_dataa, f, indent=4)
        
        print(f"Channel data saved to {filename}")
        return channel_dataa

    except HttpError as e:
        print(f"An error occurred: {e}")
        return {"error": f"Failed to fetch channel data: {e}"}

if __name__ == "__main__":
    # Replace with your own API key or OAuth credentials setup
    API_KEY = "AIzaSyAqwn9cK_etVlcJ8f0L8q5g2ZVg-dJJ2U8"  # Make sure you replace this with a valid YouTube Data API key
    youtube = build('youtube', 'v3', developerKey=API_KEY)

    # Example list of keywords derived from previous analysis
    keywords = ["technology", "innovation", "gadget"]

    # Run the function and display the result
    channel_data = search_channels_by_keywords(youtube, keywords)
    print(json.dumps(channel_data, indent=4))
