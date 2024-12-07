import json
import re
from langdetect import detect, LangDetectException
from collections import Counter
from googleapiclient.errors import HttpError


class YouTubeAPIProcessor:
    def __init__(self, youtube_client):
        """
        Initialize the processor with a YouTube API client.
        """
        self.youtube = youtube_client

    @staticmethod
    def remove_emojis(text):
        """
        Removes emojis from a given text.
        """
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

    @staticmethod
    def detect_most_common_language(comments):
        """
        Detects the most common language in a list of comments.
        """
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

    def search_channels_by_keywords(self, keywords):
        """
        Searches for YouTube channels based on given keywords.

        Parameters:
        - keywords: List of keywords derived from dominant topics and top words.

        Returns:
        - A list of relevant channels with their details.
        """
        try:
            search_request = self.youtube.search().list(
                part="snippet",
                q=' '.join(keywords),  # Combine keywords into a single search query
                type="channel",
                maxResults=3  # Fetch top 3 relevant channels
            )
            search_response = search_request.execute()

            channel_data = []

            # Fetch details for each channel found
            for item in search_response['items']:
                channel_id = item['snippet']['channelId']

                # Get detailed channel info
                channel_request = self.youtube.channels().list(
                    part="snippet,statistics",
                    id=channel_id
                )
                channel_response = channel_request.execute()

                if channel_response.get("items"):
                    channel_info = channel_response['items'][0]
                    channel_data.append({
                        "title": channel_info['snippet']['title'],
                        "subscriberCount": channel_info['statistics'].get('subscriberCount'),
                        "description": channel_info['snippet']['description']
                    })

            # Save the channel data to a JSON file
            self.save_to_json(channel_data, 'relevant_channels.json')
            print("Channel data saved successfully.")
            return channel_data

        except HttpError as e:
            print(f"An error occurred: {e}")
            return {"error": f"Failed to fetch channel data: {e}"}

    @staticmethod
    def save_to_json(data, filename):
        """
        Saves data to a JSON file.

        Parameters:
        - data: The data to save.
        - filename: The file name to save the data in.
        """
        try:
            with open(filename, 'w') as f:
                json.dump(data, f, indent=4)
            print(f"Data saved to {filename}")
        except Exception as e:
            print(f"Failed to save data to {filename}: {e}")

