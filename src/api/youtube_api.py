import json
import re
from langdetect import detect, LangDetectException
from collections import Counter
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError


class YouTubeAPI:
    def __init__(self, credentials):
        self.youtube = build("youtube", "v3", credentials=credentials)

    @staticmethod
    def remove_emojis(text):
        emoji_pattern = re.compile(
            "["
            "\U0001F600-\U0001F64F"  # emoticons
            "\U0001F300-\U0001F5FF"  # symbols & pictographs
            "\U0001F680-\U0001F6FF"  # transport & map symbols
            "\U0001F1E0-\U0001F1FF"  # flags (iOS)
            "\u2702-\u27B0"
            "\u24C2-\U0001F251"
            "]+",
            flags=re.UNICODE,
        )
        return emoji_pattern.sub(r"", text)

    @staticmethod
    def extract_hashtags(text):
        return re.findall(r"#\w+", text)

    @staticmethod
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

    def get_channel_details(self, channel_id):
        try:
            channel_request = self.youtube.channels().list(
                part="snippet,statistics,contentDetails", id=channel_id
            )
            channel_response = channel_request.execute()

            if not channel_response.get("items"):
                return {"error": "Channel not found"}

            channel_data = channel_response["items"][0]

            video_request = self.youtube.search().list(
                part="snippet",
                channelId=channel_id,
                order="viewCount",
                maxResults=10,  # Fetch top 10 most viewed videos
            )
            video_response = video_request.execute()

            top_videos = []
            video_comments = {}

            for video in video_response["items"]:
                video_id = video["id"]["videoId"]
                video_details = (
                    self.youtube.videos()
                    .list(part="snippet,statistics", id=video_id)
                    .execute()
                )
                video_data = video_details["items"][0]

                # Extract hashtags from the description
                description = video_data["snippet"]["description"]
                hashtags = self.extract_hashtags(description)

                video_info = {
                    "title": video_data["snippet"]["title"],
                    "videoId": video_id,
                    "viewCount": video_data["statistics"].get("viewCount", 0),
                    "likeCount": video_data["statistics"].get("likeCount", 0),
                    "commentCount": video_data["statistics"].get("commentCount", 0),
                    "description": description,
                    "hashtags": hashtags,
                }
                top_videos.append(video_info)

                filtered_comments = []
                try:
                    comments_request = self.youtube.commentThreads().list(
                        part="snippet",
                        videoId=video_id,
                        maxResults=30,  # Fetch up to 30 comments
                    )
                    comments_response = comments_request.execute()

                    for comment_thread in comments_response.get("items", []):
                        comment_text = comment_thread["snippet"]["topLevelComment"][
                            "snippet"
                        ]["textDisplay"]
                        comment_text = self.remove_emojis(comment_text)
                        try:
                            if detect(comment_text) == "en":
                                filtered_comments.append(comment_text)
                        except LangDetectException:
                            continue
                        except Exception as e:
                            print(
                                f"An unexpected error occurred during language detection: {e}"
                            )
                            continue

                    dominant_language = self.detect_most_common_language(
                        filtered_comments
                    )

                    video_comments[video_id] = {
                        "comments": filtered_comments,
                        "dominantLanguage": dominant_language,
                    }
                except HttpError as e:
                    print(
                        f"An error occurred when fetching comments for video {video_id}: {e}"
                    )
                    video_comments[video_id] = {
                        "comments": [],
                        "dominantLanguage": None,
                    }

            channel_info = {
                "channel": {
                    "title": channel_data["snippet"]["title"],
                    "description": channel_data["snippet"]["description"],
                    "subscriberCount": channel_data["statistics"]["subscriberCount"],
                    "viewCount": channel_data["statistics"]["viewCount"],
                    "videoCount": channel_data["statistics"]["videoCount"],
                    "publishedAt": channel_data["snippet"]["publishedAt"],
                },
                "topVideos": top_videos,
                "videoComments": video_comments,
            }

            filename = f"channel_data_{channel_id}.json"
            with open(filename, "w") as f:
                json.dump(channel_info, f, indent=4)
            print(f"Data saved to {filename}")

            return channel_info

        except HttpError as e:
            print(f"An error occurred: {e}")
            return {"error": f"Failed to fetch data from YouTube API: {e}"}
