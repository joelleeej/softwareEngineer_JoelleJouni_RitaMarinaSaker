from src.data.cleaning import DataCleaner
import pandas as pd

def test_clean_video_data():
    cleaner = DataCleaner()
    sample_data = pd.DataFrame({
        "title": ["Test Video ", "Another Video "],
        "description": ["This is a test ", "Another description"]
    })
    cleaned_data = cleaner.clean_video_data(sample_data)
    assert cleaned_data["title"].iloc[0] == "Test Video"
    assert cleaned_data["description"].iloc[0] == "This is a test"
