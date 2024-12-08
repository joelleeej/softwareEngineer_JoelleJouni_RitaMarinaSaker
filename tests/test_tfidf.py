import unittest
import pandas as pd
from src.nlp.tf_idf import TFIDFProcessor
from unittest.mock import MagicMock
import os
import re


class TestTFIDFProcessor(unittest.TestCase):
    def setUp(self):
        self.tfidf_processor = TFIDFProcessor(static_dir='test_static/')
        self.videos_df = pd.DataFrame({
            'description': [
                'This is the first video description.',
                'Another video description with important information.',
                'This is just a test video description.'
            ]
        })
        self.comments_df = pd.DataFrame({
            'Comment': [
                'Great video!',
                'Very informative content.',
                'Amazing explanation in the video.'
            ]
        })

    def tearDown(self):
        # Clean up the static directory created during testing
        if os.path.exists(self.tfidf_processor.static_dir):
            for file in os.listdir(self.tfidf_processor.static_dir):
                os.remove(os.path.join(self.tfidf_processor.static_dir, file))
            os.rmdir(self.tfidf_processor.static_dir)

    def preprocess_text(self, text):
    # Convert to lowercase
        text = text.lower()
    # Remove numbers
        text = re.sub(r'\d+', '', text)
    # Remove punctuation
        text = re.sub(r'[^\w\s]', '', text)
    # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def test_clean_data(self):
        cleaned_videos_df, cleaned_comments_df = self.tfidf_processor.clean_data(self.videos_df, self.comments_df)
        self.assertIn('Cleaned_Description', cleaned_videos_df.columns)
        self.assertIn('Cleaned_Comment', cleaned_comments_df.columns)
        self.assertNotEqual(self.videos_df['description'][0], cleaned_videos_df['Cleaned_Description'][0])

    def test_apply_tfidf(self):
        self.tfidf_processor.clean_data(self.videos_df, self.comments_df)
        tfidf_df, feature_names, tfidf_matrix = self.tfidf_processor.apply_tfidf(self.videos_df, self.comments_df)
        self.assertTrue(len(feature_names) <= self.tfidf_processor.max_features)
        self.assertEqual(tfidf_df.shape[1], len(feature_names))

    def test_visualize_tfidf_keywords(self):
        self.tfidf_processor.clean_data(self.videos_df, self.comments_df)
        tfidf_df, feature_names, tfidf_matrix = self.tfidf_processor.apply_tfidf(self.videos_df, self.comments_df)
        top_words, top_scores = self.tfidf_processor.visualize_tfidf_keywords(tfidf_df)
        output_path = os.path.join(self.tfidf_processor.static_dir, 'tfidf_keywords_descriptions.png')
        self.assertTrue(os.path.exists(output_path))
        self.assertEqual(len(top_words), 10)

    def test_visualize_video_description_keywords(self):
        self.tfidf_processor.clean_data(self.videos_df, self.comments_df)
        tfidf_df, feature_names, tfidf_matrix = self.tfidf_processor.apply_tfidf(self.videos_df, self.comments_df)
        self.tfidf_processor.visualize_video_description_keywords(self.videos_df, feature_names, tfidf_matrix)
        output_path = os.path.join(self.tfidf_processor.static_dir, 'tfidf_keywords_comments.png')
        self.assertTrue(os.path.exists(output_path))


if __name__ == "__main__":
    unittest.main()
