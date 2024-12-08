import unittest
import pandas as pd
from unittest.mock import Mock, patch
from src.model.topic_modeling import TopicModelingProcessor


class TestTopicModelingProcessor(unittest.TestCase):
    def setUp(self):
        # Initialize TopicModelingProcessor with real directories
        self.processor = TopicModelingProcessor(
            model_dir="channel_analysis_models/",
            static_dir="mock_static_dir/"
        )

        # Mock TF-IDF vectorizer
        self.processor.tfidf_vectorizer = Mock()
        self.processor.tfidf_vectorizer.transform = Mock(
            return_value=[[0.1, 0.2, 0.3]] * 3  # Match the number of rows in videos_df
        )

        # Mock loaded model
        self.processor.loaded_model = Mock()
        self.processor.loaded_model.predict = Mock(
            side_effect=lambda x: [0] * len(x)  # Generate predictions matching the input length
        )

        # Sample data
        self.videos_df = pd.DataFrame({
            'title': ["Hello World", "Python Testing", "Topic Modeling"],
            'description': ["This is a description.", "Another description.", "Yet another description."]
        })
        self.comments_df = pd.DataFrame({
            'Comment': ["Great video!", "Very helpful", "Nice explanation"]
        })

    def test_preprocess_text(self):
        text = "Hello World! 123"
        result = self.processor.preprocess_text(text)
        expected = "hello world"
        self.assertEqual(result.strip(), expected)  # Ensure trailing spaces are removed

    def test_preprocess_data(self):
        videos_df, comments_df = self.processor.preprocess_data(
            self.videos_df.copy(), self.comments_df.copy()
        )
        self.assertIn('Cleaned_Title', videos_df.columns)
        self.assertIn('Cleaned_Description', videos_df.columns)
        self.assertIn('Cleaned_Comment', comments_df.columns)
        self.assertFalse(videos_df['Cleaned_Title'].isna().any())
        self.assertFalse(comments_df['Cleaned_Comment'].isna().any())

    def test_apply_model_predictions(self):
        # Preprocess data to ensure 'Cleaned_Title' exists
        videos_df, comments_df = self.processor.preprocess_data(
            self.videos_df.copy(), self.comments_df.copy()
        )

        # Apply model predictions
        videos_df, comments_df = self.processor.apply_model_predictions(videos_df, comments_df)

        self.assertIn('Cleaned_Title_Dominant_Topic', videos_df.columns)
        self.assertIn('Comment_Dominant_Topic', comments_df.columns)
        self.assertEqual(videos_df['Cleaned_Title_Dominant_Topic'][0], 0)

    def test_generate_summary(self):
        # Preprocess and apply model predictions
        videos_df, comments_df = self.processor.preprocess_data(
            self.videos_df.copy(), self.comments_df.copy()
        )
        videos_df, comments_df = self.processor.apply_model_predictions(videos_df, comments_df)

        # Mock mode outputs for dominant topics
        with patch.object(videos_df['Cleaned_Title_Dominant_Topic'], 'mode', return_value=pd.Series([0])):
            summary, dominant_topics = self.processor.generate_summary(videos_df, comments_df)

            self.assertIn("Dominant topics in video titles", summary)
            self.assertIsInstance(dominant_topics, dict)


if __name__ == "__main__":
    unittest.main()
