import unittest
from unittest.mock import patch, MagicMock
import pandas as pd
import numpy as np
from src.nlp.bert import BERTProcessor
import re


class TestBERTProcessor(unittest.TestCase):
    def setUp(self):
        self.bert_processor = BERTProcessor(static_dir="test_static/")
        self.videos_df = pd.DataFrame(
            {
                "description": [
                    "This is the first video description.",
                    "Another video description with important information.",
                    "This is just a test video description.",
                ]
            }
        )
        self.comments_df = pd.DataFrame(
            {
                "Comment": [
                    "Great video!",
                    "Very informative content.",
                    "Amazing explanation in the video.",
                ]
            }
        )

    def preprocess_text(self, text):
        """
        Preprocess text by lowercasing, removing punctuation, and extra spaces.
        """
        text = text.lower()  # Convert to lowercase
        text = re.sub(
            r"[^a-z0-9\s]", "", text
        )  # Remove all non-alphanumeric characters except spaces
        text = re.sub(r"\s+", " ", text).strip()  # Remove extra spaces
        return text

    def test_preprocess_data(self):
        videos_df, comments_df = self.bert_processor.preprocess_data(
            self.videos_df, self.comments_df
        )
        self.assertIn("Cleaned_Description", videos_df.columns)
        self.assertIn("Cleaned_Comment", comments_df.columns)
        self.assertEqual(
            videos_df["Cleaned_Description"].iloc[0],
            "this is the first video description",
        )

    @patch("src.nlp.bert.BertTokenizer.from_pretrained")
    @patch("src.nlp.bert.BertModel.from_pretrained")
    def test_extract_bert_embeddings(self, mock_bert_model, mock_bert_tokenizer):
        # Mock BERT model and tokenizer
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_bert_model.return_value = mock_model
        mock_bert_tokenizer.return_value = mock_tokenizer

        mock_tokenizer.return_tensors.return_value = {
            "input_ids": np.array([[101, 2054, 2003, 2023, 102]]),  # Example token IDs
            "attention_mask": np.array([[1, 1, 1, 1, 1]]),
        }

        mock_model.return_value = MagicMock(last_hidden_state=np.random.rand(1, 5, 768))

        embeddings = self.bert_processor.extract_bert_embeddings(
            ["This is a test sentence."]
        )
        self.assertEqual(len(embeddings), 1)

    def test_plot_keywords(self):
        # Ensure plot_keywords saves a file without error
        keywords = ["keyword1", "keyword2", "keyword3"]
        scores = [0.9, 0.8, 0.7]
        try:
            self.bert_processor.plot_keywords(
                keywords, scores, "Test Plot", "test_plot.png"
            )
        except Exception as e:
            self.fail(f"plot_keywords raised an exception: {e}")

    @patch("src.nlp.bert.BERTProcessor.extract_bert_embeddings")
    @patch("src.nlp.bert.BERTProcessor.plot_keywords")
    def test_analyze(self, mock_plot_keywords, mock_extract_bert_embeddings):
        # Mock embedding extraction and plotting
        mock_extract_bert_embeddings.return_value = np.random.rand(3, 768)
        mock_plot_keywords.return_value = None

        results = self.bert_processor.analyze(self.videos_df, self.comments_df)
        self.assertEqual(len(results), 4)  # Ensure correct number of outputs
