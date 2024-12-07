import torch
import numpy as np
import re
from transformers import BertTokenizer, BertModel
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import os

class BERTProcessor:
    def __init__(self, static_dir='static/', top_n=10):
        self.static_dir = static_dir
        self.top_n = top_n
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = BertModel.from_pretrained('bert-base-uncased')
        
        # Ensure the static directory exists
        os.makedirs(self.static_dir, exist_ok=True)

    @staticmethod
    def preprocess_text(text):
        """ Preprocess text by removing punctuation, digits, and converting to lowercase. """
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
        text = re.sub(r'\d+', '', text)  # Remove digits
        return text

    def preprocess_data(self, videos_df, comments_df):
        """ Preprocess video descriptions and comments. """
        videos_df['Cleaned_Description'] = videos_df['description'].apply(self.preprocess_text)
        comments_df['Cleaned_Comment'] = comments_df['Comment'].apply(self.preprocess_text)
        return videos_df, comments_df

    def extract_bert_embeddings(self, text_list):
        """ Extract BERT embeddings from a list of text. """
        embeddings = []
        with torch.no_grad():
            for text in text_list:
                inputs = self.tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512)
                outputs = self.model(**inputs)
                hidden_states = outputs.last_hidden_state
                attention_mask = inputs['attention_mask']

                # Perform mean pooling (average embeddings across tokens)
                mask = attention_mask.unsqueeze(-1).expand(hidden_states.size())
                sum_hidden = (hidden_states * mask).sum(1)
                sum_mask = mask.sum(1)
                pooled = sum_hidden / sum_mask
                embeddings.append(pooled.squeeze().cpu().numpy())
        
        return np.array(embeddings)

    @staticmethod
    def extract_keywords_from_embeddings(embeddings, top_n):
        """ Extract important keywords from embeddings based on cosine similarity. """
        similarity_matrix = cosine_similarity(embeddings)
        avg_similarity = similarity_matrix.mean(axis=0)
        top_indices = avg_similarity.argsort()[-top_n:][::-1]
        return top_indices, avg_similarity[top_indices]

    def plot_keywords(self, keywords, scores, title, filename):
        """ Plot the top keywords and their scores. """
        plt.figure(figsize=(10, 6))
        plt.barh(keywords, scores, color='skyblue')
        plt.xlabel('Score')
        plt.ylabel('Keyword')
        plt.title(title)
        plt.gca().invert_yaxis()
        output_path = os.path.join(self.static_dir, filename)
        plt.savefig(output_path)
        plt.close()

    def analyze(self, videos_df, comments_df):
        # Preprocess the data
        videos_df, comments_df = self.preprocess_data(videos_df, comments_df)

        # Extract BERT embeddings for video descriptions and comments
        video_embeddings = self.extract_bert_embeddings(videos_df['Cleaned_Description'].tolist())
        comment_embeddings = self.extract_bert_embeddings(comments_df['Cleaned_Comment'].tolist())

        # Extract top keywords from the embeddings and get the scores
        top_video_indices, video_keyword_scores = self.extract_keywords_from_embeddings(video_embeddings, self.top_n)
        top_comment_indices, comment_keyword_scores = self.extract_keywords_from_embeddings(comment_embeddings, self.top_n)

        # Visualize and save results
        video_keywords = [videos_df['Cleaned_Description'].iloc[i] for i in top_video_indices]
        comment_keywords = [comments_df['Cleaned_Comment'].iloc[i] for i in top_comment_indices]
        
        self.plot_keywords(video_keywords, video_keyword_scores, 
                           "Top Keywords in Video Descriptions (BERT)", "bert_keywords_descriptions.png")
        self.plot_keywords(comment_keywords, comment_keyword_scores, 
                           "Top Keywords in Comments (BERT)", "bert_keywords_comments.png")

        return top_video_indices, video_keyword_scores, top_comment_indices, comment_keyword_scores


