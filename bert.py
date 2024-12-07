import torch
import numpy as np
import re
from transformers import BertTokenizer, BertModel
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt

# ----- Preprocessing for BERT -----
def preprocess_text(text):
    """ Preprocess text by removing punctuation, digits, and converting to lowercase. """
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    text = re.sub(r'\d+', '', text)  # Remove digits
    return text

def preprocess_data(videos_df, comments_df):
    """ Preprocess video descriptions and comments. """
    videos_df['Cleaned_Description'] = videos_df['description'].apply(preprocess_text)
    comments_df['Cleaned_Comment'] = comments_df['Comment'].apply(preprocess_text)
    return videos_df, comments_df

# ----- BERT Tokenization and Embedding Extraction -----
def load_bert_model():
    """ Load the BERT tokenizer and model. """
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')
    return tokenizer, model

def extract_bert_embeddings(text_list, tokenizer, model):
    """ Extract BERT embeddings from a list of text. """
    embeddings = []
    with torch.no_grad():
        for text in text_list:
            # Tokenize the text and get the input IDs and attention mask
            inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512)
            outputs = model(**inputs)
            # Get the last hidden state (embeddings) and pool the output (mean pooling)
            hidden_states = outputs.last_hidden_state
            attention_mask = inputs['attention_mask']
            
            # Perform mean pooling (average embeddings across tokens)
            mask = attention_mask.unsqueeze(-1).expand(hidden_states.size())
            sum_hidden = (hidden_states * mask).sum(1)
            sum_mask = mask.sum(1)
            pooled = sum_hidden / sum_mask
            embeddings.append(pooled.squeeze().cpu().numpy())
    
    return np.array(embeddings)

# ----- Combine the embeddings for keyword extraction -----
def extract_keywords_from_embeddings(embeddings, top_n=10):
    """ Extract important keywords from embeddings based on cosine similarity. """
    # Calculate cosine similarity between all the embeddings
    similarity_matrix = cosine_similarity(embeddings)
    # Average the similarities for each word
    avg_similarity = similarity_matrix.mean(axis=0)
    
    # Get the indices of the top N most important keywords
    top_indices = avg_similarity.argsort()[-top_n:][::-1]
    return top_indices, avg_similarity[top_indices]

# ----- Visualization of Top Keywords -----
def plot_keywords(keywords, scores, title, filename):
    """ Plot the top keywords and their scores. """
    plt.figure(figsize=(10, 6))
    plt.barh(keywords, scores, color='skyblue')
    plt.xlabel('Score')
    plt.ylabel('Keyword')
    plt.title(title)
    plt.gca().invert_yaxis()
    plt.savefig(filename)
    plt.close()

# Main function to process and visualize BERT data
def analyze_bert(videos_df, comments_df):
    # Preprocess the data
    videos_df, comments_df = preprocess_data(videos_df, comments_df)

    # Load BERT model and tokenizer
    tokenizer, model = load_bert_model()

    # Extract BERT embeddings for video descriptions and comments
    video_embeddings = extract_bert_embeddings(videos_df['Cleaned_Description'].tolist(), tokenizer, model)
    comment_embeddings = extract_bert_embeddings(comments_df['Cleaned_Comment'].tolist(), tokenizer, model)

    # Extract top keywords from the embeddings and get the scores
    top_video_indices, video_keyword_scores = extract_keywords_from_embeddings(video_embeddings)
    top_comment_indices, comment_keyword_scores = extract_keywords_from_embeddings(comment_embeddings)

    # Return indices and scores
    return top_video_indices, video_keyword_scores, top_comment_indices, comment_keyword_scores


