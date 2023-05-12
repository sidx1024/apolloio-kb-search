import torch
from transformers import AutoTokenizer
from sentence_transformers import SentenceTransformer, util, InputExample, losses
from fuzzywuzzy import fuzz
from tqdm import tqdm

from config import model_name, model_dir, token_size

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet

nltk.data.path.append('.')
nltk.download('wordnet', download_dir='.')
nltk.download('stopwords', download_dir='.')
nltk.download('punkt', download_dir='.')

# model = SentenceTransformer(model_dir)


def split_content(data, max_token_size=token_size):
    tokenizer = SentenceTransformer(model_dir)
    chunks_data = []

    for article in data:
        body = article['body']
        paragraphs = body.split("\n \n")
        for paragraph in paragraphs:
            tokenized_paragraph = tokenizer.tokenize(paragraph)
            if len(tokenized_paragraph) <= max_token_size:
                article_copy = article.copy()
                article_copy['body'] = paragraph
                chunks_data.append(article_copy)
            else:
                sentences = nltk.sent_tokenize(paragraph)
                chunk = ""
                for sentence in sentences:
                    tokenized_sentence = tokenizer.tokenize(sentence)
                    if len(tokenizer.tokenize(chunk)) + len(tokenized_sentence) <= max_token_size:
                        chunk += " " + sentence
                    else:
                        if chunk:  # add the chunk to the list if it's not empty
                            article_copy = article.copy()
                            article_copy['body'] = chunk
                            chunks_data.append(article_copy)
                        chunk = sentence
                if chunk:  # add the last chunk if it's not empty
                    article_copy = article.copy()
                    article_copy['body'] = chunk
                    chunks_data.append(article_copy)

    return chunks_data


# Initialize lemmatizer and stop_words outside the function
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))


def lemmatize_and_remove_stopwords(text):
    words = nltk.word_tokenize(text)

    # Use a list comprehension and set membership check for faster filtering
    filtered_words = [lemmatizer.lemmatize(
        word) for word in words if word not in stop_words]

    return ' '.join(filtered_words)


def precompute_embeddings(preprocessed_data, model, tokenizer, device, batch_size=32):
    embeddings = []

    for batch_start in tqdm(range(0, len(preprocessed_data), batch_size), desc="Precomputing embeddings"):
        batch = preprocessed_data[batch_start:batch_start + batch_size]

        encoded_batch = tokenizer.batch_encode_plus(
            [article["body"].lower() for article in batch],
            padding="max_length",
            truncation=True,
            max_length=token_size,
            return_tensors="pt",
        )

        encoded_batch.to(device)

        with torch.no_grad():
            batch_embeddings = model(
                encoded_batch["input_ids"], attention_mask=encoded_batch["attention_mask"])[0].mean(1)

        embeddings.extend(batch_embeddings.cpu().numpy().tolist())

    return embeddings


def get_fuzzy_score(search_query, article):
    body_ratio = 0.5
    label_ratio = 0.5

    cleaned_query = lemmatize_and_remove_stopwords(search_query)
    cleaned_body = article['lemmatized_body']
    cleaned_label = article['lemmatized_labels_merged']

    body_fuzzy_score = fuzz.token_set_ratio(
        cleaned_query, cleaned_body
    ) / 100

    label_fuzzy_score = fuzz.token_set_ratio(
        cleaned_query, cleaned_label
    ) / 100

    fuzzy_score = (body_fuzzy_score * body_ratio +
                   label_fuzzy_score * label_ratio)

    return fuzzy_score
