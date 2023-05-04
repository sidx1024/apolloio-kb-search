import torch
import json
import numpy as np
import argparse
from flask import Flask, request, jsonify
from transformers import AutoTokenizer, AutoModel
from fuzzywuzzy import fuzz
from tqdm import tqdm

from flask import Flask

app = Flask(__name__)

def precompute_embeddings(preprocessed_data, model, tokenizer, device, batch_size=32):
    embeddings = []

    for batch_start in tqdm(range(0, len(preprocessed_data), batch_size), desc="Precomputing embeddings"):
        batch = preprocessed_data[batch_start:batch_start + batch_size]

        encoded_batch = tokenizer.batch_encode_plus(
            [article["body"].lower() for article in batch],
            padding="max_length",
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )

        encoded_batch.to(device)

        with torch.no_grad():
            batch_embeddings = model(encoded_batch["input_ids"], attention_mask=encoded_batch["attention_mask"])[0].mean(1)

        embeddings.extend(batch_embeddings.cpu().numpy().tolist())

    return embeddings

# Load the precomputed embeddings
with open("article_embeddings.json", "r") as f:
    precomputed_embeddings = json.load(f)

# Load preprocessed data
articles = './cleaner/clean_articles.json'
preprocessed_data = json.load(open(articles, 'r'))

# Initialize tokenizer and model
model_name = "sentence-transformers/paraphrase-distilroberta-base-v2"
model_name = "sentence-transformers/all-mpnet-base-v2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# Move the model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

@app.route('/search', methods=['GET'])
def search():
    try:
        # Your existing search code
        user_query = request.args.get('q', '').lower()

        # Tokenize and encode the user query
        encoded_query = tokenizer.encode_plus(
            user_query,
            padding="max_length",
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )

        encoded_query.to(device)

        # Generate embeddings for the user query
        with torch.no_grad():
            query_embedding = model(encoded_query["input_ids"], attention_mask=encoded_query["attention_mask"])[0].mean(1)

        # Calculate similarity scores between the query and articles
        similarity_scores = []
        query_embedding = query_embedding.cpu().numpy()

        for idx, article_embedding in enumerate(precomputed_embeddings):
            similarity = np.dot(query_embedding, np.array(article_embedding)) / (np.linalg.norm(query_embedding) * np.linalg.norm(article_embedding))
            similarity_scores.append((similarity.item(), idx))

        # Sort the similarity scores in descending order
        sorted_similarity_scores = sorted(similarity_scores, key=lambda x: x[0], reverse=True)

        k = 8

        # Retrieve the top k matching chunks, associated articles, and similarity scores
        top_matches = sorted_similarity_scores[:k]

        results = []
        for match in top_matches:
            score, idx = match
            if score >= 0.6:
                article = preprocessed_data[idx]

                # Calculate the fuzzy search score for the user query and the matching chunk
                fuzzy_score = fuzz.token_set_ratio(user_query, article['body']) / 100

                # Combine the similarity_score and fuzzy_score using weights
                weighted_score = 0.3 * score + 0.7 * fuzzy_score

                results.append({
                    'matching_chunk': article['body'],
                    'article_url': article['html_url'],
                    'headings': article['headings'],
                    'similarity_score': score,
                    'fuzzy_score': fuzzy_score,
                    'weighted_score': weighted_score
                })

        # Calculate fuzzy search scores for all articles
        fuzzy_scores = [(fuzz.token_set_ratio(user_query, article['body']) / 100, idx) for idx, article in enumerate(preprocessed_data)]

        # Add fuzzy matches to the results if they are not already included
        for fuzzy_score, idx in fuzzy_scores:
            if fuzzy_score >= 0.8:
                article = preprocessed_data[idx]

                # Check if the article is already in the results
                existing_result = next((r for r in results if r['article_url'] == article['html_url']), None)

                if not existing_result:
                    results.append({
                        'matching_chunk': article['body'],
                        'article_url': article['html_url'],
                        'headings': article['headings'],
                        'similarity_score': 0,
                        'fuzzy_score': fuzzy_score,
                        'weighted_score': 0.7 * fuzzy_score
                    })

        # Sort the results based on the weighted_score in descending order
        sorted_results = sorted(results, key=lambda x: x['weighted_score'], reverse=True)

        return jsonify(sorted_results)

    except Exception as e:
        print(f"An exception occurred: {e}")
        return jsonify({"error": "An internal error occurred while processing your request."}), 500

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--precompute", help="Precompute article embeddings and save them to a JSON file.", action="store_true")
    args = parser.parse_args()

    # Load preprocessed data
    articles = './cleaner/clean_articles.json'
    preprocessed_data = json.load(open(articles, 'r'))

    # Initialize tokenizer and model
    model_name = "sentence-transformers/paraphrase-distilroberta-base-v2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)

    # Move the model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    if args.precompute:
        # Precompute article embeddings
        article_embeddings = precompute_embeddings(preprocessed_data, model, tokenizer, device)

        # Save the embeddings to a JSON file
        with open("article_embeddings.json", "w") as f:
            json.dump(article_embeddings, f)
    else:
        # Load the precomputed embeddings
        with open("article_embeddings.json", "r") as f:
            precomputed_embeddings = json.load(f)

        app.run()