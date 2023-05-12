from dotenv import load_dotenv
from flask_httpauth import HTTPTokenAuth
from flask import Flask
import torch
import json
import numpy as np
import argparse
import traceback
from flask import Flask, request, jsonify, g
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer

from config import model_name, model_dir, token_size, env, BEARER_TOKEN
from utils import precompute_embeddings, lemmatize_and_remove_stopwords, get_fuzzy_score, split_content

load_dotenv()  # Load the environment variables from the .env file

app = Flask(__name__)
auth = HTTPTokenAuth(scheme='Bearer')

tokens = {
    BEARER_TOKEN: "user"
}

# Load the precomputed embeddings
with open("article_embeddings.json", "r") as f:
    precomputed_embeddings = json.load(f)


# Load preprocessed data
articles = './cleaner/apollo-knowledge/clean_articles.json'
preprocessed_data = split_content(json.load(open(articles, 'r')))


for article in preprocessed_data:
    # Lemmatize and remove stopwords from the article body
    article['lemmatized_body'] = lemmatize_and_remove_stopwords(
        article['body'])

    # Lemmatize and remove stopwords from the labels (array of strings)
    lemmatized_labels = []
    for label in article['labels']:
        lemmatized_label = lemmatize_and_remove_stopwords(label)
        lemmatized_labels.append(lemmatized_label)

    article['lemmatized_labels_merged'] = ' '.join(lemmatized_labels)

# Initialize tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_dir)
model = AutoModel.from_pretrained(model_dir)

# Move the model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


@auth.verify_token
def verify_token(token):
    if env == 'development':
        return True
    if token in tokens:
        g.current_user = tokens[token]
        return True
    return False


@app.route('/search', methods=['GET'])
@auth.login_required
def search():
    try:
        # Your existing search code
        user_query = request.args.get('q', '').lower()

        # Tokenize and encode the user query
        encoded_query = tokenizer.encode_plus(
            user_query,
            padding="max_length",
            truncation=True,
            max_length=token_size,
            return_tensors="pt",
        )

        encoded_query.to(device)

        # Generate embeddings for the user query
        with torch.no_grad():
            query_embedding = model(
                encoded_query["input_ids"], attention_mask=encoded_query["attention_mask"])[0].mean(1)

        # Calculate similarity scores between the query and articles
        similarity_scores = []
        query_embedding = query_embedding.cpu().numpy()

        for idx, article_embedding in enumerate(precomputed_embeddings):
            similarity = np.dot(query_embedding, np.array(article_embedding)) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(article_embedding))
            similarity_scores.append((similarity.item(), idx))

        # Sort the similarity scores in descending order
        sorted_similarity_scores = sorted(
            similarity_scores, key=lambda x: x[0], reverse=True)

        k = 8

        # Retrieve the top k matching chunks, associated articles, and similarity scores
        top_matches = sorted_similarity_scores[:k]

        results = []

        for match in top_matches:
            score, idx = match
            if score >= 0.6:
                article = preprocessed_data[idx]

                # Calculate the fuzzy search score for the user query and the matching chunk
                fuzzy_score = get_fuzzy_score(user_query, article)

                # Combine the similarity_score, fuzzy_score, and labels_score using weights
                similarity_weight = 0.3
                fuzzy_weight = 0.7

                weighted_score = (similarity_weight * score +
                                  fuzzy_weight * fuzzy_score)

                results.append({
                    'matching_chunk': article['body'],
                    'article_url': article['html_url'],
                    'title': article['title'],
                    'headings': article['headings'],
                    'labels': article['labels'],
                    'similarity_score': score,
                    'fuzzy_score': fuzzy_score,
                    'weighted_score': weighted_score,
                    'created_at': article['created_at'],
                })

        # print(user_query)
        # print(lemmatize_and_remove_stopwords(user_query))

        # Calculate fuzzy search scores for all articles
        fuzzy_scores = [(get_fuzzy_score(user_query, article), idx)
                        for idx, article in enumerate(preprocessed_data)]

        # Add fuzzy matches to the results if they are not already included
        for fuzzy_score, idx in fuzzy_scores:
            if fuzzy_score >= 0.7:
                article = preprocessed_data[idx]

                # Check if the article is already in the results
                existing_result = next(
                    (r for r in results if r['article_url'] == article['html_url']), None)

                if not existing_result:
                    fuzzy_score = get_fuzzy_score(user_query, article)
                    results.append({
                        'matching_chunk': article['body'],
                        'article_url': article['html_url'],
                        'title': article['title'],
                        'headings': article['headings'],
                        'labels': article['labels'],
                        'similarity_score': 0,
                        'fuzzy_score': fuzzy_score,
                        'weighted_score': fuzzy_score,
                        'created_at': article['created_at'],
                    })
        # Sort the results based on the weighted_score in descending order
        sorted_results = sorted(
            results, key=lambda x: x['weighted_score'], reverse=True)

        return jsonify(sorted_results)

    except Exception as e:
        print(f"An exception occurred: {e}")
        traceback.print_exc()
        return jsonify({"error": "An internal error occurred while processing your request."}), 500


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--precompute", help="Precompute article embeddings and save them to a JSON file.", action="store_true")
    parser.add_argument(
        "--download", help="Download the model", action="store_true")
    args = parser.parse_args()

    if args.precompute:
        # Precompute article embeddings
        article_embeddings = precompute_embeddings(
            preprocessed_data, model, tokenizer, device)

        # Save the embeddings to a JSON file
        with open("article_embeddings.json", "w") as f:
            json.dump(article_embeddings, f)
    elif args.download:
        # Download and save the model and tokenizer
        model = SentenceTransformer(model_name)
        model.save(model_dir)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.save_pretrained(model_dir)
    else:
        # Load the precomputed embeddings
        with open("article_embeddings.json", "r") as f:
            precomputed_embeddings = json.load(f)

        if len(precomputed_embeddings) != len(preprocessed_data):
            print(
                f"Length of precomputed_embeddings: {len(precomputed_embeddings)}")
            print(f"Length of preprocessed_data: {len(preprocessed_data)}")
            raise Exception(
                "The length of embeddings and data do not match. You might want to precompute the embeddings if you have updated the data.")

        app.run(host="0.0.0.0", port="3002")
