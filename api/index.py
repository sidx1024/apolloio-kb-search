from sentence_transformers import SentenceTransformer, util
from config import articles
import json
import traceback
from flask import Flask, request, jsonify

from config import model_dir, articles

# File paths
articles_file = articles
model_dir = "model/fine_tuned_model"
model = SentenceTransformer(model_dir)

app = Flask(__name__)

# Load the original data
with open(articles_file) as f:
    data = json.load(f)


@app.route('/search', methods=['GET'])
def search():
    try:
        # Your existing search code
        user_query = request.args.get('q', '').lower()

        results = semantic_search(user_query)

        return jsonify(results)

    except Exception as e:
        print(f"An exception occurred: {e}")
        traceback.print_exc()
        return jsonify({"error": "An internal error occurred while processing your request."}), 500


def get_matched_words(query, text):
    matched_words = []
    query_tokens = query.lower().split()
    text_tokens = text.lower().split()
    for token in query_tokens:
        if token in text_tokens:
            matched_words.append({'word': token, 'weight': 1.0})
    return matched_words


def semantic_search(user_query):
    user_query_embedding = model.encode(user_query)

    closest_results = util.semantic_search(
        user_query_embedding, model.encode([d['body'] for d in data]), top_k=3)

    closest_chunks = []
    for result in closest_results[0]:
        idx = result['corpus_id']
        similarity_score = result['score']
        if similarity_score > 0.4:
            chunk = data[idx]
            chunk['similarity_score'] = similarity_score
            closest_chunks.append(chunk)

    return closest_chunks


if __name__ == '__main__':
    app.run(host="0.0.0.0", port="3002")
