from sentence_transformers import SentenceTransformer, util, InputExample, losses
from torch.utils.data import DataLoader
from config import base_model_dir, model_dir, articles
import json
import os.path

# File paths
articles_file = articles

# Load or train the model
if os.path.exists(model_dir):
    # Load existing model
    model = SentenceTransformer(model_dir)
else:
    # Load data from JSON file
    with open(articles_file) as f:
        data = (json.load(f))

    # Define the training procedure
    model = SentenceTransformer(base_model_dir)

    # Create InputExamples, with a target similarity of 1.0 for each pair
    examples = []
    for d in data:
        text_a = d['body']
        text_b = ' '.join(d['labels']) + ' ' + \
            ' '.join(d['headings'].split(' > '))
        similarity_score = util.cos_sim(model.encode(
            [text_a]), model.encode([text_b]))[0][0]
        examples.append(InputExample(
            texts=[text_a, text_b], label=similarity_score))

    # Create a DataLoader with your training examples
    train_dataloader = DataLoader(examples, shuffle=True, batch_size=24)

    train_loss = losses.CosineSimilarityLoss(model)

    # Train the model
    model.fit(train_objectives=[
              (train_dataloader, train_loss)], epochs=2, warmup_steps=100)

    # Save the model
    model.save(model_dir)

# Perform testing using the trained model
test_sentences = ["sales navigator linkedin does not work",
                  "warm up emails", "i opened the email i sent and it shows as opened", "enter a valid email while linking mailbox sid--@gmail.com", "how to resume a contact", "how many email accounts can i add", "phone trigger condition"]

# Encode the test sentences
test_embeddings = model.encode(test_sentences)

# Load the original data
with open(articles_file) as f:
    data = json.load(f)

# Find the closest match for each test sentence
for i, test_embedding in enumerate(test_embeddings):
    closest_indices = [idx['corpus_id'] for idx in util.semantic_search(
        test_embedding, model.encode([d['body'] for d in data]))[0]]
    closest_chunks = [data[idx] for idx in closest_indices][:1]

    print("======================================================================================")
    print("Test Sentence:", test_sentences[i])
    for chunk in closest_chunks:
        similarity_score = util.cos_sim(
            test_embedding, model.encode([chunk['body']]))[0][0]
        print("Similarity Score:", similarity_score)
        print("Title:", chunk['title'])
        print("Body:", chunk['body'])
        print()
