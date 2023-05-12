import os
from dotenv import load_dotenv

load_dotenv()  # Load the environment variables from the .env file

env = os.getenv('ENV')
BEARER_TOKEN = os.getenv('BEARER_TOKEN')

base_model = "sentence-transformers/all-mpnet-base-v2"
base_model_dir = "./model/" + base_model

model = "fine-tuned-model"
model_dir = "./model/" + model

articles = './cleaner/apollo-knowledge/clean_articles.json'
