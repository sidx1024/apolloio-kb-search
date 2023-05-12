import os
from dotenv import load_dotenv

load_dotenv()  # Load the environment variables from the .env file

env = os.getenv('ENV')
BEARER_TOKEN = os.getenv('BEARER_TOKEN')

model_name = "sentence-transformers/all-mpnet-base-v2"
model_dir = "./model/" + model_name
token_size = 512
