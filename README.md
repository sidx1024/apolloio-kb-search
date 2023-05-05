# Apollo KB Search

## Steps

1. Get data from Zendesk, replace at `cleaner/apollo-knowledge/raw_articles.json`.
2. Clean and process by running `npm run generate:apollo-knowledge` inside `cleaner` folder.
3. Inside docker, create and compute embeddings with `python3 api/index.py --precompute`.
   - TODO: Separate compute logic to another file
4. To run the server locally, copy and create `.env` from `.env.dev`. Then run `docker compose up`.

## Other

### Run commands inside docker

`docker exec -it api /bin/bash`
