# Apollo KB Search

## Steps

1. Get data from Zendesk, replace at `cleaner/apollo-knowledge/raw_articles.json`.
2. Clean and process by running `npm run generate:apollo-knowledge` inside `cleaner` folder.
3. Create and compute embeddings

   - Run it outside docker else it will be very slow
   - First time run:

     ```sh
     python3 -m venv .venv
     source .venv/bin/activate
     pip install -r requirements.txt
     ```

   - Everytime run:

     ```sh
     python3 api/index.py --precompute
     ```

   - TODO: Separate compute logic to another file

4. To run the server locally, copy and create `.env` from `.env.dev`. Then run `docker compose up`.

## Other

### Run commands inside docker

`docker exec -it api /bin/bash`
