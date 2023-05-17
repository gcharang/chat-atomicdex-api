# Chat With the AtomicDEX API github repo

Ask questions about the [AtomicDEX API](https://github.com/KomodoPlatform/atomicDEX-API).

## Basic architecture

The app is a NextJS/Tailwind CSS frontend with a FastAPI backend. Uses Docker.

GPT-3.5 is adequate. GPT-4 performs admirably, but is expensive. Reply with continue to ask the chatbot to continue its response

## Running locally

1. Clone the repo

```bash
git clone https://github.com/gcharang/chat-atomicdex-api
cd chat-atomicdex-api
```

2. Install Node dependencies

```bash
npm ci
```

3. Set up environment variables

Copy the .env-template file to create a .env file using the following command:

```bash
cp .env-template .env
```

Update the .env file with your values:
Replace the placeholder values with your actual API keys, organization ID, and other configuration values as needed. Save the file when you're done.

4. Build and run the backend container

```bash
cd backend
docker-compose up
```

Attach a terminal to the container and do the following:


5. Embed the repository

```bash
# still in the backend/ directory
python create_vector_db.py
```

6. Eun the backend server

```bash
gunicorn main:app --bind "[::]:5555" -k uvicorn.workers.UvicornWorker --reload
```

Run the following outside the container

7. Set up environment variables for the frontend

Copy the .env-template file to create a .env file using the following command:

```bash
cp .env.local-template .env.local
```

Update the .env.local file with your values


8. Run the Node dev server

```bash
npm run dev
```

## Potential improvements

- Replace the `chat_stream` endpoint with a websocket implementation.
- Ask the model not to generatively reference its sources. Instead, simply copy the code snippet directly.
- The splitter could be improved. Right now, it's a character splitter that favors newlines, but OpenAI has implemented a similar one that splits on tokens instead.
- The embeddings and retrieval mechanisms could account for the hierarchy of AtomicDEX API github repo's code structure, like Replit's Ghostwriter does.
- Could bring in issue/PR comments to add more relevant context

### Useful docker commands

```bash
docker-compose down && docker-compose up --build -d
docker-attach llm-backend
```