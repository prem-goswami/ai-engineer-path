import os
import time

# allows Python to open socket connections, send raw SQL text strings to a Postgres instance
import psycopg2
import wikipedia
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Establishes a direct TCP socket handshake with your running Docker instance (localhost:5432) using your custom validation credentials (pguser, pgpass)
conn = psycopg2.connect(
    host="localhost", port=5432, dbname="wikidb", user="pguser", password="pass"
)

# object that actually transports your raw SQL text queries to the engine, runs them,
# holds the temporary memory results, and brings them back to Python.
cursor = conn.cursor()

# Enable pgvector extension for this connection
cursor.execute("CREATE EXTENSION IF NOT EXISTS vector")
conn.commit()

# Reuse your existing functions from Day 2
wikipedia.set_user_agent(
    "WikiSemanticSearchDemo/1.0 (prem.puri@example.com) Python-Wikipedia-Package"
)

topics = [
    "Artificial intelligence",
    "Machine learning",
    "Artificial neural network",
    "Natural language processing",
    "Computer vision",
    "Python (programming language)",
    "JavaScript",
    "Linux",
    "Cloud computing",
    "Docker (software)",
    "World War II",
    "American Revolutionary War",
    "Roman Empire",
    "Ancient Egypt",
    "Cold War",
    "Albert Einstein",
    "Isaac Newton",
    "Nikola Tesla",
    "Marie Curie",
    "Stephen Hawking",
    "Photosynthesis",
    "DNA",
    "Black hole",
    "Quantum mechanics",
    "Theory of relativity",
    "Association football",
    "Tennis",
    "Olympic Games",
    "FIFA World Cup",
    "Bitcoin",
    "Stock market",
    "Inflation",
    "Venture capital",
    "Gross domestic product",
    "Pizza",
    "Sushi",
    "Coffee",
    "Chocolate",
    "Bread",
    "Amazon rainforest",
    "Climate change",
    "Earthquake",
    "Volcano",
    "Ocean",
    "Leonardo da Vinci",
    "William Shakespeare",
    "Ludwig van Beethoven",
    "Pablo Picasso",
    "The Beatles",
    "Deep learning",
    "Data science",
    "Reinforcement learning",
]

# Step 1 — Load articles
articles = []
for topic in topics:
    try:
        page = wikipedia.page(topic, auto_suggest=False)
        articles.append(
            {
                "title": page.title,
                "content": page.content,
                "url": page.url,
                "pageid": page.pageid,
            }
        )
        print(f"✓ Loaded: {page.title}")
        time.sleep(0.5)
    except Exception as e:
        print(f"✗ Skipped {topic}: {e}")

print(f"\nTotal articles: {len(articles)}")


# Step 2 — Chunk (reuse your existing function)
def generate_overlapping_chunks(raw_articles, chunk_size=200, overlap=50):
    processed_chunks = []
    step_size = chunk_size - overlap
    for article in raw_articles:
        words = article["content"].split()
        word_length = len(words)
        chunk_count = 0
        for i in range(0, word_length, step_size):
            chunk = words[i : i + chunk_size]
            if len(chunk) < 30:
                continue
            chunk_text = " ".join(chunk)
            processed_chunks.append(
                {
                    "text": chunk_text,
                    "source": article["title"],
                    "source_url": article["url"],
                    "chunk_index": chunk_count,
                }
            )
            chunk_count += 1
    print(f"✅ {len(processed_chunks)} chunks created")
    return processed_chunks


chunks = generate_overlapping_chunks(articles)


# Step 3 & 4 — Embed and insert in batches
def embed_and_insert(chunks, batch_size=100):
    total = len(chunks)
    inserted = 0

    for i in range(0, total, batch_size):
        batch = chunks[i : i + batch_size]
        texts = [chunk["text"] for chunk in batch]

        try:
            # Generate embeddings
            response = client.embeddings.create(
                model="text-embedding-3-small", input=texts
            )

            # Insert each chunk into pgvector
            for j, embedding_data in enumerate(response.data):
                chunk = batch[j]
                vector = embedding_data.embedding

                cursor.execute(
                    """
                    INSERT INTO documents 
                        (content, source, source_url, chunk_index, embedding)
                    VALUES 
                        (%s, %s, %s, %s, %s::vector)
                    """,
                    (
                        chunk["text"],
                        chunk["source"],
                        chunk["source_url"],
                        chunk["chunk_index"],
                        str(vector),  # cast list to vector string
                    ),
                )
            # str(vector): It transforms the native Python float list into a raw text string
            # formatting block that looks like '[0.14, -0.02, ...]'.

            # %s::vector forces the database engine to intercept that plain text string coming across the
            # network socket and cast it directly into a true, native vector token array inside memory storage.

            # commiting changes to the database
            conn.commit()
            inserted += len(batch)
            print(f"Progress: {inserted}/{total} inserted")
            time.sleep(0.1)

        except Exception as e:
            print(f"Error in batch {i}-{i+batch_size}: {e}")
            # undo all the changes if there is a error
            conn.rollback()
            time.sleep(5)
            continue

    print(f"\n✅ Done. {inserted} rows inserted into pgvector.")


embed_and_insert(chunks)

# Verify
cursor.execute("SELECT COUNT(*) FROM documents")
print(f"Total rows in pgvector: {cursor.fetchone()[0]}")

# closing the connection and clearing the query memory in cursor
cursor.close()
conn.close()
