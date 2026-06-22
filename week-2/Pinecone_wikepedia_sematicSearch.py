import os
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from openai import OpenAI
import wikipedia
import time
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

load_dotenv()
# set a user agent to bypass the wikepedia rate limiter  so that we can extract 50 articles
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

# Initialize FastAPI App instance
app = FastAPI(title="Wikipedia Semantic Search Engine API")

pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

index_name = "semantic-search-wiki"

# check if the index already exisits in pinecode if not creates a index with the following metrics
if not pc.has_index(index_name):
    print(f"Initialising index deployement: {index_name}")
    pc.create_index(
        name=index_name,
        dimension=1536,
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",  # Infrastructure host provider
            region="us-east-1",  # Target region placement
        ),
    )
    print(f"✅ Success! Index '{index_name}' created under Serverless AWS footprint.")
else:
    print(f"ℹ️ Index '{index_name}' already exists in your Pinecone project.")


# connects your code to the Pinecone index named semantic-search-wiki.
index = pc.Index("semantic-search-wiki")
# describes the status of the index
print(index.describe_index_stats())


# fetching articles from wikepedia and accumilating it in articles =[]
articles = []
for topic in topics:
    try:
        page = wikipedia.page(
            topic, auto_suggest=False
        )  # auto_suggest is set to False because we do not want wikepedia to automatically alter the topic name while fetching articles
        articles.append(
            {
                "title": page.title,
                "content": page.content,
                "url": page.url,
                "pageid": page.pageid,
            }
        )
        print(f"Loaded:{page.title}")
        time.sleep(0.5)
    except wikipedia.exceptions.DisambiguationError as e:
        print(f"Disambiguation Error for '{topic}'. Options found: {e.options[:3]}")
    except wikipedia.exceptions.PageError:
        print(f"Page Error: '{topic}' does not exist on Wikipedia.")
    except Exception as e:
        print(f"Skipped '{topic}' due to an unexpected error: {e}")

print(f"total articles loaded:{len(articles)}")


# convert the article content to smaller chuncks
def generate_overlapping_chuncks(raw_articles, chunck_size=200, overlap=50):
    processed_chuncks = []
    step_size = chunck_size - overlap

    for article in raw_articles:
        words = article["content"].split()
        word_length = len(words)
        article_chunck_count = 0
        for i in range(0, word_length, step_size):
            chuncks = words[i : i + chunck_size]
            if len(chuncks) < 30:
                continue
            chunck_text = " ".join(chuncks)

            processed_chuncks.append(
                {
                    "id": f"{article['pageid']}-chunck-{article_chunck_count}",
                    "text": chunck_text,
                    # passing metadata to pinecone so that it can produce later when needed
                    "metadata": {
                        "source_title": article["title"],
                        "source_url": article["url"],
                        "chuck_index": article_chunck_count,
                        "text": chunck_text,
                    },
                }
            )
            article_chunck_count += 1

    print(
        f"✅ Chunking Complete! Segmented 52 articles into {len(processed_chuncks)} individual dense text blocks."
    )
    return processed_chuncks


# code to test if the chuncking and overlapping is actually working
"""
test_text = " ".join([f"word{i}" for i in range(500)])
mock_articles = [
    {
        "pageid": "99999",
        "title": "Unit Test Document",
        "content": test_text,
        "url": "http://localhost/test",
    }
]
chunks = generate_overlapping_chuncks(mock_articles, chunck_size=200, overlap=50)
print(f"Last 50 words of chunk 0: {chunks[0]['text'].split()[-50:]}")
print(f"First 50 words of chunk 1: {chunks[1]['text'].split()[:50]}")
"""
chunks = generate_overlapping_chuncks(articles, chunck_size=200, overlap=50)

print("\n==================================================")
print("📊 PRODUCTION DATASET METRICS ANALYSIS")
print("==================================================")
print(f"Total Chunks Produced across 52 Articles: {len(chunks)}")


# function to pass the chuncks in batches of 100 to openai for embedding and upserting the embeddings to pinecone.
def generate_embeddings_and_upsert(chunks, index, batch_size=100):
    total_chunck_length = len(chunks)
    total_vectors_upserted = 0
    print(f"PRODUCTION UPLOAD ENGINE: {total_chunck_length} TOTAL CHUNKS READY")
    print(f"CONFIGURATION: Streaming in mega-batches of {batch_size} items...")
    print(f"==================================================")

    for i in range(0, total_chunck_length, batch_size):
        batch = chunks[i : i + batch_size]
        batch_texts = [item["text"] for item in batch]

        try:
            print(
                f"Step 1: Sending items {i} to {min(i + batch_size, total_chunck_length)} to OpenAI..."
            )

            response = client.embeddings.create(
                model="text-embedding-3-small", input=batch_texts
            )

            vectors_to_upsert = []
            for idx, embedded_data in enumerate(response.data):
                original_item = batch[idx]
                vectors_to_upsert.append(
                    {
                        "id": original_item["id"],
                        "values": embedded_data.embedding,
                        "metadata": original_item["metadata"],
                    }
                )

            print(
                f"📤 Step 2: Streaming payload block ({len(vectors_to_upsert)} vectors) to Pinecone..."
            )
            index.upsert(vectors=vectors_to_upsert, namespace="semantic-search-wiki")
            total_vectors_upserted += len(vectors_to_upsert)
            print(
                f"✅ Success: Accumulated {total_vectors_upserted}/{total_chunck_length} rows in cloud."
            )
            time.sleep(0.5)
        except Exception as e:
            print(
                f"\n Pipeline Exception occurred in batch range {i}-{i+batch_size}: {e}"
            )
            print(
                "Entering emergency sleep mode for 5 seconds before attempting next batch segment..."
            )
            time.sleep(5)
            continue
    print(f"\n==================================================")
    print(f"🎉 SUCCESS! Cloud storage deployment complete.")
    print(
        f"Total living records inside 'wikipedia-knowledge-base': {total_vectors_upserted}"
    )
    print(f"==================================================")
    print(index.describe_index_stats())


generate_embeddings_and_upsert(chunks, index)
