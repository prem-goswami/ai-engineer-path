import os
from dotenv import load_dotenv
from openai import OpenAI
import numpy as np

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

sentences = [
    # Cooking
    "Searing a thick ribeye steak in a smoking hot cast iron skillet.",
    "Whisking egg yolks and butter over low heat to emulsify a rich hollandaise sauce.",
    "Baking a loaf of artisanal sourdough bread requires precise hydration and patience.",
    "I need to prepare a quick dinner because I am completely starving tonight.",
    "The chef chopped fresh basil, garlic, and tomatoes for the pasta sauce.",
    # Sports
    "He sprinted down the pitch and executed a flawless scissor kick into the top corner of the net.",
    "Hitting a clean fade off the tee requires adjusting your stance and clubface alignment.",
    "Consistency in your daily gym split is the absolute key to building lean muscle mass.",
    "The referee blew the final whistle, sealing a dramatic comeback victory in extra time.",
    "Running five miles every morning significantly improves cardiovascular stamina and lung capacity.",
    # Technology
    "Deploying a fine-tuned deep learning model to an AWS EC2 instance using Docker.",
    "Building an agentic RAG pipeline to query vector databases with low latency.",
    "The software engineer spent hours debugging a memory leak in the asynchronous backend.",
    "A python script can automate data extraction from structured web APIs cleanly.",
    "Git merge conflicts often happen when multiple developers modify the same line of code.",
    # Finance
    "The Federal Reserve decided to hike benchmark interest rates to combat stubborn inflation.",
    "Diversifying your portfolio with index funds mitigates long-term stock market volatility.",
    "The company's Q2 earnings report surpassed Wall Street expectations, causing the stock to surge.",
    "Venture capital firms are pouring billions of dollars into early-stage artificial intelligence startups.",
    "Understanding your net cash flow is crucial for maintaining business liquidity.",
]

print(f"Fetching embeddings for {len(sentences)} sentences...")

response = client.embeddings.create(model="text-embedding-3-small", input=sentences)

print(response)

embeddings = np.array([data.embedding for data in response.data])
# print("embeddings :", embeddings)

magnitudes = np.linalg.norm(embeddings, axis=1, keepdims=True)
normalized_embeddings = embeddings / magnitudes
# print("normalized_embeddings : ", normalized_embeddings)

similarity_matrix = np.dot(normalized_embeddings, normalized_embeddings.T)

pairs = []
sentence_length = len(sentences)

for i in range(sentence_length):
    for j in range(i + 1, sentence_length):
        pairs.append(
            {
                "index_1": i,
                "index_2": j,
                "sentence_1": sentences[i],
                "sentence_2": sentences[j],
                "similarity": similarity_matrix[i, j],
            }
        )

print("pairs : ", pairs[0])
pairs.sort(key=lambda x: x["similarity"], reverse=True)

print("\n=== TOP 10 MOST SIMILAR PAIRS ===")
for idx, pair in enumerate(pairs[:10], 1):
    print(f"\n{idx}. Similarity Score: {pair['similarity']:.4f}")
    print(f"   [1] {pair['sentence_1']}")
    print(f"   [2] {pair['sentence_2']}")


def semantic_search(query, sentences, embedding):
    query_response = client.embeddings.create(
        model="text-embedding-3-small", input=query
    )
    query_embedding = np.array(query_response.data[0].embedding)

    normalized_query = query_embedding / np.linalg.norm(query_embedding)

    scores = np.dot(embedding, normalized_query)

    best_idx = np.argmax(scores)

    print(f"\n=== SEMANTIC SEARCH ===")
    print(f"Query: {query}")
    print(f"\nBest match (score: {scores[best_idx]:.4f}):")
    print(f"{sentences[best_idx]}")
    print(f"\nTop 3 matches:")
    top3 = np.argsort(scores)[::-1][:3]
    for rank, idx in enumerate(top3, 1):
        print(f"{rank}. [{scores[idx]:.4f}] {sentences[idx]}")


semantic_search(
    "I want to learn how to cook Italian food", sentences, normalized_embeddings
)

semantic_search(
    "How do I grow my savings and invest wisely", sentences, normalized_embeddings
)
