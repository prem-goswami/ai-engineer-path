import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from pinecone import Pinecone
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index_name = "semantic-search-wiki"

index = pc.Index("semantic-search-wiki")


app = FastAPI(title="Wikipedia Semantic Search Engine API")


class SearchRequest(BaseModel):
    query: str
    top_k: int = 5


class SearchResult(BaseModel):
    rank: int
    score: float
    title: str
    url: str
    chunk_index: int
    text: str


class SearchResponse(BaseModel):
    query: str
    results: list[SearchResult]


@app.get("/")
def health():
    status = index.describe_index_stats()
    return {
        "status": "ok",
        "total_vectors": status.total_vector_count,
        "dimensions": status.dimension,
    }


@app.post("/search", response_model=SearchResponse)
def getSearchResults(request: SearchRequest):
    query_embedding = client.embeddings.create(
        model="text-embedding-3-small", input=request.query
    )

    query_vector = query_embedding.data[0].embedding

    pinecone_response = index.query(
        namespace=index_name,
        vector=query_vector,
        top_k=request.top_k,
        include_metadata=True,
    )

    results = []
    for idx, match in enumerate(pinecone_response.matches, 1):
        results.append(
            SearchResult(
                rank=idx,
                score=round(match.score, 4),
                title=match.metadata.get("source_title", "Unknown"),
                url=match.metadata.get("source_url", ""),
                chunk_index=match.metadata.get("chuck_index", 0),
                text=match.metadata.get("text", ""),
            )
        )

    return SearchResponse(query=request.query, results=results)
