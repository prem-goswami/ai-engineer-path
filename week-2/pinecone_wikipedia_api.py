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


class RAGRequest(BaseModel):
    question: str
    top_k: int = 5


class RAGResponse(BaseModel):
    question: str
    answer: str
    sources: list[SearchResult]


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


@app.post("/rag", response_model=RAGResponse)
def rag_response(request: RAGRequest):

    # retriving chunks of data from pinecone
    embeding_question = client.embeddings.create(
        model="text-embedding-3-small",
        input=request.question,
    )

    question_query = embeding_question.data[0].embedding

    pinecone_response = index.query(
        namespace="semantic-search-wiki",
        vector=question_query,
        top_k=request.top_k,
        include_metadata=True,
    )

    # Formatting context from retrived chunks

    context_parts = []
    sources = []

    for idx, match in enumerate(pinecone_response.matches, 1):
        title = match.metadata.get("source_title", "UNKNOWN")
        text = match.metadata.get("text", "")
        url = match.metadata.get("source_url", "")
        chunk_index = match.metadata.get("chuck_index", 0)
        context_parts.append(f"[Source: {idx} - {title}]: \n{text}")
        sources.append(
            SearchResult(
                rank=idx,
                score=round(match.score, 4),
                title=title,
                url=url,
                chunk_index=chunk_index,
                text=text,
            )
        )

    context = "\n\n".join(context_parts)

    prompt = f"""You are a helpful assistant. Answer the user's question using ONLY 
                the provided context below. Do not use any outside knowledge.
                Cite your sources by referencing the article title in your answer.
                If the context does not contain enough information to answer, 
                say 'I don't have enough information in my knowledge base to answer this.'
                Context:
                {context}
                Question: {request.question}
                Answer:"""

    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "system", "content": prompt}],
        temperature=0.2,
    )

    answer = completion.choices[0].message.content.strip()
    return RAGResponse(question=request.question, answer=answer, sources=sources)
