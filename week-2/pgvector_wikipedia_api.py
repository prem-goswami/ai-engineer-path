import os
import asyncio
from fastapi import FastAPI, HTTPException
import psycopg2
from psycopg2.extras import (
    RealDictCursor,
)  # automatically map the incoming database rows into standard Python dictionaries.
from pydantic import BaseModel
from openai import OpenAI
from dotenv import load_dotenv

# langchain imports
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_postgres import PGEngine, PGVectorStore
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

if os.name == "nt":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

app = FastAPI(title="Wikipedia Semantic Search Engine API")
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# pgvector connection using langchain
CONNECTION_STRING = "postgresql+psycopg://pguser:pass@localhost:5432/wikidb"
pg_conn = psycopg2.connect(
    host="localhost", port=5432, dbname="wikidb", user="pguser", password="pass"
)

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small", api_key=os.getenv("OPENAI_API_KEY")
)

engine = PGEngine.from_connection_string(CONNECTION_STRING)

vectorstore = PGVectorStore.create_sync(
    engine=engine,
    table_name="documents",
    embedding_service=embeddings,
    content_column="content",
    embedding_column="embedding",
    id_column="id",
)

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)


class SearchRequest(BaseModel):
    query: str
    top_k: int = 5


class PGSearchResult(BaseModel):
    rank: int
    score: float
    title: str
    url: str
    chunk_index: int
    text: str


class PGSearchResponse(BaseModel):
    query: str
    results: list[PGSearchResult]


class RAGRequest(BaseModel):
    question: str
    top_k: int = 5


class RAGResponse(BaseModel):
    question: str
    answer: str
    sources: list[PGSearchResult]


rag_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are a helpful assistant. Answer the user's question using 
ONLY the provided context below. Do not use any outside knowledge.
Cite your sources by referencing the article title in your answer.
If the context does not contain enough information to answer,
say 'I don't have enough information in my knowledge base to answer this.'

Context:
{context}""",
        ),
        ("human", "{input}"),
    ]
)


@app.get("/")
def health():
    return {
        "status": "ok",
    }


@app.post("/search-pg", response_model=PGSearchResponse)
def search_pgvector(request: SearchRequest):
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    # Step 1 — Embed the query
    embedding_response = client.embeddings.create(
        model="text-embedding-3-small", input=request.query
    )
    query_vector = embedding_response.data[0].embedding

    # Step 2 — Query pgvector using cosine distance
    cursor = pg_conn.cursor(cursor_factory=RealDictCursor)
    cursor.execute(
        """
        SELECT 
            content,
            source,
            source_url,
            chunk_index,
            1 - (embedding <=> %s::vector) AS similarity_score
        FROM documents
        ORDER BY embedding <=> %s::vector
        LIMIT %s
        """,
        (str(query_vector), str(query_vector), request.top_k),
    )
    rows = cursor.fetchall()
    cursor.close()

    # Step 3 — Build results
    results = []
    for rank, row in enumerate(rows, 1):
        results.append(
            PGSearchResult(
                rank=rank,
                score=round(float(row["similarity_score"]), 4),
                title=row["source"],
                url=row["source_url"],
                chunk_index=row["chunk_index"],
                text=row["content"],
            )
        )

    return PGSearchResponse(query=request.query, results=results)


@app.post("/lcrag", response_model=RAGResponse)
def langChain_rag(request: RAGRequest):
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")
    try:
        dynamic_retriever = vectorstore.as_retriever(search_kwargs={"k": request.top_k})

        stuff_document = create_stuff_documents_chain(llm, rag_prompt)

        rag_chain = create_retrieval_chain(dynamic_retriever, stuff_document)

        result = rag_chain.invoke({"input": request.question})

        formatted_sources = []
        for idx, doc in enumerate(result.get("context", []), 1):
            metadata = doc.metadata
            formatted_sources.append(
                PGSearchResult(
                    rank=idx,
                    score=0.0,  # Standard LangChain retrievers hide cosine scores by default
                    title=metadata.get("source", "Unknown"),
                    url=metadata.get("source_url", ""),
                    chunk_index=int(metadata.get("chunk_index", 0)),
                    text=doc.page_content,
                )
            )
        return RAGResponse(
            question=request.question.strip(),
            answer=result["answer"].strip(),
            sources=formatted_sources,
        )
    except Exception as e:
        print(f"❌ LangChain execution exception caught: {e}")
        raise HTTPException(
            status_code=500, detail=f"LangChain pipeline error: {str(e)}"
        )
