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

# BM25 imports
from rank_bm25 import BM25Okapi

# rerank imports
from reranker import rerank

load_dotenv()

if os.name == "nt":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# Global memory slots that will hold our cached text structures
GLOBAL_BM25 = None
RAW_CHUNKS_LOOKUP = {}


# tokenizing and constructing the index for sparse vectors`                              `
def initialize_global_bm25_index():
    global GLOBAL_BM25, RAW_CHUNKS_LOOKUP
    print("⏳ System boot: Hydrating global BM25 keyword index from PostgreSQL...")

    # Establish a fast temporary connection to read raw text fields
    temp_conn = psycopg2.connect(
        host="localhost", port=5432, dbname="wikidb", user="pguser", password="pass"
    )
    cursor = temp_conn.cursor(cursor_factory=RealDictCursor)

    # Pull out every single chunk text payload paired with its database primary ID
    cursor.execute("SELECT id, content, source, source_url, chunk_index FROM documents")
    rows = cursor.fetchall()

    cursor.close()
    temp_conn.close()

    tokenized_corpus = []

    # Process text arrays for memory caching
    for row in rows:
        chunk_id = row["id"]

        # Populate our fast O(1) tracking dictionary
        # stores the row data in RAW_CHUNKS_LOOKUP variable so that the metadata can be accessed
        RAW_CHUNKS_LOOKUP[chunk_id] = row

        # Lowercase and split sentences into lists of words
        tokenized_chunk = row["content"].lower().split()
        tokenized_corpus.append(tokenized_chunk)

    # Build the keyword tracking statistical engine
    GLOBAL_BM25 = BM25Okapi(tokenized_corpus)
    print(f"Hybrid search engine ready: {len(rows)} documents cached in RAM.")


# Execute immediately when the app file compiles
initialize_global_bm25_index()


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


class HybridResponse(BaseModel):
    query: str
    results: list[PGSearchResult]


class RerankedResult(BaseModel):
    rank: int
    rerank_score: float
    original_rank: int
    title: str
    url: str
    chunk_index: int
    text: str


class RerankedResponse(BaseModel):
    query: str
    results: list[RerankedResult]


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


def execute_hybrid_rerank_retrieval(query: str, top_k: int) -> list[PGSearchResult]:
    # ==========================================
    # ENGINE 1: DENSE VECTOR RETRIEVAL (pgvector)
    # ==========================================
    # A. Generate the text embedding coordinates via OpenAI
    embedding_response = client.embeddings.create(
        model="text-embedding-3-small", input=query
    )
    query_vector = embedding_response.data[0].embedding

    # B. Query pgvector to get the top semantic matches based on distance
    # We ask for a broad set (top 50) so we have enough overlap to fuse rankings later!
    cursor = pg_conn.cursor(cursor_factory=RealDictCursor)

    # we only ask for id and score as all the other data is already stored in RAW_CHUNKS_LOOKUP
    cursor.execute(
        """
        SELECT 
            id,
            1 - (embedding <=> %s::vector) AS similarity_score
        FROM documents
        ORDER BY embedding <=> %s::vector
        LIMIT 50
        """,
        (str(query_vector), str(query_vector)),
    )
    semantic_rows = cursor.fetchall()
    cursor.close()

    # ==========================================
    # ENGINE 2: SPARSE KEYWORD RETRIEVAL (BM25)
    # ==========================================
    # A. Tokenize the incoming query string
    tokenized_query = query.lower().split()

    # B. Generate BM25 scores for EVERY single cached document chunk in memory
    bm25_scores = GLOBAL_BM25.get_scores(tokenized_query)

    print(f"Retrieved {len(semantic_rows)} vector matches and scored BM25.")

    # ==========================================
    # ENGINE 3: RECIPROCAL RANK FUSION (RRF) LOOP
    # ==========================================
    # Initialize a tracking dictionary: key = chunk_id, value = RRF score accumulation
    rrf_scores = {}

    # A. Score the Vector Stream candidates
    # semantic_rows contains up to 50 rows from pgvector
    for rank, row in enumerate(semantic_rows, 1):
        chunk_id = row["id"]
        # Apply the RRF fractional formula
        rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0.0) + (1.0 / (60.0 + rank))

    # B. Score the Keyword Stream candidates
    # claculate top 50 matches as bm25_scores contains a score of all the chunks in the database
    # Process the raw list of all 2947 scores into sorted (id, score) pairs
    all_keys = list(RAW_CHUNKS_LOOKUP.keys())
    all_bm25_matches = [(all_keys[idx], score) for idx, score in enumerate(bm25_scores)]
    all_bm25_matches.sort(key=lambda x: x[1], reverse=True)
    top_50_bm25 = all_bm25_matches[:50]

    for rank, (chunk_id, score) in enumerate(top_50_bm25, 1):
        # Accumulate scores. If a chunk exists in BOTH lists, its RRF score spikes!
        rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0.0) + (1.0 / (60.0 + rank))

    # C. Sort the unified tracking dict by RRF score descending
    sorted_rrf = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

    # D. Slice the final top results to match the user's requested limit
    final_top_matches = sorted_rrf[:top_k]

    # E. Reconstruct the full metadata rows from our memory vault (RAW_CHUNKS_LOOKUP)
    final_results = []
    for rank, (chunk_id, rrf_score) in enumerate(final_top_matches, 1):
        cached_row = RAW_CHUNKS_LOOKUP[chunk_id]

        final_results.append(
            PGSearchResult(
                rank=rank,
                score=round(rrf_score, 4),  # Expose the final calculated RRF metric
                title=cached_row["source"],
                url=cached_row["source_url"],
                chunk_index=cached_row["chunk_index"],
                text=cached_row["content"],
            )
        )
    return final_results


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


@app.post("/hybrid-search", response_model=HybridResponse)
def hybrid_search(request: SearchRequest):
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    retrived_hybrid_hits = execute_hybrid_rerank_retrieval(request.query, request.top_k)

    return HybridResponse(query=request.query, results=retrived_hybrid_hits)


@app.post("/hybrid-rag", response_model=RAGResponse)
def hybrid_rag_response(request: RAGRequest):
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    try:
        retrieved_hits = execute_hybrid_rerank_retrieval(
            request.question, request.top_k
        )

        # Parse out text arrays for prompt context injection
        context_chunks = []
        for hit in retrieved_hits:
            context_chunks.append(f"Document Source: {hit.title}\nContent: {hit.text}")

        combined_context_string = "\n\n---\n\n".join(context_chunks)

        # Format template and invoke OpenAI via LangChain
        hydrated_messages = rag_prompt.format_messages(
            context=combined_context_string, input=request.question
        )
        llm_response = llm.invoke(hydrated_messages)

        return RAGResponse(
            question=request.question.strip(),
            answer=llm_response.content.strip(),
            sources=retrieved_hits,
        )

    except Exception as e:
        print(f"❌ Hybrid RAG pipeline exception caught: {e}")
        raise HTTPException(
            status_code=500, detail=f"Hybrid RAG execution error: {str(e)}"
        )


@app.post("/search-reranked", response_model=RerankedResponse)
def search_reranked(request: SearchRequest):
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    # Step 1 — Retrieve top-10 from pgvector
    embedding_response = client.embeddings.create(
        model="text-embedding-3-small", input=request.query
    )
    query_vector = embedding_response.data[0].embedding

    cursor = pg_conn.cursor(cursor_factory=RealDictCursor)
    cursor.execute(
        """
        SELECT
            id,
            content,
            source,
            source_url,
            chunk_index,
            1 - (embedding <=> %s::vector) AS similarity_score
        FROM documents
        ORDER BY embedding <=> %s::vector
        LIMIT 10
        """,
        (str(query_vector), str(query_vector)),
    )
    rows = cursor.fetchall()
    cursor.close()

    # Step 2 — Format candidates for reranker
    candidates = []
    for rank, row in enumerate(rows, 1):
        candidates.append(
            {
                "original_rank": rank,  # rank before the rerank
                "title": row["source"],
                "url": row["source_url"],
                "chunk_index": row["chunk_index"],
                "text": row["content"],
                "similarity_score": float(row["similarity_score"]),
            }
        )

    # Step 3 — Rerank top-10 → top-3
    reranked = rerank(request.query, candidates, top_k=3)

    # Step 4 — Build response
    results = []
    for rank, item in enumerate(reranked, 1):
        results.append(
            RerankedResult(
                rank=rank,
                rerank_score=round(item["rerank_score"], 4),
                original_rank=item["original_rank"],
                title=item["title"],
                url=item["url"],
                chunk_index=item["chunk_index"],
                text=item["text"],
            )
        )

    return RerankedResponse(query=request.query, results=results)
