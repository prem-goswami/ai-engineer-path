import os
import pickle
import asyncio
from pathlib import Path

# Core LangChain document parsers and slicing utilities
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


from rank_bm25 import BM25Okapi

# Module alignments from config and database layers
from config import CHUNK_SIZE, CHUNK_OVERLAP, BM25_INDEX_PATH, COLLECTION_NAME
from database import get_db_conn, get_vector_store, update_job

# 1. Define a mapping for supported types
LOADER_MAPPING = {
    ".pdf": PyPDFLoader,
    ".txt": TextLoader,
    ".docx": Docx2txtLoader,
}


def load_and_chunk_file(file_path: str, source_filename: str):
    """
    Dynamically selects the loader based on file extension.
    """
    ext = os.path.splitext(file_path)[1].lower()

    if ext not in LOADER_MAPPING:
        raise ValueError(
            f"Unsupported file type: {ext}. Supported: {list(LOADER_MAPPING.keys())}"
        )
    print(
        f"[Ingestion] Instantiating {LOADER_MAPPING[ext].__name__} for path: {file_path}"
    )

    loader = LOADER_MAPPING[ext](file_path)
    pages = loader.load()

    print(f"[Ingestion] Successfully read {len(pages)} source layout pages.")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", " ", ""],
    )

    chunks = splitter.split_documents(pages)

    # add source_filename parameter to metadata of every chunk
    for chunk in chunks:
        chunk.metadata["source_filename"] = source_filename

    return chunks


# de serialization using pickle
def load_bm25():
    """Load BM25 index from disk. Returns (None, []) if index doesn't exist."""

    if not Path(BM25_INDEX_PATH).exists():
        return None, []
    with open(BM25_INDEX_PATH, "rb") as f:
        payload = pickle.load(f)
    return payload.get("bm25"), payload.get("chunks", [])


def update_bm25(new_chunks: list):
    _, existing_chunks = load_bm25()

    latest_chunks_dict = [
        {"text": c.page_content, "metadata": c.metadata} for c in new_chunks
    ]

    combined_chunks = existing_chunks + latest_chunks_dict

    tokenized_corpus = [doc["text"].lower().split() for doc in combined_chunks]

    # Third-party sparse retrieval statistical matching implementation
    from rank_bm25 import BM25Okapi

    updated_index = BM25Okapi(tokenized_corpus)

    with open(BM25_INDEX_PATH, "wb") as f:
        pickle.dump({"bm25": updated_index, "chunks": combined_chunks}, f)

    print(
        f"BM25 storage serialization updated. Total unified tracking scope: {len(combined_chunks)} chunks."
    )


async def rebuild_bm25_from_db():
    """
    An administrative recovery framework that reads directly from the vector store table.
    Reconstructs a clean baseline index after massive document deletions.
    """
    print(
        "\n[Admin] Resynchronizing BM25 index from underlying PostgreSQL chunks table..."
    )

    async with get_db_conn() as conn:

        cursor = await conn.execute("""
            SELECT content, langchain_metadata FROM week3_rag_docs
            """)
        records = await cursor.fetchall()

    if not records:
        print(
            "[Admin] Vector table is entirely empty. Purging local stale index files."
        )
        if Path(BM25_INDEX_PATH).exists():
            os.remove(BM25_INDEX_PATH)
        return

    # row[0] is e.document (text) | row[1] is e.cmetadata (dict)
    chunks_data = [{"text": row[0], "metadata": row[1]} for row in records]

    corpus = [c["text"].lower().split() for c in chunks_data]

    bm25 = BM25Okapi(corpus)

    payload = {"bm25": bm25, "chunks": chunks_data}

    with open(BM25_INDEX_PATH, "wb") as f:
        pickle.dump(payload, f)

    print(
        f"[Admin] BM25 successfully restored from DB. Re-indexed {len(chunks_data)} text blocks."
    )


async def ingest_file(job_id: str, file_path: str, filename: str):
    """
    Coordinates the full non-blocking asynchronous document ingestion.
    Updates the transactional processing_jobs schema state dynamically at each phase.
    """
    try:
        # Phase A: Initialize operational context and flag status to processing
        print(f"[Worker] Initializing background task routine for Job ID: {job_id}")

        await update_job(job_id, status="processing")

        # Phase B: Parse and chunk document structures
        # run_in_executor offloads the heavy CPU block away from FastAPI's primary thread
        print(f"[Worker] Dispatching PyPDFLoader tokenization loop to thread pool...")
        chunks = await asyncio.get_event_loop().run_in_executor(
            None, load_and_chunk_file, file_path, filename
        )

        # Phase C: Embed vectors and upsert to pgvector table space
        # get_vectorstore relies on synchronous networking under the hood; wrap it!
        def store_chunks():
            vectorStore = get_vector_store()
            print(
                f"[Worker] Embedding and streaming {len(chunks)} chunks down to pgvector..."
            )
            # Capture the generated list of database UUID strings returned by LangChain
            generated_ids = vectorStore.add_documents(chunks)
            return generated_ids

        db_uuid_strings = await asyncio.get_event_loop().run_in_executor(
            None, store_chunks
        )

        chunk_count = len(db_uuid_strings)

        # Synchronize the primary DB tracking keys back into our in-memory chunk objects
        # This ensures that when update_bm25 saves them to disk, they match the database rows exactly.
        for chunk_obj, generated_uuid in zip(chunks, db_uuid_strings):
            chunk_obj.metadata["id"] = str(generated_uuid)

        # Phase D: Update localized BM25 sparse vocabulary models incrementally
        print(
            f"[Worker] Appending new document properties to local binary BM25 cache..."
        )

        def run_incremental_bm25():
            update_bm25(chunks)

        await asyncio.get_event_loop().run_in_executor(None, run_incremental_bm25)

        # Phase E: Finalize transaction tracking state to complete

        await update_job(job_id, "complete", chunks_created=chunk_count)

        print(f"[Worker] Success: {filename} processed into {chunk_count} chunks.")

    except Exception as e:
        # Failure Fallback: Catch any error state and preserve metrics visibility
        print(
            f"[Worker] Critical ingestion collapse triggered over target {filename}: {e}"
        )
        await update_job(job_id, status="failed", error=str(e))

    finally:
        # Phase F: Always execute OS storage cleanup to eliminate temporary byte leaks
        if Path(file_path).exists():
            print(f"[Worker] Evicting ephemeral scratch file from disk: {file_path}")
            os.remove(file_path)
