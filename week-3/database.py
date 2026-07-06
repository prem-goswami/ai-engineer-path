import os
import uuid
import psycopg
from contextlib import asynccontextmanager
from datetime import datetime, timezone

# LangChain's modern postgres database wrappers
from langchain_postgres import PGEngine, PGVectorStore
from langchain_openai import OpenAIEmbeddings

# Pulling configurations we locked down in config.py
from config import (
    DATABASE_URL,
    PG_DSN,
    COLLECTION_NAME,
    VECTOR_SIZE,
    EMBEDDING_MODEL,
    OPENAI_API_KEY,
)

# ── PGEngine (LangChain Integration) ──────────────────────────────────
engine = PGEngine.from_connection_string(DATABASE_URL)


def init_vectorstore_table():
    engine.init_vectorstore_table(
        table_name=COLLECTION_NAME, vector_size=VECTOR_SIZE, overwrite=False
    )


def get_vector_store() -> PGVectorStore:
    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL, api_key=OPENAI_API_KEY)

    return PGVectorStore.create_sync(
        engine=engine, embeddings=embeddings, table_name=COLLECTION_NAME
    )


# ── Raw psycopg3 Connection Layer ───────────────────────────
# createa a async manager that manages connections opening and closing with the db using conn using psycopg
@asynccontextmanager
async def get_db_conn():
    async with await psycopg.AsyncConnection.connect(PG_DSN) as conn:
        try:
            yield conn
            await conn.commit()
        except Exception as e:
            await conn.rollback()
            raise


# ── Job State Tracking Table ───────────────────────────────────────
async def init_job_table():
    async with get_db_conn() as conn:
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS processing_jobs (
                job_id         TEXT PRIMARY KEY,
                filename       TEXT NOT NULL,
                status         TEXT NOT NULL DEFAULT 'pending',
                created_at     TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                completed_at   TIMESTAMPTZ,
                error          TEXT,
                chunks_created INT
            )
            """)


async def create_job(filename: str) -> str:
    job_id = str(uuid.uuid4())
    async with get_db_conn() as conn:
        await conn.execute(
            """
        INSERT INTO processing_jobs (job_id,filename,status,created_at)
        VALUES (%s, %s, 'pending', %s)
        """,
            (
                job_id,
                filename,
                datetime.now(timezone.utc),
            ),
        )
    return job_id


async def update_job(
    job_id: str, status: str, chunks_created: int = None, error: str = None
):
    async with get_db_conn() as conn:
        await conn.execute(
            """
            UPDATE processing_jobs 
            SET status = %s, completed_at = %s, chunks_created = %s, error = %s
            WHERE job_id = %s
            """,
            (
                status,
                datetime.now(timezone.utc),
                chunks_created,
                error,
                job_id,
            ),
        )


async def get_job(job_id: str):
    async with get_db_conn() as conn:
        row = await conn.execute(
            """
           SELECT * FROM processing_jobs 
           WHERE job_id = %s 
            """,
            (job_id,),
        )

        record = await row.fetchone()

    if record is None:
        return None
    cols = [
        "job_id",
        "filename",
        "status",
        "created_at",
        "completed_at",
        "error",
        "chunks_created",
    ]

    return dict(zip(cols, record))
