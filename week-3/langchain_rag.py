import os
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_postgres import PGVectorStore, PGEngine
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

# Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
CONNECTION_STRING = os.getenv("DATABASE_URL")

COLLECTION_NAME = "week3_rag_docs"
EMBEDDING_MODEL = "text-embedding-3-small"
LLM_MODEL = "gpt-4o-mini"

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200


def load_and_chunk_pdf(pdf_path: str) -> list:
    """Load a PDF and split into chunks using RecursiveCharacterTextSplitter"""

    # Load PDF - each page becomes a Document object with page_content and metadata
    print(f"Loading PDF: {pdf_path}")
    loader = PyPDFLoader(pdf_path)
    pages = loader.load()
    print(f"Loaded {len(pages)} pages")

    # Split into chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=[
            "\n\n",
            "\n",
            " ",
            "",
        ],  # explicit so you remember the priority order
    )

    chunks = splitter.split_documents(pages)
    print(f"Split into {len(chunks)} chunks")

    # Inspect first chunk so you can see what a Document looks like
    print(f"\n--- First chunk ---")
    print(f"Content: {chunks[0].page_content[:200]}...")
    print(f"Metadata: {chunks[0].metadata}")

    return chunks


def store_in_pgvector(chunks: list) -> PGVectorStore:
    """Embed chunks and store in pgvector using LangChain's PGVector wrapper"""

    print("\nInitializing embeddings and vector store...")

    embeddings = OpenAIEmbeddings(
        model=EMBEDDING_MODEL, openai_api_key=os.getenv("OPENAI_API_KEY")
    )

    # psycopg3 driver - note postgresql+psycopg not psycopg2
    CONNECTION_STRING = "postgresql+psycopg://pguser:pass@localhost:5432/wikidb"

    # PGEngine manages the connection pool
    engine = PGEngine.from_connection_string(url=CONNECTION_STRING)

    # Create the table if it doesn't exist - text-embedding-3-small outputs 1536 dims
    engine.init_vectorstore_table(
        table_name=COLLECTION_NAME,
        vector_size=1536,
    )

    # Create the vector store combining the engine, table and embeddings service together in a vector store
    vector_store = PGVectorStore.create_sync(
        engine=engine,
        table_name=COLLECTION_NAME,
        embedding_service=embeddings,
    )

    print(f"Storing {len(chunks)} chunks in table: {COLLECTION_NAME}")
    # It automatically loops through every text chunk, fires it off to OpenAI to get its math vector,
    # and runs an INSERT statement to save both the raw text string and its geometric vector into PostgreSQL.
    vector_store.add_documents(chunks)
    print("Storage complete")

    return vector_store


def build_rag_chain(vector_store: PGVectorStore):
    """Build a RAG chain using LCEL pipes"""

    # Convert vector store to retriever
    retriever = vector_store.as_retriever(
        search_type="similarity", search_kwargs={"k": 4}
    )

    # LLM
    llm = ChatOpenAI(
        model=LLM_MODEL, temperature=0, openai_api_key=os.getenv("OPENAI_API_KEY")
    )

    # Prompt - {context} gets filled by retrieved docs, {input} is the user question
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are a helpful assistant. Answer the user's question 
        using only the context provided below. If the answer is not in the context, 
        say you don't know.
        
        Context: {context}""",
            ),
            ("human", "{input}"),
        ]
    )

    # Two LCEL chains piped together
    # Chain 1: stuffs retrieved docs into the prompt + calls LLM
    stuff_chain = create_stuff_documents_chain(llm=llm, prompt=prompt)

    # Chain 2: wires retriever -> stuff_chain end to end
    rag_chain = create_retrieval_chain(
        retriever=retriever, combine_docs_chain=stuff_chain
    )

    return rag_chain


def query(rag_chain, question: str) -> dict:
    """Run a question through the RAG chain and print results"""

    print(f"\nQuestion: {question}")
    response = rag_chain.invoke({"input": question})

    print(f"Answer: {response['answer']}")
    print(f"\nSources used ({len(response['context'])} chunks):")
    for i, doc in enumerate(response["context"]):
        print(
            f"  [{i+1}] Page {doc.metadata.get('page', '?')} — {doc.page_content[:100]}..."
        )

    return response


def main():
    import sys
    import asyncio

    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    # --- CONFIG ---
    PDF_PATH = (
        "Test_content.pdf"  # drop any PDF in your week3 folder and put the name here
    )
    try:

        # --- STEP 1: Load and chunk ---
        chunks = load_and_chunk_pdf(PDF_PATH)

        # --- STEP 2: Embed and store ---
        vector_store = store_in_pgvector(chunks)

        # --- STEP 3: Build RAG chain ---
        rag_chain = build_rag_chain(vector_store)

        # --- STEP 4: Query ---
        test_questions = [
            "What is the main topic of this document?",
            "Summarize the key points.",
            "What conclusions does the document reach?",
        ]

        for question in test_questions:
            query(rag_chain, question)
            print("\n" + "=" * 60)
    except Exception as e:
        print(f"\n❌ Pipeline Execution Failure Error Context: {e}")


if __name__ == "__main__":
    main()
