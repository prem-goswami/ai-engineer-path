from sentence_transformers import CrossEncoder
import time

# Load once at module level — expensive to initialise
print("⏳ Loading cross-encoder model...")
cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
print("Cross-encoder ready.")


def rerank(query: str, candidates: list[dict], top_k: int = 3) -> list[dict]:
    """
    Takes a query and list of candidate chunks.
    Each candidate must have a 'text' field.
    Returns top_k reranked results with rerank_score added.
    """
    if not candidates:
        return []

    # Build query-document pairs for cross-encoder
    pairs = [(query, candidate["text"]) for candidate in candidates]

    # Score all pairs simultaneously — cross-encoder reads query+doc together
    scores = cross_encoder.predict(pairs)

    # Attach scores to candidates
    for i, candidate in enumerate(candidates):
        candidate["rerank_score"] = float(scores[i])

    # Sort by rerank score descending
    reranked = sorted(candidates, key=lambda x: x["rerank_score"], reverse=True)

    return reranked[:top_k]
