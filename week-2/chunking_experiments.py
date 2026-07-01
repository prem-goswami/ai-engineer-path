import nltk

nltk.download("punkt")
nltk.download("punkt_tab")
from nltk.tokenize import sent_tokenize


def chunk_fixed_200(text, chunk_size=200, overlap=50):
    """Your existing strategy from Day 2"""
    words = text.split()
    chunks = []
    step = chunk_size - overlap
    for i in range(0, len(words), step):
        chunk = words[i : i + chunk_size]
        if len(chunk) < 30:
            continue
        chunks.append(" ".join(chunk))
    return chunks


def chunk_fixed_100(text, chunk_size=100, overlap=20):
    """Smaller chunks — more precise embeddings, less context"""
    words = text.split()
    chunks = []
    step = chunk_size - overlap
    for i in range(0, len(words), step):
        chunk = words[i : i + chunk_size]
        if len(chunk) < 15:
            continue
        chunks.append(" ".join(chunk))
    return chunks


def chunk_sentences(text, window=3, overlap=1):
    """
    Sentence-based chunking.
    Groups 'window' sentences together with 'overlap' sentence overlap.
    """
    sentences = sent_tokenize(text)
    chunks = []
    step = window - overlap
    for i in range(0, len(sentences), step):
        group = sentences[i : i + window]
        if len(group) < 2:
            continue
        chunks.append(" ".join(group).strip())
    return chunks


# Test all three on a sample text
if __name__ == "__main__":
    sample = """
    Backpropagation is an algorithm used to train neural networks. 
    It works by computing the gradient of the loss function with respect 
    to each weight by the chain rule. The gradients are then used to 
    update the weights using gradient descent. This process repeats 
    until the network converges on a good solution. Neural networks 
    consist of layers of interconnected nodes. Each connection has a 
    weight that determines its strength. During training these weights 
    are adjusted to minimize prediction error.
    """

    c200 = chunk_fixed_200(sample)
    c100 = chunk_fixed_100(sample)
    csent = chunk_sentences(sample)

    print(f"200-word chunks: {len(c200)}")
    print(f"100-word chunks: {len(c100)}")
    print(f"Sentence chunks: {len(csent)}")
    print(f"\nSentence chunk 0: {csent[0]}")
    print(f"Sentence chunk 1: {csent[1]}")
