import os
import json
from pathlib import Path
from typing import List

from sentence_transformers import SentenceTransformer
import faiss
from tqdm import tqdm

# Config
DATA_DIR = Path("data")
VECTOR_DIR = Path("vectorstore")
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

VECTOR_DIR.mkdir(exist_ok=True)

model = SentenceTransformer(EMBEDDING_MODEL)


def chunk_text(
    text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP
) -> List[str]:
    chunks = []
    start = 0
    length = len(text)
    while start < length:
        end = min(start + chunk_size, length)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end == length:
            break
        start = end - overlap
    return chunks


def load_documents(data_dir: Path) -> List[dict]:
    docs = []
    for path in sorted(data_dir.glob("**/*")):
        if path.is_file() and path.suffix.lower() in (".txt", ".md"):
            text = path.read_text(encoding="utf-8", errors="ignore")
            chunks = chunk_text(text)
            for i, c in enumerate(chunks):
                docs.append(
                    {
                        "id": f"{path.name}::{i}",
                        "source": str(path),
                        "text": c,
                    }
                )
    return docs


def build_vectorstore(docs: List[dict]):
    texts = [d["text"] for d in docs]
    embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)

    # normalize for cosine-sim using inner product
    faiss.normalize_L2(embeddings)
    index.add(embeddings)

    faiss.write_index(index, str(VECTOR_DIR / "index.faiss"))
    with open(VECTOR_DIR / "meta.json", "w", encoding="utf-8") as f:
        json.dump(docs, f, ensure_ascii=False, indent=2)


def main():
    if not DATA_DIR.exists():
        print(
            f"Data directory '{DATA_DIR}' does not exist. Create it and add .txt/.md files."
        )
        return
    docs = load_documents(DATA_DIR)
    if not docs:
        print("No documents found in data directory.")
        return
    print(f"Loaded {len(docs)} chunks from documents. Building embeddings...")
    build_vectorstore(docs)
    print("Vectorstore written to vectorstore/ (index.faiss + meta.json)")


if __name__ == "__main__":
    main()