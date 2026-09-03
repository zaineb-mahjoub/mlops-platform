from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer


EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

CHROMA_PATH = Path(__file__).resolve().parents[1] / "data" / "chroma"


_embedding_model = None
_chroma_client = None
_collection = None


def get_embedding_model():
    global _embedding_model

    if _embedding_model is None:
        _embedding_model = SentenceTransformer(EMBEDDING_MODEL)

    return _embedding_model


def get_collection():
    global _chroma_client
    global _collection

    if _collection is None:
        CHROMA_PATH.mkdir(parents=True, exist_ok=True)

        _chroma_client = chromadb.PersistentClient(
            path=str(CHROMA_PATH)
        )

        _collection = _chroma_client.get_or_create_collection(
            name="project_documents",
            configuration={
                "hnsw": {
                    "space": "cosine"
                }
            }
        )

    return _collection


def create_embeddings(texts: list[str]) -> list[list[float]]:
    model = get_embedding_model()

    embeddings = model.encode(
        texts,
        normalize_embeddings=True
    )

    return embeddings.tolist()


def store_document_chunks(
    document_id: str,
    chunks: list[str]
):
    collection = get_collection()

    embeddings = create_embeddings(chunks)

    ids = [
        f"{document_id}_{i}"
        for i in range(len(chunks))
    ]

    collection.upsert(
        ids=ids,
        documents=chunks,
        embeddings=embeddings,
        metadatas=[
            {
                "document_id": document_id,
                "chunk_index": i
            }
            for i in range(len(chunks))
        ]
    )


def search_similar_chunks(
    question: str,
    document_id: str,
    top_k: int = 3
) -> list[str]:

    collection = get_collection()

    query_embedding = create_embeddings([question])[0]

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        where={
            "document_id": document_id
        }
    )

    documents = results.get("documents", [[]])

    if not documents:
        return []

    return documents[0]
