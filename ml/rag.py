from groq import Groq
from dotenv import load_dotenv

load_dotenv()
from vector_store import (
    store_document_chunks,
    search_similar_chunks
)


RAG_MODEL = "qwen/qwen3.6-27b"


def chunk_text(
    text: str,
    chunk_size: int = 120,
    overlap: int = 30
) -> list[str]:

    words = text.split()

    chunks = []

    start = 0

    while start < len(words):

        end = start + chunk_size

        chunk = " ".join(words[start:end])

        if chunk.strip():
            chunks.append(chunk)

        start += chunk_size - overlap

    return chunks


def index_document(
    document_id: str,
    raw_text: str
):

    chunks = chunk_text(raw_text)

    if not chunks:
        raise ValueError("Document contains no text.")

    store_document_chunks(
        document_id,
        chunks
    )

    return {
        "document_id": document_id,
        "chunks_indexed": len(chunks)
    }


def generate_answer(
    question: str,
    relevant_chunks: list[str]
) -> str:

    if not relevant_chunks:
        return (
            "The document does not contain enough information "
            "to answer this question."
        )

    context = "\n\n".join(relevant_chunks)

    client = Groq()

    completion = client.chat.completions.create(
        model=RAG_MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a document question-answering assistant. "
                    "Answer ONLY using the provided document context. "
                    "Do not invent information. "
                    "If the answer is not present in the context, "
                    "clearly say that the document does not provide "
                    "the answer."
                )
            },
            {
                "role": "user",
                "content": (
                    f"Document context:\n\n"
                    f"{context}\n\n"
                    f"Question:\n{question}"
                )
            }
        ],
        temperature=0
    )

    return completion.choices[0].message.content


def answer_question(
    document_id: str,
    question: str
) -> str:

    relevant_chunks = search_similar_chunks(
        question=question,
        document_id=document_id,
        top_k=3
    )

    return generate_answer(
        question,
        relevant_chunks
    )
