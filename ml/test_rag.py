from rag import index_document, answer_question


DOCUMENT_ID = "sample_request"


TEXT = """
We need a web platform for managing customer orders,
with Stripe payment integration and PostgreSQL database.
The system should send real-time notifications when orders
are placed. Deadline: 45 days.
"""


print("INDEXING DOCUMENT...")

result = index_document(
    DOCUMENT_ID,
    TEXT
)

print(result)


print("\nASKING QUESTION...")

answer = answer_question(
    DOCUMENT_ID,
    "What programming language should be used?"
)

print("\nANSWER:")
print(answer)
