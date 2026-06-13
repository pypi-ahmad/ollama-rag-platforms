# Model Operations Notes

The default local embedding model for retrieval is **embeddinggemma:latest**.

The response cache TTL for repeated prompts is **18 hours**.

At runtime, the assistant uses a maximum of **4 retrieved context chunks** per answer.

Evaluation compares baseline (no retrieval) against RAG retrieval with keyword recall.
