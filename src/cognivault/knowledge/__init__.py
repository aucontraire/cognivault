"""Knowledge persistence & semantic retrieval.

Closes the knowledge loop: persists each pipeline run's refined question, topics,
and synthesis output to the existing database tables, generates topic embeddings,
and backs the Historian's semantic retrieval — all non-blocking and DB-optional.

See specs/002-knowledge-persistence-semantic-retrieval/.
"""
