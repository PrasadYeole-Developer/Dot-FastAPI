
import asyncio
import os

from pinecone import Pinecone


api_key = os.getenv("PINECONE_API_KEY")
index_name = os.getenv("PINECONE_INDEX_NAME")

if not api_key:
    raise RuntimeError("PINECONE_API_KEY is not configured")

if not index_name:
    raise RuntimeError("PINECONE_INDEX_NAME is not configured")

pc = Pinecone(api_key=api_key)
index = pc.Index(index_name)


async def create_memory(
    vectors: list[float],
    metadata: dict,
    message_id: str,
):
    await asyncio.to_thread(
        index.upsert,
        vectors=[
            {
                "id": message_id,
                "values": vectors,
                "metadata": metadata,
            }
        ],
    )


async def query_memory(
    query_vector: list[float],
    limit: int = 5,
    metadata: dict | None = None,
):
    filter_expression = (
        {
            key: {"$eq": str(value)}
            for key, value in metadata.items()
        }
        if metadata
        else None
    )

    results = await asyncio.to_thread(
        index.query,
        vector=query_vector,
        top_k=limit,
        filter=filter_expression,
        include_metadata=True,
    )

    return results.matches
