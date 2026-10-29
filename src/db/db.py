import os

from pymongo import AsyncMongoClient


client = None
database = None


async def connect_to_db():
    global client, database

    mongo_uri = os.getenv("MONGO_URI")

    if not mongo_uri:
        raise RuntimeError("MONGO_URI is not configured")

    client = AsyncMongoClient(mongo_uri)

    try:
        await client.admin.command("ping")
        database = client.get_default_database()
        print("Connected to DB!")
    except Exception:
        await client.close()
        client = None
        database = None
        raise


async def close_db():
    global client, database

    if client is not None:
        await client.close()
        client = None
        database = None
