
import os

# Dummy credentials so importing src.app (and its transitive ai_service /
# vector_service / db imports) never requires real Gemini / Pinecone /
# MongoDB credentials. No real network calls happen at import time;
# individual tests mock any provider call that would otherwise fire.
os.environ.setdefault("JWT_SECRET", "test-jwt-secret")
os.environ.setdefault("MONGO_URI", "mongodb://localhost:27017/test-db")
os.environ.setdefault("GEMINI_API_KEY", "test-gemini-key")
os.environ.setdefault("PINECONE_API_KEY", "test-pinecone-key")
os.environ.setdefault("PINECONE_INDEX_NAME", "test-index")

# src.sockets.socket_server imports src.services.vector_service, which calls
# pinecone.Pinecone(...).Index(...) at *module import time* to resolve the
# index host over the network. Fake the Pinecone class before src.app (and
# therefore vector_service) is ever imported, so test collection never makes
# a real Pinecone API call or touches real infrastructure.
import pinecone  # noqa: E402


class _FakePineconeIndex:
    def upsert(self, *args, **kwargs):
        return None

    def query(self, *args, **kwargs):
        return None


class _FakePinecone:
    def __init__(self, *args, **kwargs):
        pass

    def Index(self, *args, **kwargs):
        return _FakePineconeIndex()


pinecone.Pinecone = _FakePinecone
