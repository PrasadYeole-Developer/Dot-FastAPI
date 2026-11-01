
from contextlib import asynccontextmanager

import socketio
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.db.db import connect_to_db, close_db
from src.routes.auth_routes import router as auth_router
from src.routes.chat_routes import router as chat_router
from src.routes.chunking_routes import router as chunking_router
from src.routes.message_routes import router as message_router
from src.sockets.socket_server import sio


@asynccontextmanager
async def lifespan(app: FastAPI):
    await connect_to_db()
    try:
        yield
    finally:
        await close_db()


app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://localhost:4173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth_router, prefix="/api/auth", tags=["Authentication"])
app.include_router(chat_router, prefix="/api/chat", tags=["Chats"])
app.include_router(message_router, prefix="/api/messages", tags=["Messages"])
app.include_router(chunking_router, prefix="/api/chunking", tags=["Chunking"])


@app.get("/")
async def home():
    return {"message": "Chatbot Python backend is running!"}


socket_app = socketio.ASGIApp(
    sio,
    other_asgi_app=app,
)
