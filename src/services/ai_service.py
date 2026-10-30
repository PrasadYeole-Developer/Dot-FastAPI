
import os

from google import genai


api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise RuntimeError("GEMINI_API_KEY is not configured")

ai = genai.Client(api_key=api_key)


async def generate_response(prompt: str) -> str:
    response = await ai.aio.models.generate_content(
        model="gemini-3-flash-preview",
        contents=prompt,
    )

    return response.text or ""


async def generate_vectors(content: str) -> list[float]:
    response = await ai.aio.models.embed_content(
        model="gemini-embedding-001",
        contents=content,
        config={
            "output_dimensionality": 768,
        },
    )

    if not response.embeddings:
        raise RuntimeError("Gemini returned no embeddings")

    return response.embeddings[0].values
