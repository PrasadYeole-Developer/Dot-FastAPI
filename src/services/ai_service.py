import os

from google import genai
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompt_values import PromptValue
from langchain_core.output_parsers import StrOutputParser
from src.schemas.ai_schemas import AIResponse

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise RuntimeError("GEMINI_API_KEY is not configured")


# LangChain Gemini Chat Model
chat_model = ChatGoogleGenerativeAI(
    model="gemini-3-flash-preview",
    google_api_key=api_key,
    temperature=0,
)

output_parser = StrOutputParser()

# Existing Google GenAI client (for embeddings)
ai = genai.Client(api_key=api_key)

structured_chat_model = chat_model.with_structured_output(AIResponse)

async def generate_response(prompt: str | PromptValue) -> str:
    response = await chat_model.ainvoke(prompt)
    return output_parser.invoke(response)

async def generate_structured_response(prompt: str) -> AIResponse:
    response = await structured_chat_model.ainvoke(prompt)
    return response

async def generate_vectors(content: str) -> list[float]:
    response = await ai.aio.models.embed_content(
        model="gemini-embedding-001",
        contents=content,
        config={"output_dimensionality": 768},
    )

    if not response.embeddings:
        raise RuntimeError("Gemini returned no embeddings")

    return response.embeddings[0].values