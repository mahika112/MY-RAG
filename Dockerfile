FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    gcc g++ git curl \
    && rm -rf /var/lib/apt/lists/*

RUN pip install uv

RUN uv pip install --system \
    fastapi uvicorn python-dotenv python-multipart \
    faiss-cpu numpy pydantic \
    langchain langchain-groq langchain-community \
    pypdf sentence-transformers \
    --extra-index-url https://download.pytorch.org/whl/cpu \
    torch

COPY . .

RUN mkdir -p data/uploaded

EXPOSE 7860

CMD python3 api.py# force rebuild
