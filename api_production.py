"""
Production RAG API with Real LLM
Minimal, deployable entrypoint for Render or Docker.
Uses environment variables for secrets (HUGGINGFACE_TOKEN) and paths.
"""

import os
import time
import torch
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from transformers import AutoModelForCausalLM, AutoTokenizer

# ---------------------------------------------------------------------------
# Basic configuration
# ---------------------------------------------------------------------------
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
LLM_MODEL = os.getenv("LLM_MODEL", "google/gemma-3-4b-it")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "pubmedqa_instruction")
CHROMA_DIR = Path(os.getenv("CHROMA_DIR", "./chroma_db"))
TOP_K = int(os.getenv("TOP_K", 4))
FETCH_K = int(os.getenv("FETCH_K", 8))

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------------------------------------------------------------------
# Minimal system prompt
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = os.getenv("SYSTEM_PROMPT", "You are a careful medical question-answering assistant. Use ONLY the provided context.")

# ---------------------------------------------------------------------------
# Models and state
# ---------------------------------------------------------------------------
embeddings = None
vectorstore = None
retriever = None
tokenizer = None
model = None
rag_chain = None

# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------
class QueryRequest(BaseModel):
    question: str
    top_k: Optional[int] = TOP_K

class QueryResponse(BaseModel):
    question: str
    answer: str
    generation_time: float
    model: str
    device: str

# ---------------------------------------------------------------------------
# Initialization helpers (kept minimal)
# ---------------------------------------------------------------------------
def initialize_chromadb():
    global vectorstore, retriever, embeddings
    if vectorstore is not None:
        return vectorstore, retriever

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": DEVICE},
        encode_kwargs={"normalize_embeddings": True},
    )

    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    vectorstore = Chroma(
        persist_directory=str(CHROMA_DIR),
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
    )

    retriever = vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": TOP_K, "fetch_k": FETCH_K})
    return vectorstore, retriever


def initialize_llm():
    global tokenizer, model
    if model is not None:
        return model

    # Hugging Face token should be provided via environment variable or mounted .netrc
    hf_token = os.getenv("HUGGINGFACE_TOKEN")

    tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL, use_auth_token=hf_token)
    model = AutoModelForCausalLM.from_pretrained(LLM_MODEL, device_map="auto")
    model.eval()
    return model


def build_rag_chain():
    global rag_chain, retriever
    if rag_chain is not None:
        return rag_chain

    initialize_chromadb()
    initialize_llm()

    def format_docs(docs):
        parts = []
        for i, doc in enumerate(docs, 1):
            source = doc.metadata.get("source_dataset", "unknown")
            parts.append(f"[Chunk {i} | source={source}]\n{doc.page_content.strip()}")
        return "\n\n".join(parts)

    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", "Question: {question}\n\nRetrieved context:\n{context}"),
    ])

    chain = (
        {
            "context": retriever | RunnableLambda(format_docs),
            "question": RunnablePassthrough(),
        }
        | prompt
        | RunnableLambda(lambda x: "")
        | StrOutputParser()
    )

    rag_chain = chain
    return rag_chain

# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(title="RAG API (minimal)")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"], allow_credentials=True)

@app.get("/health")
async def health():
    return {"status": "healthy", "device": DEVICE}

@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    if not request.question or not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    start = time.time()
    try:
        chain = build_rag_chain()
        # We use chain.invoke for compatibility with the original pipeline
        answer = chain.invoke(request.question)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    gen_time = time.time() - start
    return QueryResponse(question=request.question, answer=answer or "I do not have enough information in the provided text to answer that.", generation_time=round(gen_time, 2), model=LLM_MODEL, device=DEVICE)

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    host = os.getenv("HOST", "0.0.0.0")
    uvicorn.run(app, host=host, port=port)
