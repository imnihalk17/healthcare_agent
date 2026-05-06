"""Render-friendly RAG API.

Startup stays light so Render can bind a port before any heavy ML imports.
Embedding/model packages are imported only when a request actually needs them.
"""

from __future__ import annotations

import os
import re
import time
from pathlib import Path
from typing import Optional

import requests
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
LLM_MODEL = os.getenv("LLM_MODEL", "google/gemma-3-4b-it")
HF_REMOTE_MODEL = os.getenv("HF_REMOTE_MODEL", LLM_MODEL)
HF_MODEL_POLICY = os.getenv("HF_MODEL_POLICY", "fastest")
HF_ROUTER_URL = os.getenv("HF_ROUTER_URL", "https://router.huggingface.co/v1/chat/completions")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "pubmedqa_instruction")
CHROMA_DIR = Path(os.getenv("CHROMA_DIR", "./chroma_db"))
TOP_K = int(os.getenv("TOP_K", "4"))
FETCH_K = int(os.getenv("FETCH_K", "8"))
REMOTE_TIMEOUT = float(os.getenv("REMOTE_TIMEOUT", "45"))
USE_REMOTE_LLM = os.getenv("USE_REMOTE_LLM", "1").lower() in {"1", "true", "yes"}

DEVICE = os.getenv("DEVICE", "cpu")

SYSTEM_PROMPT = os.getenv(
    "SYSTEM_PROMPT",
    "You are a careful medical question-answering assistant. Use only the provided context.",
)

SAMPLE_DOCUMENTS = [
    {
        "content": "The common cold often causes runny nose, sore throat, cough, and mild fatigue.",
        "metadata": {"source_dataset": "sample", "topic": "cold"},
    },
    {
        "content": "Fever is a temporary increase in body temperature and can occur with infections.",
        "metadata": {"source_dataset": "sample", "topic": "fever"},
    },
    {
        "content": "Dehydration can cause thirst, dry mouth, dizziness, and reduced urination.",
        "metadata": {"source_dataset": "sample", "topic": "dehydration"},
    },
]

embeddings = None
vectorstore = None
retriever = None
tokenizer = None
model = None


class QueryRequest(BaseModel):
    question: str
    top_k: Optional[int] = TOP_K


class QueryResponse(BaseModel):
    question: str
    answer: str
    generation_time: float
    model: str
    device: str


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip()


def _looks_copied(answer: str, docs) -> bool:
    answer_norm = _normalize(answer)
    if not answer_norm:
        return True
    context_norm = _normalize(" ".join(doc.page_content for doc in docs))
    if any(marker in answer_norm for marker in ("context:", "instruction:", "answer:")):
        return True
    return len(answer_norm) > 120 and answer_norm in context_norm


def _format_docs(docs) -> str:
    parts = []
    for index, doc in enumerate(docs, start=1):
        source = doc.metadata.get("source_dataset", "unknown")
        parts.append(f"[Chunk {index} | source={source}]\n{doc.page_content.strip()}")
    return "\n\n".join(parts)


def _extractive_fallback(question: str, docs) -> str:
    if not docs:
        return "I do not have enough information in the provided text to answer that."

    question_tokens = {token for token in re.findall(r"[a-zA-Z]+", question.lower()) if len(token) > 2}
    for doc in docs:
        sentences = re.split(r"(?<=[.!?])\s+", doc.page_content.strip())
        for sentence in sentences:
            sentence_norm = sentence.lower()
            if any(token in sentence_norm for token in question_tokens):
                return sentence.strip()

    return docs[0].page_content.strip()


def initialize_chromadb():
    global vectorstore, retriever, embeddings
    if vectorstore is not None:
        return vectorstore, retriever

    from langchain_core.documents import Document
    from langchain_community.vectorstores import Chroma
    from langchain_huggingface import HuggingFaceEmbeddings

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": DEVICE},
        encode_kwargs={"normalize_embeddings": True},
    )

    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    if any(CHROMA_DIR.iterdir()):
        vectorstore = Chroma(
            persist_directory=str(CHROMA_DIR),
            collection_name=COLLECTION_NAME,
            embedding_function=embeddings,
        )
    else:
        documents = [Document(page_content=item["content"], metadata=item["metadata"]) for item in SAMPLE_DOCUMENTS]
        vectorstore = Chroma.from_documents(
            documents=documents,
            embedding=embeddings,
            persist_directory=str(CHROMA_DIR),
            collection_name=COLLECTION_NAME,
        )
        if hasattr(vectorstore, "persist"):
            vectorstore.persist()

    retriever = vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": TOP_K, "fetch_k": FETCH_K})
    return vectorstore, retriever


def retrieve_docs(question: str, top_k: Optional[int] = None):
    vectorstore, current_retriever = initialize_chromadb()
    if top_k is None or top_k == TOP_K:
        return current_retriever.invoke(question)

    retriever_override = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": top_k, "fetch_k": max(top_k * 2, FETCH_K)},
    )
    return retriever_override.invoke(question)


def initialize_local_llm():
    global tokenizer, model
    if model is not None:
        return model

    from transformers import AutoModelForCausalLM, AutoTokenizer

    hf_token = os.getenv("HUGGINGFACE_TOKEN") or os.getenv("HF_TOKEN")
    tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL, token=hf_token)
    model = AutoModelForCausalLM.from_pretrained(LLM_MODEL, device_map="auto")
    model.eval()
    return model


def generate_remote_answer(question: str, context: str) -> str:
    hf_token = os.getenv("HUGGINGFACE_TOKEN") or os.getenv("HF_TOKEN")
    if not hf_token:
        return ""

    prompt_text = (
        f"{SYSTEM_PROMPT}\n\n"
        f"Question: {question}\n\n"
        f"Retrieved context:\n{context}\n\n"
        "Answer in one short paragraph."
    )
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt_text},
    ]

    remote_model = HF_REMOTE_MODEL
    if HF_MODEL_POLICY:
        remote_model = f"{remote_model}:{HF_MODEL_POLICY}"

    headers = {
        "Authorization": f"Bearer {hf_token}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": remote_model,
        "messages": messages,
        "stream": False,
    }

    try:
        response = requests.post(HF_ROUTER_URL, headers=headers, json=payload, timeout=REMOTE_TIMEOUT)
        response.raise_for_status()
        data = response.json()
        if isinstance(data, dict):
            choices = data.get("choices", [])
            if choices:
                message = choices[0].get("message", {})
                content = message.get("content", "")
                if content:
                    return content.strip()
            generated_text = data.get("generated_text")
            if generated_text:
                return str(generated_text).strip()
    except Exception:
        pass

    legacy_response = requests.post(
        f"https://api-inference.huggingface.co/models/{HF_REMOTE_MODEL}",
        headers={"Authorization": f"Bearer {hf_token}"},
        json={"inputs": prompt_text, "options": {"wait_for_model": True}},
        timeout=REMOTE_TIMEOUT,
    )
    legacy_response.raise_for_status()
    payload = legacy_response.json()

    if isinstance(payload, list) and payload and isinstance(payload[0], dict):
        return payload[0].get("generated_text", "").strip()
    if isinstance(payload, dict):
        return str(payload.get("generated_text", payload)).strip()
    return str(payload).strip()


def generate_local_answer(question: str, context: str) -> str:
    initialize_local_llm()
    import torch

    prompt = (
        f"{SYSTEM_PROMPT}\n\n"
        f"Question: {question}\n\n"
        f"Retrieved context:\n{context}\n\n"
        "Answer:"
    )
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {key: value.to(model.device) for key, value in inputs.items()}
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=128, do_sample=False)
    return tokenizer.decode(output[0], skip_special_tokens=True).strip()


def answer_question(question: str, docs) -> tuple[str, str]:
    context = _format_docs(docs)

    if USE_REMOTE_LLM:
        try:
            answer = generate_remote_answer(question, context)
            if answer:
                return answer, f"hf-inference:{LLM_MODEL}"
        except Exception:
            pass

    if not USE_REMOTE_LLM:
        try:
            answer = generate_local_answer(question, context)
            if answer:
                return answer, f"local:{LLM_MODEL}"
        except Exception:
            pass

    return _extractive_fallback(question, docs), "extractive-fallback"


app = FastAPI(title="RAG API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True,
)


@app.get("/")
async def root():
    return {"status": "ok", "service": "rag-api"}


@app.get("/health")
async def health():
    return {"status": "healthy", "device": DEVICE, "mode": "remote" if USE_REMOTE_LLM else "local"}


@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    if not request.question or not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    start_time = time.time()
    try:
        docs = retrieve_docs(request.question, request.top_k)
        answer, backend = answer_question(request.question, docs)
        if _looks_copied(answer, docs):
            answer = _extractive_fallback(request.question, docs)
            backend = "extractive-fallback"
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    generation_time = round(time.time() - start_time, 2)
    return QueryResponse(
        question=request.question,
        answer=answer,
        generation_time=generation_time,
        model=backend,
        device=DEVICE,
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8000")))
