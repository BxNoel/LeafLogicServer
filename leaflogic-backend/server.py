import os
import math
from typing import List

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI
from pydantic import BaseModel, conlist

load_dotenv()

app = FastAPI()

# DEV: allow requests from anywhere (lock down later)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


# -----------------------------
# Request Models
# -----------------------------
class SimilarityRequest(BaseModel):
    a: str
    b: str


class EmbeddingRequest(BaseModel):
    text: str


class VectorSimilarityRequest(BaseModel):
    # JSON arrays of floats, e.g. {"a":[...], "b":[...]}
    a: conlist(float, min_length=1)
    b: conlist(float, min_length=1)


# -----------------------------
# Math helpers
# -----------------------------
def dot(u: List[float], v: List[float]) -> float:
    return sum(x * y for x, y in zip(u, v))


def norm(v: List[float]) -> float:
    return math.sqrt(dot(v, v))


def cosine_similarity(u: List[float], v: List[float]) -> float:
    denom = norm(u) * norm(v)
    if denom == 0.0:
        # Avoid division by zero if someone sends a zero vector
        return 0.0
    return dot(u, v) / denom


@app.get("/")
def hello():
    return {"hello": True}


@app.get("/health")
def health():
    return {"ok": True}


@app.post("/similarity")
def similarity(req: SimilarityRequest):
    emb = client.embeddings.create(
        model="text-embedding-3-small",
        input=[req.a, req.b],
    )
    v1 = emb.data[0].embedding
    v2 = emb.data[1].embedding
    return {"similarity": cosine_similarity(v1, v2)}


@app.post("/embedding")
def embedding(req: EmbeddingRequest):
    emb = client.embeddings.create(
        model="text-embedding-3-small",
        input=req.text,
    )
    return {"embedding": emb.data[0].embedding}

# Compute cosine similarity for TWO provided vectors
@app.post("/similarity/embed")
def similarity_embedded(req: VectorSimilarityRequest):
    if len(req.a) != len(req.b):
        raise HTTPException(
            status_code=400,
            detail=f"Vectors must be the same length (got {len(req.a)} and {len(req.b)})",
        )

    return {"similarity": cosine_similarity(req.a, req.b)}
