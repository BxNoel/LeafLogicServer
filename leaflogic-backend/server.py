import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
app = FastAPI()

# DEV: allow requests from anywhere (you can lock down later)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

class SimilarityRequest(BaseModel):
    a: str
    b: str

def dot(u, v):
    return sum(x * y for x, y in zip(u, v))

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
    return {"similarity": dot(v1, v2)}
