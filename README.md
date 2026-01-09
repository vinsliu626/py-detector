import hashlib
import os
import re
import time
from collections import OrderedDict

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from model import GPT2PPL

# ---------- app ----------
app = FastAPI(title="py-detector", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "https://ai-multimodel-erhw.vercel.app",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Model: keep accuracy
MODEL_ID = os.getenv("MODEL_ID", "gpt2-medium")
detector = GPT2PPL(model_id=MODEL_ID)

class Req(BaseModel):
    text: str

def word_count(text: str) -> int:
    return len(re.findall(r"\S+", text.strip()))

def normalize_text(t: str) -> str:
    # normalize whitespace so same article copied twice hits cache
    return re.sub(r"\s+", " ", (t or "").strip())

# ---------- cache (LRU + TTL) ----------
CACHE_MAX = 128
CACHE_TTL_SEC = 60 * 30  # 30 minutes
_cache = OrderedDict()   # key -> (ts, payload)

def cache_get(key: str):
    now = time.time()
    if key not in _cache:
        return None
    ts, payload = _cache[key]
    if now - ts > CACHE_TTL_SEC:
        _cache.pop(key, None)
        return None
    _cache.move_to_end(key)
    return payload

def cache_set(key: str, payload):
    _cache[key] = (time.time(), payload)
    _cache.move_to_end(key)
    while len(_cache) > CACHE_MAX:
        _cache.popitem(last=False)

@app.get("/")
def root():
    return {"ok": True, "service": "py-detector", "model": MODEL_ID}

@app.post("/detect")
def detect(req: Req):
    text = normalize_text(req.text)

    if word_count(text) < 80:
        return {
            "ok": False,
            "error": "Need at least 80 words for stable detection.",
            "debug": {"word_count": word_count(text), "text_len": len(text)}
        }

    key = hashlib.sha256(text.encode("utf-8")).hexdigest()
    cached = cache_get(key)
    if cached is not None:
        return {**cached, "cached": True}

    result = detector(text)  # (metrics_dict, message)
    payload = {"ok": True, "result": result}
    cache_set(key, payload)
    return {**payload, "cached": False}
