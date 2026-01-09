# server.py
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from fastapi import Response
import re, hashlib, time
from collections import OrderedDict

from model import GPT2PPL

app = FastAPI()

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "https://ai-multimodel-erhw.vercel.app",  # ✅ 改成你的 Vercel 域名
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ Use smaller model on Render free
detector = GPT2PPL(model_id="distilgpt2")

class Req(BaseModel):
    text: str

def word_count(text: str) -> int:
    return len(re.findall(r"\S+", text.strip()))

def normalize_text(t: str) -> str:
    return re.sub(r"\s+", " ", (t or "").strip())

# ====== small memory cache (LRU + TTL) ======
CACHE_MAX = 64           # ✅ free plan memory small
CACHE_TTL_SEC = 60 * 20  # 20 minutes
_cache = OrderedDict()   # key -> (ts, payload)

def cache_get(key: str):
    now = time.time()
    item = _cache.get(key)
    if not item:
        return None
    ts, payload = item
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
    return {"ok": True, "service": "py-detector"}

@app.head("/")
def head_root():
    return Response(status_code=200)

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
