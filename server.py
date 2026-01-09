from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import re
import os
import hashlib
import time
from collections import OrderedDict

from model import GPT2PPL

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 先放开，等你稳定再收紧到你的 Vercel 域名
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Req(BaseModel):
    text: str

def word_count(text: str) -> int:
    return len(re.findall(r"\S+", text.strip()))

def normalize_text(text: str) -> str:
    # 归一化：合并空白，防止同文不同空格导致 hash 不同
    return re.sub(r"\s+", " ", (text or "").strip())

# -------- Lazy model (avoid OOM at boot) --------
DETECTOR = None

def get_detector():
    global DETECTOR
    if DETECTOR is None:
        # 也可以用环境变量 MODEL_ID 控制
        model_id = os.getenv("MODEL_ID", "distilgpt2")
        DETECTOR = GPT2PPL(model_id=model_id)
    return DETECTOR

# -------- Small LRU cache (same input => same output) --------
CACHE_MAX = int(os.getenv("CACHE_MAX", "32"))
CACHE_TTL = int(os.getenv("CACHE_TTL", "3600"))  # 1 hour

_cache = OrderedDict()  # key -> (ts, payload)

def cache_get(key: str):
    item = _cache.get(key)
    if not item:
        return None
    ts, payload = item
    if time.time() - ts > CACHE_TTL:
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
def health():
    return {"ok": True, "service": "py-detector"}

@app.post("/detect")
def detect(req: Req):
    text = normalize_text(req.text)

    if word_count(text) < 80:
        return {
            "ok": False,
            "error": "Need at least 80 words for stable detection.",
            "debug": {"word_count": word_count(text), "text_len": len(text), "text_preview": text[:200]},
        }

    # hash key
    key = hashlib.sha256(text.encode("utf-8")).hexdigest()

    hit = cache_get(key)
    if hit is not None:
        return {"ok": True, "cached": True, "result": hit}

    detector = get_detector()
    result = detector(text)  # (metrics, message)

    cache_set(key, result)
    return {"ok": True, "cached": False, "result": result}
