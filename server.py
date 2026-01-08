from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

import os
import re
import time
import hashlib
from typing import Any, Dict, Tuple

from model import GPT2PPL


print("SERVER.PY PATH =", os.path.abspath(__file__))

app = FastAPI()

# ✅ 你自己的前端域名（部署后把 vercel 域名加进来）
ALLOWED_ORIGINS = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    # 你上线后加：
    # "https://ai-multimodel-erhw.vercel.app",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

detector = GPT2PPL()

class Req(BaseModel):
    text: str


def word_count(text: str) -> int:
    return len(re.findall(r"\S+", text.strip()))


# =======================
# ✅ Small Memory Cache
# - same text -> same result
# - TTL to avoid infinite memory usage
# =======================
CACHE_TTL_SECONDS = int(os.getenv("CACHE_TTL_SECONDS", "3600"))  # 1 hour default
CACHE_MAX_ITEMS = int(os.getenv("CACHE_MAX_ITEMS", "200"))       # cap size

_cache: Dict[str, Tuple[float, Dict[str, Any]]] = {}  # key -> (expire_ts, response)


def normalize_text_for_cache(text: str) -> str:
    # normalize whitespace to avoid "same article" with different spaces
    return re.sub(r"\s+", " ", (text or "").strip())


def cache_key(text: str) -> str:
    norm = normalize_text_for_cache(text)
    return hashlib.sha256(norm.encode("utf-8")).hexdigest()


def cache_get(key: str):
    now = time.time()
    item = _cache.get(key)
    if not item:
        return None
    exp, val = item
    if exp < now:
        _cache.pop(key, None)
        return None
    return val


def cache_set(key: str, val: Dict[str, Any]):
    # cleanup if oversized (simple strategy)
    if len(_cache) >= CACHE_MAX_ITEMS:
        # remove expired first
        now = time.time()
        expired = [k for k, (exp, _) in _cache.items() if exp < now]
        for k in expired:
            _cache.pop(k, None)
        # still too big -> pop arbitrary oldest-ish
        while len(_cache) >= CACHE_MAX_ITEMS:
            _cache.pop(next(iter(_cache)), None)

    _cache[key] = (time.time() + CACHE_TTL_SECONDS, val)


@app.get("/health")
def health():
    return {"ok": True, "status": "healthy"}


@app.post("/detect")
def detect(req: Req):
    text = (req.text or "").strip()

    if word_count(text) < 80:
        return {
            "ok": False,
            "error": "Need at least 80 words for stable detection.",
            "debug": {
                "word_count": word_count(text),
                "text_len": len(text),
                "text_preview": text[:200],
            },
        }

    key = cache_key(text)
    cached = cache_get(key)
    if cached:
        # ✅ tell client it was cached (optional)
        cached2 = dict(cached)
        cached2["cache"] = {"hit": True, "ttlSeconds": CACHE_TTL_SECONDS}
        return cached2

    result = detector(text)  # (metrics_dict, message)
    resp = {"ok": True, "result": result, "cache": {"hit": False, "ttlSeconds": CACHE_TTL_SECONDS}}
    cache_set(key, resp)
    return resp
