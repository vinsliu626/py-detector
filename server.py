from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import re, os, hashlib, time
from collections import OrderedDict

from model import GPT2PPL

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "https://ai-multimodel-erhw.vercel.app",  # ✅ 换成你自己的 Vercel 域名
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

detector = GPT2PPL()

class Req(BaseModel):
    text: str

def word_count(text: str) -> int:
    return len(re.findall(r"\S+", text.strip()))

def normalize_text(t: str) -> str:
    # 去掉多余空白，让“同一篇文章复制两次”命中缓存
    return re.sub(r"\s+", " ", (t or "").strip())

# ====== small memory cache (LRU + TTL) ======
CACHE_MAX = 128
CACHE_TTL_SEC = 60 * 30  # 30分钟
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
    return {"ok": True, "service": "py-detector"}

@app.post("/detect")
def detect(req: Req):
    text_raw = req.text or ""
    text = normalize_text(text_raw)

    if word_count(text) < 80:
        return {
            "ok": False,
            "error": "Need at least 80 words for stable detection.",
            "debug": {"word_count": word_count(text), "text_len": len(text)}
        }

    # ✅ hash key
    key = hashlib.sha256(text.encode("utf-8")).hexdigest()
    cached = cache_get(key)
    if cached is not None:
        return {**cached, "cached": True}

    result = detector(text)  # (metrics_dict, message)
    payload = {"ok": True, "result": result}

    cache_set(key, payload)
    return {**payload, "cached": False}
