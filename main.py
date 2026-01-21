import os
from fastapi import FastAPI, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI

app = FastAPI()

# Allow calls from the browser (GEE runs in the browser)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # easiest for trial; later you can restrict
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

client = OpenAI()  # reads OPENAI_API_KEY from environment

# Optional protection so random people don't spam your endpoint
BACKEND_SHARED_SECRET = os.getenv("BACKEND_SHARED_SECRET", "")

class ChatRequest(BaseModel):
    message: str
    yearA: int | None = None
    yearB: int | None = None
    bbox: list[float] | None = None  # [west, south, east, north]

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/chat")
def chat(req: ChatRequest, x_secret: str | None = Header(default=None)):
    # Optional simple auth
    if BACKEND_SHARED_SECRET and x_secret != BACKEND_SHARED_SECRET:
        return {"reply": "Unauthorized (wrong secret)."}

    system = (
        "You are a helpful assistant for a Google Earth Engine app that shows "
        "Dynamic World landcover for Abu Dhabi with Year A/Year B sliders, "
        "Sentinel-2 RGB layers, and a Change layer. "
        "Answer simply and give practical steps."
    )

    extra = ""
    if req.yearA is not None and req.yearB is not None:
        extra += f"\nSelected Year A={req.yearA}, Year B={req.yearB}."
    if req.bbox is not None:
        extra += f"\nAOI bbox={req.bbox}."

    # OpenAI Responses API (recommended for new projects) :contentReference[oaicite:1]{index=1}
    resp = client.responses.create(
        model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        input=[
            {"role": "system", "content": system + extra},
            {"role": "user", "content": req.message},
        ],
    )

    return {"reply": resp.output_text}
