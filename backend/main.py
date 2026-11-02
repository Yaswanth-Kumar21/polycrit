"""Polycrit API + web UI."""
import json
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel

from agents import stream_review

load_dotenv()

app = FastAPI(title="Polycrit", version="0.1.0")

FRONTEND = Path(__file__).resolve().parent.parent / "frontend" / "index.html"


class ReviewRequest(BaseModel):
    repo_url: str | None = None
    code: str | None = None
    language: str = "python"


@app.get("/")
def index():
    return HTMLResponse(FRONTEND.read_text(encoding="utf-8"))


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.post("/api/review")
def review(req: ReviewRequest):
    if not req.repo_url and not req.code:
        return JSONResponse({"error": "Provide either a repo_url or pasted code."}, status_code=400)

    def event_stream():
        try:
            for event in stream_review(code=req.code, repo_url=req.repo_url, language=req.language):
                yield f"data: {json.dumps(event)}\n\n"
        except Exception as e:  # surface the error as a final SSE event
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")
