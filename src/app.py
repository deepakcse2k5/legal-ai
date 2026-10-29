"""FastAPI service. Upload a PDF, get a session id, then ask / summarise / draft against that session."""
import logging

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src import service
from src.config import settings
from src.llm import LLMError
from src.loader import DocumentError

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
log = logging.getLogger("legal-ai")

app = FastAPI(title="Legal AI", version="2.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=list(settings.cors_origins),
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["*"],
)


class UploadResponse(BaseModel):
    session_id: str
    filename: str
    pages: int
    passages: int


class Source(BaseModel):
    page: int
    chunk: int
    text: str


class QueryRequest(BaseModel):
    session_id: str
    question: str = Field(min_length=3, max_length=500)


class QueryResponse(BaseModel):
    answer: str
    sources: list[Source]


class SessionRequest(BaseModel):
    session_id: str


class SummaryResponse(BaseModel):
    summary: str


class DraftRequest(BaseModel):
    session_id: str
    kind: str


class DraftResponse(BaseModel):
    draft: str


def _session(sid: str) -> service.Session:
    try:
        return service.store.get(sid)
    except service.SessionNotFound:
        raise HTTPException(404, "Unknown or expired session. Upload the document again.")


def _llm(call):
    try:
        return call()
    except LLMError as exc:
        log.error("LLM error: %s", exc)
        raise HTTPException(502, str(exc))


@app.get("/health")
def health():
    return {"status": "ok", "provider": settings.llm_provider}


@app.get("/drafts")
def drafts():
    return {"kinds": list(service.DRAFTS)}


@app.post("/upload", response_model=UploadResponse)
async def upload(file: UploadFile = File(...)):
    data = await file.read(settings.max_upload_mb * 1024 * 1024 + 1)
    if len(data) > settings.max_upload_mb * 1024 * 1024:
        raise HTTPException(413, f"File is larger than {settings.max_upload_mb} MB.")
    if not data.startswith(b"%PDF"):
        raise HTTPException(415, "Only PDF files are supported.")
    try:
        s = service.store.create(file.filename or "document.pdf", data)
    except DocumentError as exc:
        raise HTTPException(422, str(exc))
    return UploadResponse(session_id=s.id, filename=s.filename, pages=s.pages, passages=len(s.chunks))


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest):
    return _llm(lambda: service.answer(_session(req.session_id), req.question))


@app.post("/summary", response_model=SummaryResponse)
def summary(req: SessionRequest):
    return SummaryResponse(summary=_llm(lambda: service.summarize(_session(req.session_id))))


@app.post("/draft", response_model=DraftResponse)
def draft(req: DraftRequest):
    if req.kind not in service.DRAFTS:
        raise HTTPException(400, f"Unknown draft type. Choose one of: {', '.join(service.DRAFTS)}")
    return DraftResponse(draft=_llm(lambda: service.draft(_session(req.session_id), req.kind)))


@app.delete("/session/{session_id}")
def delete_session(session_id: str):
    service.store.delete(session_id)
    return {"deleted": True}
