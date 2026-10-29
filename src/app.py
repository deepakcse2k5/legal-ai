"""FastAPI service. Upload a PDF, get a session id, then ask / summarise / draft against that session."""
import logging

import hmac

from fastapi import APIRouter, Depends, FastAPI, File, Header, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src import service
from src.config import settings
from src.llm import LLMError
from src.loader import DocumentError

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
log = logging.getLogger("legal-ai")

def require_key(x_api_key: str = Header(default="")):
    """If API_KEY is set, every call except /health must send it (the website Worker and legal-ui do)."""
    if settings.api_key and not hmac.compare_digest(x_api_key.encode(), settings.api_key.encode()):
        raise HTTPException(401, "Missing or invalid API key.")


app = FastAPI(title="Legal AI", version="3.0.0")
api = APIRouter(dependencies=[Depends(require_key)])   # everything except /health
app.add_middleware(
    CORSMiddleware,
    allow_origins=list(settings.cors_origins),
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["*"],
)


class UploadResponse(BaseModel):
    session_id: str
    semantic_search: bool = False
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





@api.get("/drafts")
def drafts():
    return {"kinds": list(service.DRAFTS)}


@api.post("/upload", response_model=UploadResponse)
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
    return UploadResponse(semantic_search=s.retriever.semantic, session_id=s.id, filename=s.filename, pages=s.pages, passages=len(s.chunks))


@api.post("/query", response_model=QueryResponse)
def query(req: QueryRequest):
    return _llm(lambda: service.answer(_session(req.session_id), req.question))


@api.post("/summary", response_model=SummaryResponse)
def summary(req: SessionRequest):
    return SummaryResponse(summary=_llm(lambda: service.summarize(_session(req.session_id))))


@api.post("/draft", response_model=DraftResponse)
def draft(req: DraftRequest):
    if req.kind not in service.DRAFTS:
        raise HTTPException(400, f"Unknown draft type. Choose one of: {', '.join(service.DRAFTS)}")
    return DraftResponse(draft=_llm(lambda: service.draft(_session(req.session_id), req.kind)))


@api.delete("/session/{session_id}")
def delete_session(session_id: str):
    service.store.delete(session_id)
    return {"deleted": True}


@app.get("/health")
def health():
    return {"status": "ok", "provider": settings.llm_provider, "semantic_search": settings.embeddings == "cloudflare"}


app.include_router(api)
