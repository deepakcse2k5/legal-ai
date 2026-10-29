"""Document sessions plus question answering, summarisation and drafting, all with page citations."""
import threading
import time
import uuid
from dataclasses import dataclass, field

from src import llm
from src.config import settings
from src.loader import Chunk, chunk_pages, extract_pages
from src.retrieval import Retriever

RULES = (
    "Use ONLY the text provided. Cite every point with its page as [p. N]. "
    "Never invent facts, quotes, case names or terms. If the text does not say, say so. "
    "This is decision support, not legal advice."
)

DRAFTS = {
    "case_note": "Draft a one-page case or document note: parties and context, key facts, issues, outcome or key terms, and why it matters. Use short headings. Under 300 words.",
    "client_email": "Draft a short plain-English email to a client explaining the main points of this document and what they should do next. Subject line first. No jargon. Under 220 words.",
    "memo": "Draft a short internal legal memo for a senior lawyer: key provisions or holdings, risks, and practical implications. Use headings. Under 300 words.",
    "negotiation_note": "Draft a short note proposing changes to the provisions that look least favourable to our side. For each: the provision, the concern, and the change requested. Under 250 words.",
}


@dataclass
class Session:
    id: str
    filename: str
    pages: int
    chunks: list[Chunk]
    retriever: Retriever
    created: float = field(default_factory=time.time)
    summary: str | None = None


class SessionNotFound(KeyError):
    pass


class Store:
    """In-memory, per-upload sessions with expiry, so users never see each other's documents."""

    def __init__(self):
        self._items: dict[str, Session] = {}
        self._lock = threading.Lock()

    def _purge(self):
        now = time.time()
        for sid in [s for s, v in self._items.items() if now - v.created > settings.session_ttl_seconds]:
            del self._items[sid]
        while len(self._items) >= settings.max_sessions:
            oldest = min(self._items.values(), key=lambda v: v.created)
            del self._items[oldest.id]

    def create(self, filename: str, data: bytes) -> Session:
        pages = extract_pages(data, settings.max_pages)
        chunks = chunk_pages(pages, settings.chunk_chars)
        sess = Session(uuid.uuid4().hex, filename, len(pages), chunks, Retriever(chunks))
        with self._lock:
            self._purge()
            self._items[sess.id] = sess
        return sess

    def get(self, sid: str) -> Session:
        with self._lock:
            self._purge()
            if sid not in self._items:
                raise SessionNotFound(sid)
            return self._items[sid]

    def delete(self, sid: str) -> None:
        with self._lock:
            self._items.pop(sid, None)


def _fmt(chunks: list[Chunk]) -> str:
    return "\n\n".join(f"[p. {c.page}] {c.text}" for c in chunks)


def answer(sess: Session, question: str) -> dict:
    sources = sess.retriever.search(question, settings.top_k)
    if not sources:
        return {"answer": "I could not find anything in this document that relates to that question.", "sources": []}
    system = f"You answer questions about a legal document. Be concise (under 150 words). {RULES}"
    text = llm.generate(system, f"Document passages:\n\n{_fmt(sources)}\n\nQuestion: {question}", settings.max_answer_tokens)
    return {"answer": text, "sources": [{"page": c.page, "chunk": c.id, "text": c.text} for c in sources]}


def _batches(chunks: list[Chunk], max_chars: int) -> list[list[Chunk]]:
    out, cur, size = [], [], 0
    for c in chunks:
        if cur and size + len(c.text) > max_chars:
            out.append(cur)
            cur, size = [], 0
        cur.append(c)
        size += len(c.text)
    if cur:
        out.append(cur)
    return out


def _condense(sess: Session) -> str:
    """Whole document if it fits the context, else a map-reduce of per-section notes."""
    budget = int(settings.context_tokens * 4 * 0.55)          # about 55% of context, in characters
    batches = _batches(sess.chunks, budget)
    if len(batches) == 1:
        return _fmt(batches[0])
    notes = []
    for batch in batches[:8]:                                  # hard cap on model calls
        notes.append(llm.generate(
            f"You take notes on one section of a legal document. List the key facts, obligations, dates, amounts and decisions as short bullets. {RULES}",
            _fmt(batch), 500))
    return "\n\n".join(notes)


def summarize(sess: Session) -> str:
    if sess.summary is None:
        system = (
            "You summarise a legal document for a busy lawyer: (1) what it is, in one or two sentences; "
            "(2) key points as 5 to 7 bullets; (3) deadlines, amounts or obligations to remember. " + RULES
        )
        sess.summary = llm.generate(system, _condense(sess), settings.max_summary_tokens)
    return sess.summary


def draft(sess: Session, kind: str) -> str:
    if kind not in DRAFTS:
        raise ValueError(f"Unknown draft type. Choose one of: {', '.join(DRAFTS)}")
    system = f"You are a legal drafting assistant. {DRAFTS[kind]} {RULES} Write the draft only, with no preamble."
    return llm.generate(system, _condense(sess), settings.max_summary_tokens)


store = Store()
