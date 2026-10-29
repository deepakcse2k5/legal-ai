import io

import pytest
from fastapi.testclient import TestClient
from fpdf import FPDF

from src import llm
from src.app import app

CLAUSES = [
    "1. Payment. The Client shall pay all undisputed invoices within thirty (30) days of the invoice date.",
    "2. Termination. Either party may terminate this Agreement on sixty (60) days written notice.",
    "3. Governing law. This Agreement is governed by the laws of India.",
]


def make_pdf(pages=None) -> bytes:
    pdf = FPDF()
    for text in pages or ["\n\n".join(CLAUSES)]:
        pdf.add_page()
        pdf.set_font("Helvetica", size=11)
        pdf.multi_cell(0, 6, text)
    return bytes(pdf.output())


@pytest.fixture
def client(monkeypatch):
    calls = []

    def fake(system, user, max_tokens):
        calls.append((system, user))
        return "Fake answer [p. 1]"

    monkeypatch.setattr(llm, "generate", fake)
    c = TestClient(app)
    c.calls = calls
    return c


@pytest.fixture
def session_id(client):
    r = client.post("/upload", files={"file": ("a.pdf", io.BytesIO(make_pdf()), "application/pdf")})
    assert r.status_code == 200, r.text
    return r.json()["session_id"]
