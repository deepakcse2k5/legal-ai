import os
import logging
from fastapi import FastAPI, UploadFile, File, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict
from tempfile import NamedTemporaryFile
from src.utils import load_pdf, create_retriever, get_qa_chain, summarize_document

# Configure logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# FastAPI app initialization
app = FastAPI()

# CORS Middleware (restrict origins in production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change "*" to specific domains
    allow_credentials=True,
    allow_methods=["POST"],
    allow_headers=["*"],
)


class QueryRequest(BaseModel):
    question: str


class SummaryResponse(BaseModel):
    summary: str


class AnswerResponse(BaseModel):
    answer: str


class UploadResponse(BaseModel):
    message: str


class DocumentProcessor:
    """Handles document processing and retrieval operations."""

    def __init__(self):
        self.latest_docs = None
        self.latest_retriever = None
        self.latest_qa_chain = None
        self.latest_summary = None

    async def process_upload(self, file: UploadFile) -> Dict[str, str]:
        """Uploads and processes a PDF file."""
        try:
            # Use NamedTemporaryFile to avoid manual file cleanup
            with NamedTemporaryFile(delete=True, suffix=".pdf") as temp_file:
                temp_file.write(await file.read())
                temp_file.flush()  # Ensure all data is written before processing

                self.latest_docs = load_pdf(temp_file.name)
                self.latest_retriever = create_retriever(self.latest_docs)
                self.latest_qa_chain = get_qa_chain(self.latest_retriever)
                self.latest_summary = None  # Reset summary

            logger.info("File uploaded and processed successfully")
            return {"message": "File uploaded and processed successfully"}

        except Exception as e:
            logger.error(f"Error processing file: {e}")
            raise HTTPException(status_code=500, detail="Error processing file")

    def generate_summary(self) -> SummaryResponse:
        """Generates a summary for the latest uploaded document."""
        if not self.latest_docs:
            raise HTTPException(status_code=404, detail="No document uploaded")

        if self.latest_summary is None:
            self.latest_summary = summarize_document(self.latest_docs)

        if not self.latest_summary or "summary" not in self.latest_summary or not self.latest_summary[
            "summary"].strip():
            raise HTTPException(status_code=500, detail="Failed to generate a valid summary")

        return SummaryResponse(summary=self.latest_summary["summary"].strip())

    def answer_question(self, question: str) -> AnswerResponse:
        """Answers a question related to the latest document."""
        if not self.latest_qa_chain:
            raise HTTPException(status_code=404, detail="No document uploaded")

        try:
            response = self.latest_qa_chain(question)

            if not response or "answer" not in response or not response["answer"].strip():
                raise HTTPException(status_code=500, detail="Failed to retrieve a valid answer")

            return AnswerResponse(answer=response["answer"].strip())

        except Exception as e:
            logger.error(f"Error answering question: {e}")
            raise HTTPException(status_code=500, detail="Error processing question")


# Singleton instance of DocumentProcessor
document_processor = DocumentProcessor()


@app.post("/upload", response_model=UploadResponse)
async def upload_file(file: UploadFile = File(...)):
    """Endpoint for uploading and processing a PDF document."""
    return await document_processor.process_upload(file)


@app.post("/summary", response_model=SummaryResponse)
async def get_summary():
    """Endpoint for retrieving the document summary."""
    return document_processor.generate_summary()


@app.post("/query", response_model=AnswerResponse)
async def ask_question(request: QueryRequest):
    """Endpoint for querying the processed document."""
    return document_processor.answer_question(request.question)
