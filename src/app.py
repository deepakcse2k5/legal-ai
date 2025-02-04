import os
import logging
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from src.utils import load_pdf, create_retriever, get_qa_chain, summarize_document

logging.basicConfig(level=logging.INFO)

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change "*" to specific domains if needed
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QueryRequest(BaseModel):
    question: str

class DocumentProcessor:
    """Handles document processing and retrieval operations."""
    def __init__(self):
        self.latest_docs = None
        self.latest_retriever = None
        self.latest_qa_chain = None
        self.latest_summary = None

    async def process_upload(self, file: UploadFile):
        """Uploads and processes a PDF file."""
        try:
            temp_filename = "latest_uploaded.pdf"
            with open(temp_filename, "wb") as buffer:
                buffer.write(await file.read())

            self.latest_docs = load_pdf(temp_filename)
            self.latest_retriever = create_retriever(self.latest_docs)
            self.latest_qa_chain = get_qa_chain(self.latest_retriever)
            self.latest_summary = None  # Reset summary

            os.remove(temp_filename)
            return {"message": "File uploaded and processed successfully"}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    def generate_summary(self):
        """Generates a summary for the latest uploaded document."""
        if not self.latest_docs:
            raise HTTPException(status_code=404, detail="No document uploaded")

        if self.latest_summary is None:
            self.latest_summary = summarize_document(self.latest_docs)

        # Ensure the response contains a properly formatted summary
        if not self.latest_summary or "summary" not in self.latest_summary or not self.latest_summary[
            "summary"].strip():
            raise HTTPException(status_code=500, detail="Failed to generate a valid summary")

        return {self.latest_summary["summary"].strip()}  # Return only the clean summary

    def answer_question(self, question: str):
        """Answers a question related to the latest document."""
        if not self.latest_qa_chain:
            raise HTTPException(status_code=404, detail="No document uploaded")

        try:
            response = self.latest_qa_chain(question)

            # Ensure response is valid and contains an answer
            if not response or "answer" not in response or not response["answer"].strip():
                raise HTTPException(status_code=500, detail="Failed to retrieve a valid answer")

            return {"answer": response["answer"].strip()}  # Return only the clean answer

        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))


document_processor = DocumentProcessor()

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    return await document_processor.process_upload(file)

@app.post("/summary")
async def get_summary():
    return document_processor.generate_summary()

@app.post("/query")
async def ask_question(request: QueryRequest):
    return document_processor.answer_question(request.question)
