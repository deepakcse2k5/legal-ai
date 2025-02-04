from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from src.utils import load_pdf, create_retriever, get_qa_chain, summarize_document
from pydantic import BaseModel
import os
from fastapi.middleware.cors import CORSMiddleware
import logging

logging.basicConfig(level=logging.INFO)
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change "*" to specific domains if needed
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global storage for the latest document and its related objects
latest_docs = None
latest_retriever = None
latest_qa_chain = None
latest_summary = None
class QueryRequest(BaseModel):
    question: str

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    """
    Upload a PDF file, process it, and store retriever and QA chain for future queries.
    This only keeps the latest uploaded document.
    """
    global latest_docs, latest_retriever, latest_qa_chain, latest_summary

    try:
        # Save the uploaded file temporarily
        temp_filename = "latest_uploaded.pdf"

        # ✅ Properly write the file
        with open(temp_filename, "wb") as buffer:
            buffer.write(await file.read())  # ✅ FIXED: Read file content asynchronously

        # ✅ Process the PDF file
        latest_docs = load_pdf(temp_filename)  # ✅ Load the PDF into `latest_docs`
        latest_retriever = create_retriever(latest_docs)  # ✅ Create retriever
        latest_qa_chain = get_qa_chain(latest_retriever)  # ✅ Initialize QA chain

        # ✅ Reset summary when a new document is uploaded
        latest_summary = None  # ✅ This ensures a new summary is generated when requested

        # ✅ Remove temporary file
        os.remove(temp_filename)

        return {"message": "File uploaded and processed successfully"}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))




@app.post("/summary")
async def get_summary():
    """
    Get the summary of the latest uploaded document.
    """
    global latest_docs, latest_summary

    if latest_docs is None:
        logging.error("No document uploaded")
        raise HTTPException(status_code=404, detail="No document uploaded")

    if latest_summary is None:
        logging.info("Generating summary for the latest document")
        latest_summary = summarize_document(latest_docs)

    return {"summary": latest_summary}


@app.post("/query")
async def ask_question(request: QueryRequest):
    """
    Ask a question related to the latest uploaded document.
    """
    global latest_qa_chain

    if latest_qa_chain is None:
        raise HTTPException(status_code=404, detail="No document uploaded")

    try:
        response = latest_qa_chain(request.question)  # Directly call the function
        print(f"response: {response}")

        if not response or "answer" not in response:
            raise HTTPException(status_code=500, detail="Failed to retrieve a valid answer")

        return {
            "answer": response["answer"]
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
