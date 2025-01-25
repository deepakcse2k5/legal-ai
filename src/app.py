import logging
import os
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from src.pipeline import save_to_chroma, query_rag, split_text
from langchain.document_loaders import PyPDFLoader

# Configure Logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("app.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# FastAPI Application
app = FastAPI(title="Legal Query API", description="A REST API for uploading legal documents and querying them using RAG.", version="1.0")

# Constants
UPLOAD_DIRECTORY = "uploaded_docs"
CHROMA_PATH = "chroma"

# Pydantic Model for Query
class QueryRequest(BaseModel):
    query: str

# Ensure upload directory exists
if not os.path.exists(UPLOAD_DIRECTORY):
    os.makedirs(UPLOAD_DIRECTORY)

@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    """
    Endpoint to upload a document and process it for querying.

    Args:
        file (UploadFile): The PDF document to upload.

    Returns:
        JSONResponse: Confirmation message on successful processing.
    """
    try:
        # Save the uploaded file to the upload directory
        file_path = os.path.join(UPLOAD_DIRECTORY, file.filename)
        with open(file_path, "wb") as f:
            f.write(await file.read())
        logger.info(f"File uploaded and saved at {file_path}")

        # Load the uploaded document
        loader = PyPDFLoader(file_path)
        documents = loader.load()
        logger.info(f"Loaded {len(documents)} pages from the uploaded document.")

        # Split the text and save it to Chroma
        chunks = split_text(documents)
        save_to_chroma(chunks, CHROMA_PATH)
        logger.info(f"Document processed and added to Chroma database.")

        return JSONResponse(content={"message": "Document uploaded and processed successfully."}, status_code=200)
    except Exception as e:
        logger.error(f"Error processing the uploaded document: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="An error occurred while processing the document.")

@app.post("/query")
async def handle_query(request: QueryRequest):
    """
    Handle a query and return the RAG response.

    Args:
        request (QueryRequest): The query request containing the query text.

    Returns:
        dict: Response containing the answer and sources.
    """
    try:
        logger.info(f"Received query: {request.query}")
        response_text = query_rag(request.query, CHROMA_PATH)
        if response_text == "":
            raise HTTPException(status_code=404, detail="No relevant information found.")
        return {"response": response_text}
    except Exception as e:
        logger.error(f"Error handling query: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="An error occurred while processing the query.")
