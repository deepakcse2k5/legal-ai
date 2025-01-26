import logging
import os
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from typing import List
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
app = FastAPI(title="Legal Query API", description="A REST API for uploading and managing legal documents.", version="1.1")

# Constants
UPLOAD_DIRECTORY = "uploaded_docs"
CHROMA_PATH = "chroma"

# Ensure upload directory exists
if not os.path.exists(UPLOAD_DIRECTORY):
    os.makedirs(UPLOAD_DIRECTORY)

# Pydantic Models
class QueryRequest(BaseModel):
    query: str

class DeleteRequest(BaseModel):
    document_ids: List[str]  # List of document IDs to delete

@app.post("/upload")
async def upload_documents(files: List[UploadFile] = File(...)):
    """
    Endpoint to upload multiple documents and process them for querying.

    Args:
        files (List[UploadFile]): List of PDF documents to upload.

    Returns:
        JSONResponse: Confirmation message on successful processing.
    """
    try:
        document_paths = []
        for file in files:
            file_path = os.path.join(UPLOAD_DIRECTORY, file.filename)
            with open(file_path, "wb") as f:
                f.write(await file.read())
            logger.info(f"File uploaded and saved at {file_path}")
            document_paths.append(file_path)

        # Process each document and add it to Chroma DB
        for file_path in document_paths:
            loader = PyPDFLoader(file_path)
            documents = loader.load()
            logger.info(f"Loaded {len(documents)} pages from {file_path}")

            chunks = split_text(documents)
            save_to_chroma(chunks, CHROMA_PATH)
            logger.info(f"Document {file_path} processed and added to Chroma database.")

        return JSONResponse(content={"message": "Documents uploaded and processed successfully."}, status_code=200)
    except Exception as e:
        logger.error(f"Error processing uploaded documents: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="An error occurred while processing the documents.")

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

@app.delete("/delete")
async def delete_documents(request: DeleteRequest):
    """
    Endpoint to delete specific documents from Chroma DB.

    Args:
        request (DeleteRequest): Request containing a list of document IDs to delete.

    Returns:
        JSONResponse: Confirmation message on successful deletion.
    """
    try:
        from chromadb import PersistentClient

        client = PersistentClient(persist_directory=CHROMA_PATH)
        collection = client.get_collection("default")  # Adjust collection name if necessary

        # Delete the specified document IDs
        collection.delete(ids=request.document_ids)
        logger.info(f"Deleted documents with IDs: {request.document_ids}")

        return JSONResponse(content={"message": "Documents deleted successfully."}, status_code=200)
    except Exception as e:
        logger.error(f"Error deleting documents: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="An error occurred while deleting the documents.")
