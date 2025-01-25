import os
import shutil
import logging
from dotenv import load_dotenv
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.schema import Document
from langchain.vectorstores.chroma import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain.chat_models import ChatOpenAI

# Load environment variables
load_dotenv('openai.env')

# Configure Logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("pipeline.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Constants
CHROMA_PATH = "chroma"
PROMPT_TEMPLATE = """
You are a legal expert specializing in the given context. Answer the question based solely on the following context:

{context}

---

Provide a legally accurate response to the following question based on the above context: {question}
"""

# Utility Functions
def split_text(documents: list[Document], chunk_size: int = 400, chunk_overlap: int = 100) -> list[Document]:
    """
    Split text content of documents into smaller chunks.

    Args:
        documents (list[Document]): List of Document objects.
        chunk_size (int): Size of each chunk in characters.
        chunk_overlap (int): Overlap between consecutive chunks.

    Returns:
        list[Document]: Split text chunks.
    """
    logger.info(f"Splitting {len(documents)} documents into chunks...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap, add_start_index=True
    )
    chunks = splitter.split_documents(documents)
    logger.info(f"Split into {len(chunks)} chunks.")
    return chunks

def save_to_chroma(chunks: list[Document], directory: str = CHROMA_PATH):
    """
    Save text chunks to Chroma vector store.

    Args:
        chunks (list[Document]): List of Document objects.
        directory (str): Directory to save the Chroma database.
    """
    if os.path.exists(directory):
        logger.info(f"Clearing existing Chroma database at {directory}.")
        shutil.rmtree(directory)
    else:
        logger.info(f"Creating new Chroma database at {directory}.")
    db = Chroma.from_documents(chunks, OpenAIEmbeddings(), persist_directory=directory)
    db.persist()
    logger.info(f"Saved {len(chunks)} chunks to {directory}.")

def query_rag(query_text: str, directory: str = CHROMA_PATH) -> tuple[str, str]:
    """
    Query a Retrieval-Augmented Generation (RAG) system using Chroma and OpenAI.

    Args:
        query_text (str): Query text for the RAG system.
        directory (str): Path to the Chroma database.

    Returns:
        tuple[str, str]: Formatted response and the raw response text.
    """
    logger.info(f"Querying RAG system with query: '{query_text}'")
    db = Chroma(persist_directory=directory, embedding_function=OpenAIEmbeddings())
    results = db.similarity_search_with_relevance_scores(query_text, k=3)

    if not results or results[0][1] < 0.7:
        logger.warning("Unable to find matching results.")
        return "Unable to find matching results.", ""

    context_text = "\n\n---\n\n".join([doc.page_content for doc, _ in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)

    logger.info("Generating response from OpenAI Chat model.")
    model = ChatOpenAI()
    response_text = model.predict(prompt)
    # sources = [doc.metadata.get("source", None) for doc, _ in results]

    logger.info("Query processing completed.")
    # formatted_response = f"Response: {response_text}\nSources: {sources}"
    return response_text

def process_and_save_document(file_path: str, directory: str = CHROMA_PATH):
    """
    Load, process, and save a document to the Chroma vector store.

    Args:
        file_path (str): Path to the document file.
        directory (str): Directory to save the Chroma database.
    """
    logger.info(f"Processing document: {file_path}")
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    logger.info(f"Loaded {len(documents)} pages from document.")

    chunks = split_text(documents)
    save_to_chroma(chunks, directory)
    logger.info(f"Document processed and saved to {directory}.")
