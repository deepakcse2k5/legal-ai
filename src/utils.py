import re
import logging
import ollama
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from src.config import (
    LLM_MODEL, MAX_TOKENS, STOP_SEQUENCES, TEMPERATURE, TOP_P, TOP_K, FREQUENCY_PENALTY, PRESENCE_PENALTY
)

# Configure logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Singleton for embedding model to optimize performance
EMBEDDINGS = HuggingFaceEmbeddings()


def load_pdf(file_path):
    """Load a PDF file and return parsed documents."""
    try:
        return PDFPlumberLoader(file_path).load()
    except Exception as e:
        logger.error(f"Error loading PDF {file_path}: {e}")
        raise ValueError("Failed to load the PDF file")


def create_retriever(docs, k=3):
    """Create a deterministic FAISS retriever."""
    try:
        chunked_docs = SemanticChunker(EMBEDDINGS).split_documents(docs)
        vectorstore = FAISS.from_documents(chunked_docs, EMBEDDINGS)
        return vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": k})
    except Exception as e:
        logger.error(f"Error creating retriever: {e}")
        raise RuntimeError("Failed to create document retriever")


def generate_response(prompt):
    """Generate a deterministic response using the Ollama model."""
    try:
        response = ollama.chat(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            options={
                "temperature": TEMPERATURE,  # Ensures deterministic responses
                "top_p": TOP_P,        # Uses all tokens in order, no randomness
                "top_k": TOP_K,        # Picks the highest probability token
                "max_tokens": MAX_TOKENS,
                "frequency_penalty": FREQUENCY_PENALTY,
                "presence_penalty": PRESENCE_PENALTY
            }
        )
        content = response.get("message", {}).get("content", "").strip()

        if not content:
            logger.warning("Received an empty response from Ollama")
            raise ValueError("No valid response from language model")

        return re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()

    except Exception as e:
        logger.error(f"Error generating response: {e}")
        raise RuntimeError("Failed to generate response from the model")


def get_qa_chain(retriever):
    """Set up the deterministic QA system."""

    def ollama_qa(question):
        """Retrieve relevant context and generate an answer deterministically."""
        try:
            documents = retriever.get_relevant_documents(question)
            context = "\n".join([doc.page_content for doc in documents])

            if not context.strip():
                return {"answer": "The provided context does not contain sufficient information to answer this question."}

            qa_prompt = f"""
            You are a legal expert specializing in the given context. Answer the question concisely and accurately **only** using the provided context.

            ---
            Guidelines:
            1. **Use only the given context. Do not add external knowledge.**
            2. **Provide a direct and precise answer.** No explanations or unnecessary text.
            3. If the context lacks sufficient information, respond with:
               - "The provided context does not contain sufficient information to answer this question."
            4. **No introductions, speculative analysis, or personal opinions.**
            5. **Return only the final answer.**
            6. **Ensure clear and structured formatting.**

            Context:
            {context}

            Question:
            {question}

            Answer:
            """

            final_answer = generate_response(qa_prompt)
            return {"answer": final_answer}

        except Exception as e:
            logger.error(f"Error processing question: {e}")
            raise RuntimeError("Error processing question")

    return ollama_qa


def summarize_document(docs):
    """Generate a deterministic structured summary of a legal document."""
    try:
        document_text = "\n".join([doc.page_content for doc in docs]).strip()

        if not document_text:
            logger.warning("No content available for summarization.")
            raise ValueError("The document is empty or unreadable.")

        summary_prompt = f"""
        You are a legal expert tasked with summarizing the provided legal document concisely while preserving its key details.

        ---
        Guidelines:
        1. **Use only the given context. No external knowledge.**
        2. **Provide a clear and direct summary.** Avoid unnecessary details.
        3. **Retain key legal terms and critical details.**
        4. **No introductions, personal opinions, or speculative analysis.**
        5. **Ensure structured formatting.**
        6. **Return only the summary.**

        Document:
        {document_text}

        Summary:
        """

        summary = generate_response(summary_prompt)

        return {"summary": summary}

    except Exception as e:
        logger.error(f"Error summarizing document: {e}")
        raise RuntimeError("Failed to generate document summary")
