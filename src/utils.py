import re
import ollama
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from sympy.physics.units import temperature

from src.config import (
    LLM_MODEL, TEMPERATURE, TOP_P, TOP_K, MAX_TOKENS,
    FREQUENCY_PENALTY, PRESENCE_PENALTY, STOP_SEQUENCES
)


def load_pdf(file_path):
    """Load a PDF file and return documents."""
    return PDFPlumberLoader(file_path).load()


def create_retriever(docs, k=3):
    """Process documents and create a retriever."""
    embedder = HuggingFaceEmbeddings()
    documents = SemanticChunker(embedder).split_documents(docs)
    return FAISS.from_documents(documents, embedder).as_retriever(
        search_type="similarity", search_kwargs={"k": k}
    )


def generate_response(prompt):
    """Generate a response from the Ollama model."""
    response = ollama.chat(
        model=LLM_MODEL,
        messages=[{'role': 'user', 'content': prompt}]
    )
    response_content = response['message']['content']
    return re.sub(r'<think>.*?</think>', '', response_content, flags=re.DOTALL).strip()


def get_qa_chain(retriever):
    """Set up the QA chain using the retriever."""

    def ollama_qa(question):
        documents = retriever.get_relevant_documents(question)
        context = "\n".join([doc.page_content for doc in documents])

        qa_prompt = f"""
        You are a legal expert specializing in the provided context. Your task is to answer the user's question concisely and accurately based only on the given context.

        ---
        Guidelines:
        1. Use only the provided context; do not add external knowledge.
        2. Provide a direct and precise answer. Avoid any reasoning, explanations, or unnecessary text.
        3. If the context lacks sufficient information, state exactly: "The provided context does not contain sufficient information to answer this question."
        4. Format your response clearly and reference the context where applicable.
        5. **Do not include introductory phrases, speculative analysis, or personal opinions.**
        6. **Return only the final answer.**
        7. **Ensure the answer is structured properly.**

        Context: {context}
        Question: {question}

        Answer:
        """

        final_answer = generate_response(qa_prompt).strip()

        return {"answer": final_answer}

    return ollama_qa



def summarize_document(docs):
    """Generate a structured summary of the legal document."""
    document_text = "\n".join([doc.page_content for doc in docs])

    summary_prompt = f"""
    You are a legal expert tasked with summarizing the given legal document concisely while preserving its key details and intent.

    ---
    Guidelines:
    1. Use only the provided context; do not add external knowledge.
    2. Provide a direct and precise summary. Avoid any reasoning, or unnecessary text.
    3. Format your response clearly and reference the context where applicable.
    4. **Do not include introductory phrases,external information, speculative analysis, or personal opinions.**
    5. Maintain clarity and conciseness.
    6. Retain key legal terms and critical details.
    7. **Return only the summary.**
    8. Ensure the summary is structured properly.

    Document:
    {document_text}

    Summary:
    """

    summary = generate_response(summary_prompt).strip()

    # Format the summary properly
    # formatted_summary = f"**Summary:**\n{summary}"

    return {"summary": summary}
