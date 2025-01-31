import streamlit as st
import torch
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate
from langchain.chains.llm import LLMChain
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chains import RetrievalQA

def apply_styles():
    """Apply custom Streamlit styles."""
    st.markdown(
        """
        <style>
        .stApp { background-color: #F5F5F5; color: #4561e9; }
        .stButton>button { background-color: #1E90FF; color: white; border-radius: 5px; padding: 10px 20px; font-size: 16px; }
        .stTextInput>div>div>input { border: 2px solid #1E90FF; border-radius: 5px; padding: 10px; font-size: 16px; }
        .stFileUploader>div>div>div>button { background-color: #FF6347; color: white; border-radius: 5px; padding: 10px 20px; font-size: 16px; }
        </style>
        """,
        unsafe_allow_html=True,
    )

def load_pdf(file_path):
    """Load a PDF file properly."""
    return PDFPlumberLoader(file_path).load()

def create_retriever(docs):
    """Process documents and create a retriever."""
    embedder = HuggingFaceEmbeddings()
    documents = SemanticChunker(embedder).split_documents(docs)
    return FAISS.from_documents(documents, embedder).as_retriever(search_type="similarity", search_kwargs={"k": 3})

def get_qa_chain(retriever):
    """Set up the QA chain using the retriever with GPU fallback."""
    model_name = "deepseek-r1:1.5b"
    llm = Ollama(model=model_name)

    qa_prompt = PromptTemplate.from_template(
        """
        You are a legal expert specializing in the provided context. Your task is to answer the user's question based solely on the provided content.

        ---
        Guidelines:
        1. Use only the provided context; do not add external knowledge.
        2. Be concise and accurate. If the context lacks sufficient information, state: "The provided context does not contain sufficient information to answer this question."
        3. Format your response clearly and reference the context where applicable.
        4. Avoid speculation and do not provide legal advice beyond the context.

        Context: {context}
        Question: {question}

        Helpful Answer:
        """
    )
    return RetrievalQA(
        combine_documents_chain=StuffDocumentsChain(
            llm_chain=LLMChain(llm=llm, prompt=qa_prompt, verbose=True),
            document_variable_name="context",
            document_prompt=PromptTemplate(input_variables=["page_content", "source"], template="Context:\\n{page_content}\\nSource: {source}"),
        ),
        retriever=retriever,
        verbose=True,
        return_source_documents=True,
    )

def summarize_document(docs):
    """Generate a summary of the document with GPU fallback."""
    model_name = "deepseek-r1:1.5b"
    llm = Ollama(model=model_name)

    summary_prompt = PromptTemplate.from_template(
        """
        Summarize the following legal document concisely, preserving key details and intent.

        ---
        Document: {document}

        Summary:
        """
    )
    llm_chain = LLMChain(llm=llm, prompt=summary_prompt, verbose=True)
    return llm_chain.run({"document": " ".join([doc.page_content for doc in docs])})
