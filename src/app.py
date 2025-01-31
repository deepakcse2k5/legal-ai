import streamlit as st
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate
from langchain.chains.llm import LLMChain
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chains import RetrievalQA
from utils import load_pdf, create_retriever, get_qa_chain, summarize_document, apply_styles

# Apply Streamlit styles
apply_styles()

# App title
st.title("Legal Help Question Answering System")

# File upload
uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")
if uploaded_file:
    docs = load_pdf(uploaded_file)
    retriever = create_retriever(docs)
    qa_chain = get_qa_chain(retriever)

    # Summarization
    with st.spinner("Generating Summary..."):
        summary = summarize_document(docs)
        st.subheader("Document Summary:")
        st.write(summary)

    # User input
    user_input = st.text_input("Ask a question related to the PDF:")
    if user_input:
        with st.spinner("Processing..."):
            response = qa_chain(user_input)["result"]
            st.write("Response:")
            st.write(response)
else:
    st.write("Please upload a PDF file to proceed.")