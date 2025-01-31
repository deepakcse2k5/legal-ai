import streamlit as st
from utils import load_pdf, create_retriever, get_qa_chain, apply_styles

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

    # User input
    user_input = st.text_input("Ask a question related to the PDF:")
    if user_input:
        with st.spinner("Processing..."):
            response = qa_chain(user_input)["result"]
            st.write("Response:")
            st.write(response)
else:
    st.write("Please upload a PDF file to proceed.")