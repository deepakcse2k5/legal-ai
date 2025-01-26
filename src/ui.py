import chromadb
import streamlit as st

# Initialize the Chroma Client (new architecture, no deprecated arguments)
client = chromadb.Client()

# Streamlit UI
st.title("Chroma DB Explorer")

# List collections
collections = client.list_collections()
st.sidebar.header("Collections")
selected_collection = st.sidebar.selectbox("Select a collection", [col.name for col in collections])

if selected_collection:
    collection = client.get_collection(selected_collection)
    st.header(f"Documents in Collection: {selected_collection}")


    # Display documents
    documents = collection.get(include=["documents", "metadatas"])
    if "documents" in documents:
        for i, (doc, meta) in enumerate(zip(documents["documents"], documents["metadatas"])):
            with st.expander(f"Document {i+1}"):
                st.subheader("Content")
                st.text(doc)
                st.subheader("Metadata")
                st.json(meta)
    else:
        st.write("No documents found in this collection.")

    # Query the collection
    st.header("Query Collection")
    query = st.text_input("Enter your query:")
    if query:
        results = collection.query(query_texts=[query], n_results=3)
        st.write("Results:")
        for doc, score, meta in zip(results["documents"], results["scores"], results["metadatas"]):
            with st.expander(f"Relevance: {score:.2f}"):
                st.text(doc)
                st.json(meta)
