import streamlit as st
from sentence_transformers import SentenceTransformer
import chromadb

# Page Configuration
st.set_page_config(
    page_title="Academic AI Search",
    page_icon="📚"
)

st.title("📚 Academic Semantic Search")
st.markdown("Powered by **ChromaDB**")

# Load Embedding Model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Connect to Persistent ChromaDB
client = chromadb.PersistentClient(path="./chroma_db")

# Create / Load Collection
collection = client.get_or_create_collection(
    name="academic_notes"
)

# User Input
query = st.text_input("Ask a question from your notes:")

if query:
    with st.spinner("Searching..."):

        # Convert Query to Embedding
        query_vector = model.encode(query).tolist()

        # Semantic Search
        results = collection.query(
            query_embeddings=[query_vector],
            n_results=3
        )

        st.subheader("Relevant Results")

        if results["metadatas"] and len(results["metadatas"][0]) > 0:
            for res in results["metadatas"][0]:
                st.write(f"📖 {res['text']}")
                st.divider()
        else:
            st.warning("No relevant results found.")
