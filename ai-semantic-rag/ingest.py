import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import chromadb

# 1. Load and Chunk PDF
def ingest_data(pdf_path):
    print(f"Loading {pdf_path}...")

    # Load PDF
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    # Split into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    chunks = text_splitter.split_documents(documents)

    # 2. Setup ChromaDB Client
    client = chromadb.Client()

    # Create Collection
    collection = client.get_or_create_collection(
        name="academic_notes"
    )

    # Load Embedding Model
    model = SentenceTransformer("all-MiniLM-L6-v2")

    # 3. Store Chunks in ChromaDB
    for i, chunk in enumerate(chunks):

        embedding = model.encode(chunk.page_content).tolist()

        collection.add(
            ids=[f"id_{i}"],
            embeddings=[embedding],
            metadatas=[{"text": chunk.page_content}]
        )

    print(f"Successfully stored {len(chunks)} chunks in ChromaDB!")

if __name__ == "__main__":

    if os.path.exists("notes.pdf"):
        ingest_data("notes.pdf")
    else:
        print("Error: Please upload a file named 'notes.pdf' first.")
