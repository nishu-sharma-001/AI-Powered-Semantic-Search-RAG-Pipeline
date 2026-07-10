import os
import chromadb
import google.generativeai as genai
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure Gemini
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# Load Gemini Model
gemini_model = genai.GenerativeModel("gemini-2.5-flash")

# Load Embedding Model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# Connect to ChromaDB
client = chromadb.PersistentClient(path="./chroma_db")

collection = client.get_or_create_collection(
    name="academic_notes"
)

def generate_answer(query):

    # Convert query into embedding
    query_embedding = embedding_model.encode(query).tolist()

    # Retrieve top 3 relevant chunks
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=3
    )

    contexts = []

    if results["metadatas"] and len(results["metadatas"][0]) > 0:

        for item in results["metadatas"][0]:
            contexts.append(item["text"])

    context = "\n\n".join(contexts)

    prompt = f"""
You are an AI academic assistant.

Answer ONLY using the provided context.

If the answer is not available in the context, say:
"I could not find the answer in the uploaded notes."

Context:
{context}

Question:
{query}

Answer:
"""

    response = gemini_model.generate_content(prompt)

    return response.text
