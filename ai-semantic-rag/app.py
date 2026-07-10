import streamlit as st
from rag import generate_answer

# Page Configuration
st.set_page_config(
    page_title="Academic AI Assistant",
    page_icon="📚"
)

# Title
st.title("📚 Academic AI Assistant")

st.markdown(
    "Ask questions from your uploaded PDF notes using **ChromaDB + Gemini RAG**."
)

# User Input
query = st.text_input("Ask your question:")

# Generate Answer
if query:

    with st.spinner("Generating Answer..."):

        answer = generate_answer(query)

        st.subheader("Answer")

        st.write(answer)
