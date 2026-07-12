import os
import shutil
import streamlit as st

from rag_pipeline import config
from rag_pipeline.document_loader import DocumentLoader
from rag_pipeline.text_chunker import TextChunker
from rag_pipeline.embedding_engine import EmbeddingEngine
from rag_pipeline.vector_store import VectorStore
from rag_pipeline.answer_generator import AnswerGenerator

st.set_page_config(
    page_title="Document Question Answering System (RAG)",
    page_icon="📄",
    layout="wide"
)

st.title("📄 Document Question Answering System (RAG)")
st.write("Upload your PDF/Text documents and ask questions.")

# ----------------------------------------------------
# Create sample_documents folder if not present
# ----------------------------------------------------

os.makedirs(config.SOURCE_DOCUMENTS_FOLDER, exist_ok=True)

# ----------------------------------------------------
# Upload Files
# ----------------------------------------------------

uploaded_files = st.file_uploader(
    "Upload PDF or TXT files",
    type=["pdf", "txt", "md"],
    accept_multiple_files=True
)

# ----------------------------------------------------
# Build Knowledge Base
# ----------------------------------------------------

if st.button("Build Knowledge Base"):

    # Clear previous files
    for file in os.listdir(config.SOURCE_DOCUMENTS_FOLDER):
        os.remove(os.path.join(config.SOURCE_DOCUMENTS_FOLDER, file))

    # Save uploaded files
    for uploaded_file in uploaded_files:
        save_path = os.path.join(
            config.SOURCE_DOCUMENTS_FOLDER,
            uploaded_file.name
        )

        with open(save_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

    with st.spinner("Loading documents..."):

        document_loader = DocumentLoader(config.SOURCE_DOCUMENTS_FOLDER)
        documents = document_loader.load_all_documents()

    st.success(f"Loaded {len(documents)} document(s).")

    with st.spinner("Chunking documents..."):

        chunker = TextChunker(
            config.CHUNK_SIZE_IN_CHARACTERS,
            config.CHUNK_OVERLAP_IN_CHARACTERS
        )

        chunks = chunker.chunk_all_documents(documents)

    st.success(f"Created {len(chunks)} chunks.")

    with st.spinner("Generating embeddings... (First run may take a few minutes)"):

        embedding_engine = EmbeddingEngine(
            config.EMBEDDING_MODEL_NAME
        )

        chunk_texts = [chunk["text"] for chunk in chunks]

        embeddings = embedding_engine.generate_embeddings(chunk_texts)

    vector_store = VectorStore(embeddings.shape[1])
    vector_store.add_chunks(chunks, embeddings)

    answer_generator = AnswerGenerator(
        config.ANSWER_GENERATION_MODEL_NAME
    )

    st.session_state.embedding_engine = embedding_engine
    st.session_state.vector_store = vector_store
    st.session_state.answer_generator = answer_generator

    st.success("✅ Knowledge Base Ready!")

# ----------------------------------------------------
# Ask Question
# ----------------------------------------------------

if "vector_store" in st.session_state:

    st.divider()

    question = st.text_input(
        "Ask a Question"
    )

    if st.button("Get Answer"):

        if question.strip() == "":
            st.warning("Please enter a question.")

        else:

            with st.spinner("Searching documents..."):

                question_embedding = (
                    st.session_state.embedding_engine
                    .generate_single_embedding(question)
                )

                retrieved_chunks = (
                    st.session_state.vector_store
                    .search_similar_chunks(
                        question_embedding,
                        config.NUMBER_OF_CHUNKS_TO_RETRIEVE
                    )
                )

                answer = (
                    st.session_state.answer_generator
                    .generate_answer(
                        question,
                        retrieved_chunks
                    )
                )

            st.subheader("Answer")

            st.success(answer)

            st.subheader("Source Documents")

            sources = sorted(
                {chunk["source_file"] for chunk in retrieved_chunks}
            )

            for source in sources:
                st.write(f"📄 {source}")

            with st.expander("Retrieved Context"):

                for i, chunk in enumerate(retrieved_chunks, start=1):

                    st.markdown(f"### Chunk {i}")

                    st.write(chunk["text"])