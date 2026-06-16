import streamlit as st
import os
import numpy as np
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

@st.cache_resource
def load_models():

    embedding_model = SentenceTransformer(
        "sentence-transformers/all-MiniLM-L6-v2"
    )

    model_name = "google/flan-t5-base"

    tokenizer = AutoTokenizer.from_pretrained(
        model_name
    )

    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_name
    )

    return embedding_model, tokenizer, model


embedding_model, tokenizer, model = load_models()

@st.cache_resource
def load_documents():

    documents = []

    data_folder = "data"

    for filename in os.listdir(data_folder):

        if filename.endswith(".pdf"):

            pdf_path = os.path.join(
                data_folder,
                filename
            )

            reader = PdfReader(pdf_path)

            pdf_text = ""

            for page in reader.pages:

                text = page.extract_text()

                if text:
                    pdf_text += text + "\n"

            documents.append({
                "source": filename,
                "content": pdf_text
            })

    return documents


documents = load_documents()

st.success(
    f"Knowledge Base Loaded: {len(documents)} PDFs"
)

def chunk_text(text, chunk_size=500):

    chunks = []

    for i in range(0, len(text), chunk_size):

        chunks.append(
            text[i:i + chunk_size]
        )

    return chunks

@st.cache_resource
def create_chunks():

    all_chunks = []

    for doc in documents:

        chunks = chunk_text(
            doc["content"]
        )

        for chunk in chunks:

            all_chunks.append({
                "source": doc["source"],
                "content": chunk
            })

    return all_chunks


all_chunks = create_chunks()

st.info(
    f"Created {len(all_chunks)} chunks"
)

@st.cache_resource
def create_embeddings():

    texts = [
        chunk["content"]
        for chunk in all_chunks
    ]

    embeddings = embedding_model.encode(
        texts,
        show_progress_bar=False
    )

    return embeddings


embeddings = create_embeddings()



st.success(
    f"Generated {len(embeddings)} embeddings"
)

def retrieve_chunks(question, top_k=3):

    query_embedding = embedding_model.encode(
        [question]
    )[0]

    similarities = np.dot(
        embeddings,
        query_embedding
    )

    top_indices = np.argsort(
        similarities
    )[-top_k:][::-1]

    retrieved_chunks = []

    for idx in top_indices:

        retrieved_chunks.append(
            all_chunks[idx]
        )

    return retrieved_chunks

st.set_page_config(
    page_title="Interview Coach AI",
    page_icon="🎯",
    layout="wide"
)

st.title("🎯 Interview Coach AI")

st.write("""
Ask interview questions across:

- Software Engineering
- Databases
- AI / Machine Learning
- Data Science
- Cyber Security
- Cloud & DevOps
""")


question = st.text_input(
    "Enter your interview question:"
)

if st.button("Ask"):

    chunks = retrieve_chunks(
        question
    )

    st.write("### Retrieved Chunks")

    for chunk in chunks:

        st.write(
            f"Source: {chunk['source']}"
        )

        st.write(
            chunk["content"][:300]
        )

        st.divider()
