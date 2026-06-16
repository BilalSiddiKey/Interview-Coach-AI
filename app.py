import streamlit as st
import os
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

    st.success(
        f"You asked: {question}"
    )
