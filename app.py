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
