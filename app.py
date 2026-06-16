import streamlit as st
import os
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

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
