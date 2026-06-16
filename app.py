import streamlit as st

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

    st.info(
        f"You asked: {question}"
    )
