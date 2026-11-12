
import streamlit as st
import os
from pathlib import Path
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate

# Project configuration
BASE_DIR = Path(__file__).parent
load_dotenv(BASE_DIR / ".env")

st.set_page_config(
    page_title="Medical Q&A Chatbot",
    page_icon="🩺",
    layout="centered"
)

# Application heading
st.title("🩺 Medical Q&A Chatbot")
st.write("Ask questions about medical conditions and health topics.")

st.info(
    "Educational information only. This chatbot cannot diagnose "
    "diseases or prescribe medicines."
)

# Load database and language model
@st.cache_resource
def load_chatbot():

    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        raise ValueError("Groq API key not found in .env file.")

    vector_dir = BASE_DIR / "vectorstore" / "chroma_db"

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore = Chroma(
        collection_name="medical_qa",
        embedding_function=embeddings,
        persist_directory=str(vector_dir)
    )

    llm = ChatGroq(
        model="openai/gpt-oss-20b",
        temperature=0.2,
        max_tokens=1024,
        api_key=api_key
    )

    return vectorstore, llm


medical_prompt = ChatPromptTemplate.from_template("""
You are a Medical Information Assistant.

Answer using ONLY the medical context below.

Rules:
1. Do not invent medical facts.
2. Do not diagnose diseases.
3. Do not prescribe medicines or treatments.
4. If the context is insufficient, say so clearly.
5. Use simple English.
6. Recommend consulting a qualified healthcare professional.
7. For emergency symptoms, advise seeking emergency care.

Medical Context:
{context}

Question:
{question}

Answer:
""")


def get_medical_answer(question, vectorstore, llm):

    docs = vectorstore.similarity_search(question, k=4)

    if not docs:
        return (
            "The medical knowledge base does not contain "
            "sufficient information for this question."
        ), []

    context = "\n\n".join(doc.page_content for doc in docs)

    prompt = medical_prompt.format(
        context=context,
        question=question
    )

    response = llm.invoke(prompt)

    return response.content, docs


# Initialize chatbot
try:
    vectorstore, llm = load_chatbot()
    st.success("Medical knowledge base connected successfully!")

except Exception as error:
    st.error(f"Unable to initialize chatbot: {error}")
    st.stop()


# Chat history
if "messages" not in st.session_state:
    st.session_state.messages = []


for message in st.session_state.messages:

    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        if message.get("sources"):
            with st.expander("View medical sources"):
                for i, source in enumerate(message["sources"], 1):
                    st.markdown(f"**Source {i}**")
                    st.write(source.page_content[:500])


# User question
question = st.chat_input("Ask your medical question...")

if question:

    st.session_state.messages.append({
        "role": "user",
        "content": question
    })

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):

        with st.spinner("Searching medical knowledge base..."):

            try:
                answer, sources = get_medical_answer(
                    question,
                    vectorstore,
                    llm
                )

                st.markdown(answer)

                with st.expander("View medical sources"):
                    for i, source in enumerate(sources, 1):
                        st.markdown(f"**Source {i}**")
                        st.write(source.page_content[:500])

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "sources": [
                        {"content": doc.page_content}
                        for doc in sources
                    ]
                })

            except Exception as error:
                st.error(f"An error occurred: {error}")


st.caption(
    "Medical Q&A Chatbot | Built using Python, LangChain, "
    "Hugging Face, ChromaDB and Groq"
)
