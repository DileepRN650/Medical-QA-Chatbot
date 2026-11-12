# Medical Q&A Chatbot

## 1. Project Overview

The Medical Q&A Chatbot is an AI-powered application
that answers medical-related questions using a
medical question-answer dataset.

It uses Retrieval-Augmented Generation (RAG) to
retrieve relevant medical information and generate
answers in simple English.

## 2. Project Objectives

- Build an AI-based medical question-answering system.
- Retrieve relevant information from medical documents.
- Generate context-based answers using an LLM.
- Provide a simple web interface using Streamlit.

## 3. Technologies Used

- Python
- Jupyter Notebook
- Pandas and NumPy
- Matplotlib
- LangChain
- Hugging Face Embeddings
- ChromaDB
- Groq LLM
- Streamlit

## 4. Dataset

MedQuAD (Medical Question Answering Dataset).

## 5. Project Architecture

1. Download medical dataset.
2. Extract and clean medical questions and answers.
3. Split text into smaller chunks.
4. Generate embeddings.
5. Store embeddings in ChromaDB.
6. Retrieve relevant medical documents.
7. Generate answers using Groq LLM.
8. Display answers through Streamlit.

## 6. Project Structure

- data/ : Medical dataset
- vectorstore/ : ChromaDB database
- outputs/ : Evaluation results
- app.py : Streamlit application
- requirements.txt : Required Python libraries
- .env : Groq API key (private)
- .gitignore : Files excluded from GitHub

## 7. Installation

Install the required libraries:

    pip install -r requirements.txt

Configure your Groq API key in the .env file:

    GROQ_API_KEY=your_groq_api_key

Do not share or upload your .env file.

## 8. Run the Application

Start the Streamlit application:

    streamlit run app.py

Open:

    http://localhost:8501

## 9. Features

- Medical question answering
- Semantic document retrieval
- Context-based answer generation
- Medical source display
- Interactive Streamlit interface

## 10. Medical Disclaimer

This chatbot is for educational and informational
purposes only. It does not provide medical diagnosis,
prescriptions, or professional medical advice.

Always consult a qualified healthcare professional
for personal medical concerns.
