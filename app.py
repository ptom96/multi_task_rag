# app.py
from load_docs import load_documents,chunk_documents
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import HuggingFaceHub
import os
import streamlit as st
from langchain.chains import RetrievalQA
from langchain_community.llms import HuggingFaceEndpoint
from transformers import pipeline
from langchain_community.llms import HuggingFacePipeline
from langchain.chains import ConversationalRetrievalChain

# 1. Title of the app
st.title("📄 Tom’s RAGtime Show 🎤📚🤖")

# 2. File uploader
uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

if uploaded_file:
    # Load PDF
    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.read())

    documents = load_documents("temp.pdf")
    # 2. Split into chunks
    chunks = chunk_documents(documents)

    # Create embeddings + vector DB
    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-base-en")
    vectorstore = Chroma.from_documents(chunks, embeddings)

    # Build HuggingFace LLM
    hf_pipeline = pipeline("text2text-generation", model="google/flan-t5-base", device=0)
    llm = HuggingFacePipeline(pipeline=hf_pipeline)

    # RetrievalQA chain
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    qa_chain = ConversationalRetrievalChain.from_llm(llm=llm, retriever=retriever)

    # Maintain chat history across queries
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # 3. Input box for user query
    query = st.text_input("Ask a question about the document:")

    # 4. Generate answer
    if query:
        with st.spinner("Generating answer..."):
            result = qa_chain.invoke({
                "question": query,
                "chat_history": st.session_state.chat_history
            })

        # Append to history
        st.session_state.chat_history.append((query, result["answer"]))

        st.write("**Answer:**", result["answer"])
