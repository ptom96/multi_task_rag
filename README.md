# multi_task_rag: conversational PDF Q&A

A Streamlit app for chatting with any PDF. You upload a document, ask questions, and ask follow-ups that depend on earlier answers (for example, "Who is Jeff Bezos?" then "What's his age?"). It runs on open-source Hugging Face models, with no paid API.

This was my first RAG build, made with LangChain to understand the end-to-end flow quickly. I later rebuilt each stage by hand, with evaluation, in [rag-foundation](https://github.com/ptom96/rag-foundation).

## How it works

```
PDF upload → PyPDFLoader → CharacterTextSplitter (1000 chars, 200 overlap)
          → BGE embeddings (BAAI/bge-base-en) → Chroma vector store
          → top-3 retrieval → Flan-T5 (google/flan-t5-base)
          → ConversationalRetrievalChain (chat history kept in session state)
```

- **Conversational retrieval:** each follow-up is rewritten into a standalone question using the chat history, so vague questions like "what's his age?" still retrieve the right chunks.
- **Local and free:** BGE embeddings and Flan-T5 run through Hugging Face `transformers`, with no API keys.
- **Streamlit UI:** upload a PDF, type a question, and get the answer in the browser.

## Stack

Python · LangChain · Chroma · Hugging Face Transformers · sentence-transformers (BGE) · Flan-T5 · PyPDF · Streamlit

## Run it

```bash
pip install langchain langchain-community chromadb transformers sentence-transformers torch pypdf streamlit
streamlit run app.py
```

Then open the local URL Streamlit prints, upload a PDF and start asking questions. `multi_rag.ipynb` walks through the same pipeline step by step.

## Files

```
app.py            Streamlit app: upload, index, conversational Q&A
load_docs.py      PDF loading and chunking helpers
multi_rag.ipynb   notebook version of the pipeline
```

## What I learned and what I'd change

- **Framework vs from scratch:** LangChain made the first version fast, but it hid chunking and retrieval decisions. That's why rag-foundation builds each stage by hand.
- **Generator size matters:** Flan-T5-base gives short, extractive answers. A stronger instruction-tuned model would improve multi-sentence and multi-hop answers.
- **Planned fixes:**
  - Cache the index with `st.cache_resource`, so the PDF isn't re-embedded on every question
  - Show source chunks and page numbers with each answer
  - Add a small evaluation set
