from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter

def load_documents(file_path):
    loader = PyPDFLoader(file_path)
    docs = loader.load()
    return docs


def chunk_documents(documents, chunk_size=1000, overlap=200):
    splitter = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)
    return splitter.split_documents(documents)