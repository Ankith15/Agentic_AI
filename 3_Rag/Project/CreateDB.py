from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS  # Using FAISS as an alternative
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os
import shutil
load_dotenv()

data_path = r"C:\\Users\\USER\\Downloads\\OTC_ADDDF_2022.pdf"
faiss_path = "faiss_index"  # Changed path for FAISS

def generate_data_store():
    documents = load_documents()
    chunks = split_text(documents)
    save_to_faiss(chunks)  # Changed function call

def load_documents():
    loader = PyPDFLoader(data_path)
    documents = loader.load()
    return documents

def main():
    generate_data_store()

def split_text(documents: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=100,
        length_function=len,
        add_start_index=True,
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Split {len(documents)} documents into {len(chunks)} chunks.")

    document = chunks[10]
    print(document.page_content)
    print(document.metadata)
    return chunks

def save_to_faiss(chunks: list[Document]):  # Changed function name
    # Clear out the database first
    if os.path.exists(faiss_path):
        shutil.rmtree(faiss_path)

    db = FAISS.from_documents(  # Changed to FAISS
        chunks, HuggingFaceEmbeddings()
    )

    db.save_local(faiss_path)  # Changed save method
    print(f"Saved {len(chunks)} chunks to {faiss_path}.")

if __name__ == "__main__":
    main()