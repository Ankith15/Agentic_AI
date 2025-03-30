from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

#defining the loader
loader = TextLoader(r"F:\\machine learning\\Prodigal_Internship\\3_Rag\\Files\\text_data.txt")
documents = loader.load()
#let us split this loaded file to smaller chunks

text_splitter = RecursiveCharacterTextSplitter(chunk_size = 100,chunk_overlap = 10,length_function = len,add_start_index = True)
split_docs = text_splitter.split_documents(documents) 
embeddings = HuggingFaceEmbeddings()
vector_db = FAISS.from_documents(split_docs,embeddings)
vector_db.save_local("faiss_index")
print("document successfully stored in the database")