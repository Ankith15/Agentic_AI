from langchain.chains import RetrievalQA
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers.ensemble import EnsembleRetriever

embeddings = HuggingFaceEmbeddings()
loader = TextLoader(r"F:\\machine learning\\Prodigal_Internship\\3_Rag\\Files\\text_data.txt")
documents = loader.load()
#let us split this loaded file to smaller chunks

text_splitter = RecursiveCharacterTextSplitter(chunk_size = 100,chunk_overlap = 10,length_function = len,add_start_index = True)
split_docs = text_splitter.split_documents(documents) 
# Load FAISS index
vector_db = FAISS.load_local(r'F:\\machine learning\\Prodigal_Internship\\3_Rag\\Codes\\faiss_index', embeddings, allow_dangerous_deserialization=True)
retriever = vector_db.as_retriever(search_kwargs={"k": 20})
bm25_retri = BM25Retriever.from_documents(split_docs)

hybrid = EnsembleRetriever(retrievers = [retriever,bm25_retri],weights = [0.5,0.5])

query = 'who was the indian captain?'
retrieved_docs = hybrid.invoke(query)

for doc in retrieved_docs[0]:
    print(f"retrieved documents are \n {doc.content}")
