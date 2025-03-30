# from langchain_community.llms import GroqLLM
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA
import os 
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader
from dotenv import load_dotenv
load_dotenv()
api_key = os.getenv("groq_api")
embeddings = HuggingFaceEmbeddings()

# Load FAISS index
vector_db = FAISS.load_local(r'F:\\machine learning\\Prodigal_Internship\\3_Rag\\Codes\\faiss_index', embeddings, allow_dangerous_deserialization=True)

# Create retriever with top 2 results
retriever = vector_db.as_retriever(search_kwargs={"k": 20})

llm = ChatGroq(groq_api_key=api_key, model_name="llama3-8b-8192")
qa_chain = RetrievalQA.from_chain_type(llm,retriever=retriever)
query= "can you see any menctioning about Mr. Modi In this document?"
response= qa_chain.run(query)

print(f"AI Response is \n {response}")