from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader

# Load embeddings
embeddings = HuggingFaceEmbeddings()

# Load FAISS index
vector_db = FAISS.load_local(r'F:\\machine learning\\Prodigal_Internship\\3_Rag\\Codes\\faiss_index', embeddings, allow_dangerous_deserialization=True)

# Create retriever with top 2 results
retriever = vector_db.as_retriever(search_kwargs={"k": 4})

# Query the retriever
query = 'scientist?'
retrieved_docs = retriever.get_relevant_documents(query)

# Print the top 2 retrieved results
for doc in retrieved_docs:
    print(doc.page_content)