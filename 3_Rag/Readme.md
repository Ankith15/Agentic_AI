
# **RAG Chatbot with Caching** 🎃  
A **Streamlit-powered chatbot** that leverages **Retrieval-Augmented Generation (RAG)** to answer **Data Science-related queries** efficiently. It includes:  
- **Basic RAG**: Retrieves relevant context before generating a response.  
- **Advanced RAG**: Enhances retrieval accuracy with additional techniques.  
- **Cache RAG**: Caches responses to improve speed and reduce computation costs.  

---

## **Features**
**Retrieval-Augmented Generation (RAG)** for accurate responses.  
**Advanced RAG techniques** for better document retrieval.  
**Caching with `st.cache_data`** to store and reuse previous responses.  
**Streamlit chat interface** with persistent chat history.  

##  **Project Video**

https://github.com/user-attachments/assets/0608b03c-1461-4245-8cc3-2b4a2e02f82e


## **Installation**
1 **Clone the repository**  
```sh
git clone https://github.com/your-username/RAG_Chatbot.git
cd RAG_Chatbot
```

2 **Create a virtual environment (Optional but recommended)**  
```sh
python -m venv venv
venv\Scripts\activate     
```

3 **Install dependencies**  
```sh
pip install -r requirements.txt
```

---

## **How to Run**
Run the Streamlit app:  
```sh
streamlit run app.py
```
This will start the chatbot on `http://localhost:8501/`. 🎃

---

## **How It Works**
### **1 Basic RAG**
- Extracts **relevant documents** from a knowledge base.  
- Uses **LLM (Language Model)** to generate responses based on retrieved content.  

### **2 Advanced RAG**
- Enhances retrieval accuracy using **embedding models** and **vector databases**.  
- Improves **context relevance** before generating responses.  

### **3 Cache RAG**
- Uses **`st.cache_data`** to store responses.  
- Prevents re-running expensive queries for **previously asked questions**.  
- Reduces **response time** for repeated queries.  

---

## **Example Queries**
```sh
🔹 "What is gradient boosting in machine learning?"
🔹 "Explain PCA in simple terms."
🔹 "How does RAG improve chatbot responses?"
```

---

## **Future Improvements**
- 🔹 Integrate **OpenAI / Hugging Face models** for better responses.  
- 🔹 Use a **vector database** like FAISS / Pinecone for **efficient retrieval**.  
- 🔹 Add **multi-turn conversation memory** for better context handling.  

## **Contributing**
Feel free to fork this repository and submit pull requests with improvements.

## **Note**
The speed of the agent might vary based on your system configuration.

