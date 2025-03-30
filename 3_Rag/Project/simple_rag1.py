from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
import os
from dotenv import load_dotenv

load_dotenv()  # Load environment variables
api_key = os.getenv('groq_api')
faiss_path = r'F:\\machine learning\\Prodigal_Internship\\3_Rag\\Project\\faiss_index'  

prompt_template = """
Answer the question based only on the following context:
{context}
you have to elaborate the answer very well do not hallucinate give just on to the point answers
the answer can be of one or two paragraph Not more than that and contains atleast 5 lines in a paragraph.

answer the question on the above context: {question}\n
Note: just give the answer at the end dont tell according to this context or anything

"""

def main(query):
    
    # Preparing the database
    embeddings = HuggingFaceEmbeddings()
    db = FAISS.load_local(faiss_path, embeddings, allow_dangerous_deserialization=True) 

    # Searching in db
    results = db.similarity_search_with_relevance_scores(query, k=3)

    if len(results) == 0 or results[0][1] < 0.3:
        print("Unable to find matching results")
        return

    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])

    prompt_template_obj = ChatPromptTemplate.from_template(prompt_template) #changed variable name to avoid confusion

    prompt = prompt_template_obj.format(context=context_text, question=query)
    # print("prompt is given by \n",prompt)

    #gets the api key from the .env file.
    if not api_key:
        print("GROQ_API_KEY not found in environment variables.")
        return

    model = ChatGroq(model="llama-3.1-8b-instant",api_key=api_key)
    response = model.invoke(prompt)

    sources = [doc.metadata.get('source', None) for doc, _score in results]
    formatted_response = f"Response: {response.content}\n\nSources: {sources[0]}" 
    print('formatted response is: \n',formatted_response)
    return formatted_response

if __name__ == "__main__":
    main()