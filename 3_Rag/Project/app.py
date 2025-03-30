import streamlit as st
from simple_rag1 import main
from langchain_core.messages import HumanMessage, AIMessage

st.set_page_config(page_title='Rag Chatbot', page_icon="🎃", layout='wide', initial_sidebar_state='expanded')

@st.cache_data(ttl=3600)  # Cache expires in 1 hour
def store_response(query: str, response: str):
    """
    Stores the query and response in the cache.

    Parameters:
    - query (str): The user query
    - response (str): The generated response
    """
    # Retrieve existing cache
    cache_data = st.session_state.get("query_cache", {})
    
    # Store new response
    cache_data[query] = response
    st.session_state["query_cache"] = cache_data

def get_cached_response(query: str):
    """
    Retrieves a cached response for a given query.

    Parameters:
    - query (str): The user query

    Returns:
    - str: Cached response if found, else None
    """
    cache_data = st.session_state.get("query_cache", {})
    return cache_data.get(query, None)
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


left, space, right = st.columns([8, 1, 12])

with left:
    st.image(r"F:\\machine learning\\Prodigal_Internship\\3_Rag\\images\\rag.jpg", use_container_width=True)
    st.markdown("<p style='color: black;font-size: 30px;font-weight: bold;margin-bottom: 1px;'>Data Science Experts</p>", unsafe_allow_html=True)
    st.markdown("<p style='color: black;font-size: 20px;font-weight: bold;margin-bottom: 1px;'>Ask queries related to Data Science</p>", unsafe_allow_html=True)

with space:
    st.markdown("<br><br><br>", unsafe_allow_html=True)  

with right:
    st.title("Data Science Professor ☠️")

    chat_container= st.container()

    with chat_container:
        for msg in st.session_state.chat_history:
            if isinstance(msg,HumanMessage):
                with st.chat_message("user"):
                    st.markdown(msg.content)
            elif isinstance(msg,AIMessage):
                with st.chat_message("assistant"):
                    st.markdown(msg.content)

    
    
    user_query = st.chat_input("Ask me questions related to Data Science 🎃")

    if user_query:
        with st.spinner("Fetching the details..."):
            cached_response = get_cached_response(user_query)
            st.session_state.chat_history.append(HumanMessage(user_query))
            with chat_container:
                with st.chat_message("user"):  
                    st.markdown(user_query)
            if cached_response:
                response = cached_response
            else:
                response = main(user_query)
                store_response(user_query,response)
            st.session_state.chat_history.append(AIMessage(response))  
            with chat_container:
                with st.chat_message("assistant"):
                    st.markdown(response)

            