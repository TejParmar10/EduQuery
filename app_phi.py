#Streamlit is a free, open-source Python library that helps developers 
#and data scientists create interactive web applications for machine learning 
#and data science
import streamlit as st
from streamlit_chat import message

#ConversationalRetrievalChain is designed to process queries and generate responses by leveraging 
#information extracted from the associated documents. 
#It represents a form of Retrieval-Augmented Generation (RAG), offering a method to 
#enhance the quality of generated responses through retrieved documents.
from langchain.chains import ConversationalRetrievalChain

#Conversational memory is the mechanism that empowers a chatbot to respond 
#coherently to multiple queries, providing a chat-like experience. 
#It ensures continuity in the conversation, allowing the chatbot to consider 
#past interactions and provide contextually relevant responses.
from langchain.memory import ConversationBufferMemory

# All utility functions
import utils

from PIL import Image

def initialize_session_state():
    """
    Session State is a way to share variables between reruns, for each user session.
    """

    st.session_state.setdefault('history', [])
    st.session_state.setdefault('generated', ["Hello! I am here to provide answers to questions extracted from uploaded PDF files."])
    st.session_state.setdefault('past', ["Hello Buddy!"])

def create_conversational_chain(llm, vector_store):
    """
    Creating conversational chain using Phi-1.5 LLM instance and vector store instance

    Args:
    - llm: Instance of Phi-1.5 model
    - vector_store: Instance of FAISS Vector store having all the PDF document chunks 
    """

    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    
    chain = ConversationalRetrievalChain.from_llm(llm=llm, chain_type='stuff',
                                                 retriever=vector_store.as_retriever(search_kwargs={"k": 2}),
                                                 memory=memory)
    return chain

def display_chat(conversation_chain):
    """
    Streamlit related code where we are passing conversation_chain instance created earlier
    It creates two containers:
    - container: To group our chat input form
    - reply_container: To group the generated chat response

    Args:
    - conversation_chain: Instance of LangChain ConversationalRetrievalChain
    """
    reply_container = st.container()
    container = st.container()

    with container:
        with st.form(key='chat_form', clear_on_submit=True):
            user_input = st.text_input("Question:", placeholder="Ask me questions from uploaded PDF", key='input')
            submit_button = st.form_submit_button(label='Send ⬆️')
        
        if submit_button and user_input:
            generate_response(user_input, conversation_chain)
    
    display_generated_responses(reply_container)

def generate_response(user_input, conversation_chain):
    """
    Generate LLM response based on the user question by retrieving data from Vector Database.
    Also, store information in session states 'past' and 'generated' for conversational type of chats.
    
    Args:
    - user_input: User input as a string.
    - conversation_chain: Instance of ConversationalRetrievalChain.
    """

    with st.spinner('Spinning a snazzy reply...'):
        output = conversation_chat(user_input, conversation_chain, st.session_state['history'])

    st.session_state['past'].append(user_input)
    st.session_state['generated'].append(output)

def conversation_chat(user_input, conversation_chain, history):
    """
    Returns LLM response after invoking model through conversation_chain.
    
    Args:
    - user_input: User input.
    - conversation_chain: Conversational chain.
    - history: Chat history.
    
    Returns:
    - result["answer"]: Response generated from LLM.
    """
    result = conversation_chain.invoke({"question": user_input, "chat_history": history})
    history.append((user_input, result["answer"]))
    return result["answer"]

def display_generated_responses(reply_container):
    """
    Display generated LLM responses to Streamlit Web UI.
    
    Args:
    - reply_container: Streamlit container created at the previous step.
    """
    if st.session_state['generated']:
        with reply_container:
            for i in range(len(st.session_state['generated'])):
                message(st.session_state["past"][i], is_user=True, key=f"{i}_user", avatar_style="adventurer")
                message(st.session_state["generated"][i], key=str(i), avatar_style="bottts")

def main():
    """
    First function to call when we start the Streamlit app.
    """
    initialize_session_state()
    
    st.title("Chat Bot")

    image = Image.open('chatbot.jpg')
    st.image(image, width=150)
    
    hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
    """
    st.markdown(hide_streamlit_style, unsafe_allow_html=True)

    st.sidebar.title("Upload Pdf")
    pdf_files = st.sidebar.file_uploader("", accept_multiple_files=True)
    
    # Step 3: Create instance of Phi-1.5 model    
    llm = utils.create_llm()

    # Step 4: Create Vector Store and store uploaded Pdf file to in-memory Vector Database (FAISS)
    vector_store = utils.create_vector_store(pdf_files)

    if vector_store:
        # Step 5: Create the conversational chain
        chain = create_conversational_chain(llm, vector_store)

        # Step 6: Display Chat UI
        display_chat(chain)
    else:
        print('Initialized App.')

if __name__ == "__main__":
    main()
