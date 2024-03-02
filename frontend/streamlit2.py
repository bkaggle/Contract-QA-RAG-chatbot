import streamlit as st
import os
from dotenv import load_dotenv
from llama_index import GPTVectorStoreIndex, SimpleDirectoryReader, LLMPredictor, PromptHelper, ServiceContext
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma

# Load environment variables
load_dotenv()

# Apply background image
st.markdown(
    """
    <style>
    body {
        background-image: url('https://www.google.com/url?sa=i&url=https%3A%2F%2Fgithub.com%2Fovertake%2FTelegramSwift%2Fissues%2F728&psig=AOvVaw3ItaqEGYp3uXj3OKBWILLy&ust=1709478128549000&source=images&cd=vfe&opi=89978449&ved=0CBMQjRxqFwoTCNDAzueU1oQDFQAAAAAdAAAAABAE');
        background-size: cover;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# Set title
st.title('Lizzy AI')
page_bg_img = f"""<style>
    [data-testid="stAppViewContainer"] > .main {{
    background-image: url("https://i.postimg.cc/4xgNnkfX/Untitled-design.png");
    background-size: cover;
    background-position: center center;
    background-repeat: no-repeat;
    background-attachment: local;
    }}
    [data-testid="stHeader"] {{
    background: rgba(0,0,0,0);
    }}
    </style>
    """
st.markdown(page_bg_img,
      unsafe_allow_html=True)

# Set up Streamlit title and description
st.write('Upload a document and ask a question.')

# File upload functionality
with st.container():
    uploaded_file = st.file_uploader("Upload Document", type=['docx', 'pdf'])

    if uploaded_file is not None:
        # Save the uploaded file temporarily
        with open("temp_file." + uploaded_file.type.split('/')[1], "wb") as f:
            f.write(uploaded_file.getbuffer())

        # Load documents
        temp_file_path = "temp_file." + uploaded_file.type.split('/')[1]
        if uploaded_file.type == 'application/pdf':
            loader = PyPDFLoader(temp_file_path)
        elif uploaded_file.type == 'application/vnd.openxmlformats-officedocument.wordprocessingml.document':
            loader = Docx2txtLoader(temp_file_path)
        else:
            st.error("Unsupported file format")
            st.stop()

        documents = loader.load()

        # Split the documents into chunks
        text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=256, chunk_overlap=20, model_name="gpt-4-1106-preview")
        texts = text_splitter.split_documents(documents)

        # Embeddings
        embeddings = OpenAIEmbeddings()
        store = Chroma.from_documents(texts, embeddings, collection_name="contract")

        # Create chain
        llm = OpenAI(temperature=0)
        chain = RetrievalQA.from_chain_type(llm, retriever=store.as_retriever())

        # Function to handle user input and display bot response
        def chat():
            # Get user input
            user_input = st.text_input("You:")

            # If user sends a message, show it in the chat UI
            if st.button("Send", key='send_button'):
                with st.container():
                    st.markdown(f'<div class="message-container"><p class="user-message">{user_input}</p></div>', unsafe_allow_html=True)

                    # Get bot response
                    bot_response = chain.run(user_input)
                    st.markdown(f'<div class="message-container"><p class="bot-message">{bot_response}</p></div>', unsafe_allow_html=True)

        # Render the chat interface
        chat()

        # Delete the temporary file
        os.remove(temp_file_path)
