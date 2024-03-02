import streamlit as st
from htmltemplates import css,bot2_template, user3_template
from io import BytesIO
import os
from dotenv import load_dotenv
load_dotenv()

OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')

import sys
sys.path.append('../')

from scripts.document_loader import DocumentLoader
from scripts.chunk_embed import TextProcessor
from scripts.chat import ChatModel
from scripts.chain import ConversationChain


GPT_MODEL_NAME = 'gpt-3.5-turbo'
CHUNK_SIZE = 600
CHUNK_OVERLAP = 10
#print(OPENAI_API_KEY)

def main():
    st.title("Lizzy AI")
    st.write(css, unsafe_allow_html=True)
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

    st.markdown(page_bg_img, unsafe_allow_html=True)

    
    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.header("Contract Analyzer")
    
    st.sidebar.subheader("Your Documents")
    uploaded_file = st.sidebar.file_uploader("Upload contract", type=["pdf", "docx"])
    if uploaded_file is not None:
        with st.spinner('Loading and processing the document...'):
            doc_loader = DocumentLoader(uploaded_file)
            documents = doc_loader.load_document()
            

            
            #embeddings_creator = EmbeddingsCreator()
            #embeddings = embeddings_creator.create_embeddings(OPENAI_API_KEY)
            text_processor = TextProcessor(documents)
            

            chunks = text_processor.chunk_text()
            qa_chain = text_processor.generate_embeddings_store(chunks)

            
            # chat_model = ChatModel()
            # chat_model = chat_model.initialize_chat_model(OPENAI_API_KEY, GPT_MODEL_NAME)
            
            
            # qa_chain = ConversationChain()
            # qa_chain = qa_chain.create_retrieval_qa_chain(chat_model, retriever)
            
            st.success("Document processed successfully!")

            st.subheader("Ask Lizzy:")
            user_question = st.text_input("Type your question here:")

            if st.button("Get Answer"):
                if user_question:
                    # Assuming chain is your ConversationChain object
                    answer = qa_chain.run(user_question)
                    st.write("Answer from Lizzy:")
                    st.write(answer)  # Print the answer here
        
        
if __name__ == '__main__':
    main()