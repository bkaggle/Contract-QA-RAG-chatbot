import logging
from functools import wraps
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA

# Setup logging configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Decorator for logging function calls
def log_function_call(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        logger.info(f"Calling function: {func.__name__}")
        return func(*args, **kwargs)
    return wrapper

class TextProcessor:
    def __init__(self, documents):
        self.documents = documents

    @log_function_call
    def chunk_text(self):
        try:
            text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(chunk_size=256, chunk_overlap=20, model_name="gpt-4-1106-preview")
            texts = text_splitter.split_documents(self.documents)
            return texts
        except Exception as e:
            logger.error(f"Error occurred while chunking text: {str(e)}")
            return []

    @log_function_call
    def generate_embeddings(self, texts):
        try:
            embeddings = OpenAIEmbeddings()
            store = Chroma.from_documents(texts, embeddings, collection_name="contract")
            return store
        except Exception as e:
            logger.error(f"Error occurred while generating embeddings: {str(e)}")
            return None

    @log_function_call
    def retrieve_response(self, store):
        try:
            llm = OpenAI(temperature=0)
            chain = RetrievalQA.from_chain_type(llm, retriever=store.as_retriever())
            return chain
        except Exception as e:
            logger.error(f"Error occurred while retrieving response: {str(e)}")
            return None