import logging
from functools import wraps
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.llms import OpenAI
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.storage import InMemoryStore
from langchain.retrievers import ParentDocumentRetriever
#from models import RagResponse
#from document_loader import DocumentLoader

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
    def generate_embeddings_store(self, texts):
        
            
            # parent_splitter = RecursiveCharacterTextSplitter(
            # separators=["\n\n", "\n", ".", " "],
            # chunk_size=600,
            # chunk_overlap=10
            # )
            # child_splitter = RecursiveCharacterTextSplitter(
            #     separators=["\n\n", "\n", ".", " "],
            #     chunk_size=1900,
            #     chunk_overlap=10
            # )

            # vectorstore = Chroma(collection_name="contract", embedding_function=embeddings())
            # store = InMemoryStore()

            # parent_document_retriever = ParentDocumentRetriever(
            #     vectorstore=vectorstore,
            #     docstore=store,
            #     child_splitter=child_splitter,
            #     parent_splitter=parent_splitter,
            #     search_kwargs={"k": 10},
                
            # )
            # parent_document_retriever.add_documents(texts)
            
            # return parent_document_retriever
        try:
            embeddings = OpenAIEmbeddings()
            store = Chroma.from_documents(texts, embeddings, collection_name="contract")
            llm = OpenAI(temperature=0)
            retriever=store.add_documents()
            #add_documents(docs)
            return RetrievalQA.from_chain_type(llm, retriever=retriever)
        except Exception as e:
            logger.error(f"Error occurred while generating embeddings: {str(e)}")
            return None

   