import logging
from langchain_community.document_loaders import Docx2txtLoader

class DocumentLoader:
    def __init__(self, docx_path):
        """
        Initialize the DocumentLoader object.

        Parameters:
        - docx_path (str): The path to the document file.
        """
        self.docx_path = docx_path
        self.logger = logging.getLogger(__name__)

    def load_document(self):
        """
        Load a document using the specified loader.

        Returns:
        - document (str): The loaded document content.
        """
        try:
            loader = Docx2txtLoader(self.docx_path)
            document = loader.load()
            self.logger.info("Document loaded successfully.")

        except Exception as e:
            self.logger.error(f"Error loading document: {str(e)}")
