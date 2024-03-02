from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.prompts import ChatPromptTemplate
class ChatModel:
    def initialize_chat_model(self, api_key, model_name):
        """Initializes the chat model with specified AI model."""
        return ChatOpenAI(openai_api_key=api_key, model_name=model_name, temperature=0.0)

class ChatPrompt:
    @staticmethod
    def craft_prompt():
        template = """You are a legal expert tasked with acting as the best lawyer and contract analyzer. Your task is to thoroughly understand the provided context and answer questions related to legal matters, contracts, and relevant laws. You are also capable of computing and comparing currency values. 
        You must provide accurate responses based solely on the information provided in the context. If the question can be answered as either yes or no, respond with either "Yes." or "No." first and include the explanation in your response.:

        ### CONTEXT
        {context}

        ### QUESTION
        Question: {question}
        """

        prompt = ChatPromptTemplate.from_template(template)
        return prompt