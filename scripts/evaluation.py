import os
import sys
from dotenv import load_dotenv
from llama_index import GPTVectorStoreIndex, SimpleDirectoryReader, LLMPredictor, PromptHelper, ServiceContext
from langchain.chat_models import ChatOpenAI

import os
from langchain.chains import RetrievalQA
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain_community.document_loaders import Docx2txtLoader
from datasets import Dataset

OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')

docx_path = os.path.abspath(os.path.join('..','data', 'Raptor Contract.docx'))

# load documents
loader = Docx2txtLoader(docx_path)
documents = loader.load()
# split the documents into chunks
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
texts = text_splitter.split_documents(documents)

from functools import partial
from langchain.text_splitter import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(chunk_size = 256, chunk_overlap=20, model_name = "gpt-4-1106-preview")
texts  = text_splitter.split_documents(documents)

embeddings = OpenAIEmbeddings()
store = Chroma.from_documents(texts,embeddings, collection_name="contract")

llm = OpenAI(temperature=0)
retriever=store.as_retriever()
chain = RetrievalQA.from_chain_type(llm,retriever=retriever)


questions = [
        "Who are the parties to the Agreement and what are their defined names?",
        "What is the termination notice?",
        "What are the payments to the Advisor under the Agreement?",
        "Can the Agreement or any of its obligations be assigned?",
        "Who owns the IP?",
        "Is there a non-compete obligation to the Advisor?",
        "Can the Advisor charge for meal time?",
        "In which street does the Advisor live?",
        "Is the Advisor entitled to social benefits?",
        "What happens if the Advisor claims compensation based on employment relationship with the Company?",
    ]

ground_truths = [
    ["Cloud Investments Ltd. (“Company”) and Jack Robinson (“Advisor”)"],
    [
        "According to section 4:14 days for convenience by both parties. The Company may terminate without notice if the Advisor refuses or cannot perform the Services or is in breach of any provision of this Agreement."
    ],
    [
        "According to section 6: 1. Fees of $9 per hour up to a monthly limit of $1,500, 2. Workspace expense of $100 per month, 3. Other reasonable and actual expenses if approved by the company in writing and in advance."
    ],
    [
        "Under section 1.1 the Advisor can’t assign any of his obligations without the prior written consent of the Company, 2. Under section 9  the Advisor may not assign the Agreement and the Company may assign it, 3 Under section 9 of the Undertaking the Company may assign the Undertaking."
    ],
    [
        "According to section 4 of the Undertaking (Appendix A), Any Work Product, upon creation, shall be fully and exclusively owned by the Company."
    ],
    [
        "Yes. During the term of engagement with the Company and for a period of 12 months thereafter."
    ],
    ["No. See Section 6.1, Billable Hour doesn’t include meals or travel time. "],
    ["1 Rabin st, Tel Aviv, Israel "],
    [
        "No. According to section 8 of the Agreement, the Advisor is an independent consultant and shall not be entitled to any overtime pay, insurance, paid vacation, severance payments or similar fringe or employment benefits from the Company."
    ],
    [
        "If the Advisor is determined to be an employee of the Company by a governmental authority, payments to the Advisor will be retroactively reduced so that 60% constitutes salary payments and 40% constitutes payment for statutory rights and benefits. The Company may offset any amounts due to the Advisor from any amounts payable under the Agreement. The Advisor must indemnify the Company for any losses or expenses incurred if an employer/employee relationship is determined to exist."
    ],
]

answers = []
contexts = []

# Inference
for query in questions:
  answers.append(chain.invoke(query))
  contexts.append([docs.page_content for docs in retriever.get_relevant_documents(query)])

data = {
    "question": questions,
    "answer": answers,
    "contexts": contexts,
    "ground_truths": ground_truths
}

# Convert dict to dataset
dataset = Dataset.from_dict(data)

from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_recall,
    context_precision,
)

result = evaluate(
    dataset = dataset, 
    metrics=[
        context_precision,
        context_recall,
        faithfulness,
        answer_relevancy,
    ],
)

df = result.to_pandas()
print(df)

