import os
import textwrap
import chromadb
import langchain
import openai
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import WebBaseLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.indexes import VectorstoreIndexCreator
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.vectorstores import Chroma
from watermark import watermark


print(watermark())


def print_response(response: str):
    print("\n".join(textwrap.wrap(response, width=100)))


os.environ["OPENAI_API_KEY"] = "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"


model = OpenAI(temperature = 0)

print(model("hello who are you?"))





