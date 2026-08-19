from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
import os

load_dotenv()

LLM = HuggingFaceEndpoint(
    repo_id= "meta-llama/Llama-3.1-8B-Instruct",
    task= "text-generartion",
    huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN")
)

model = ChatHuggingFace(llm = LLM)

loader = TextLoader("Paper_Id_175_NPDSM26_Manuscript_Word.txt")
docs = loader.load()

template = PromptTemplate(
    template= "Give me a Summary of My uploaded documents \n. {Summary}",
    input_variables= ['Summary']
)

parser = StrOutputParser()

chain = template | model | parser

result = chain.invoke({'Summary' : docs[0].page_content})

print(result)