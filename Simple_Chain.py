from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
import os

load_dotenv()

LLM = HuggingFaceEndpoint(
    repo_id= "meta-llama/Llama-3.1-8B-Instruct",
    task= "text-generartion",
    huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN")
)

model = ChatHuggingFace(llm = LLM)

template = PromptTemplate(
    template=' explain the {topic} with its all types,features and its working \n',
    input_variables= ['topic']
)

parser = StrOutputParser()

chain = template | model | parser

result = chain.invoke({'topic' : 'array'})

print(result)

chain.get_graph().print_ascii()

