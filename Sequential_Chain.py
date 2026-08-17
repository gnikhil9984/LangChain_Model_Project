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

template1 = PromptTemplate(
    template=' Deatiled explain on {topic} with its all types,features and its working \n',
    input_variables= ['topic']
)

template2 = PromptTemplate(
    template= " Give me only Important Concept of topic in 5 lines. \n {text}",
    input_variables= ['text']
)

parser = StrOutputParser()

chain = template1 | model | parser| template2 | model | parser

result = chain.invoke({'topic': 'array'})

print(result)

chain.get_graph().print_ascii()