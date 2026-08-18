from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain_core.runnables import RunnableSequence
import os

load_dotenv()

LLM = HuggingFaceEndpoint(
    repo_id= "meta-llama/Llama-3.1-8B-Instruct",
    task= "text-generartion",
    huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN")
)

model = ChatHuggingFace(llm = LLM)

template1 = PromptTemplate(
    template= 'write a joke about the {topic} that must be 10 line .\n',
    input_variables= ['topic']
)

template2 = PromptTemplate(
    template= 'Summary of joke in 1 line {text} \n.',
    input_variables= ['text']
)

parser = StrOutputParser()

chain1 = RunnableSequence(template1, model, parser)

chain2 = RunnableSequence(template2, model ,parser)

final_chain = RunnableSequence(chain1 ,chain2)

result = final_chain.invoke({'topic': 'Technology Fail'})

print(result)  

#with the help of runnablesequence
chain3 = RunnableSequence(template1 , model, parser, template2 , model , parser)

print(chain3.invoke({'topic' : "Technology Fail"}))