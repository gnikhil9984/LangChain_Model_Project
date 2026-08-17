from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
import os
from langchain_core.runnables import RunnableParallel

load_dotenv()

LLM = HuggingFaceEndpoint(
    repo_id= "meta-llama/Llama-3.1-8B-Instruct",
    task= "text-generartion",
    huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN")
)

model1 = ChatHuggingFace(llm = LLM)

model2 = ChatHuggingFace(llm = LLM)

template1 = PromptTemplate(
    template= 'Generate shorts notes with examples for my interview preparation on {topic}. \n ',
    input_variables= ['topic']
)

template2 = PromptTemplate(
    template= 'Generate Quiz for my interview preparation based on {topic} and also gives its answer. \n',
    input_variables= ['topic']
)

Final_prompt = PromptTemplate(
    template= "Give me a single document for my {notes} and {quiz} for interview preparation",
    input_variables=['notes','quiz']
)

parser = StrOutputParser()

parallel_chain = RunnableParallel({
    'notes' : template1 | model1 | parser ,
    'quiz'  : template2 | model2 | parser
})

merge_chain = Final_prompt | model1 | parser

chain = parallel_chain | merge_chain

result = chain.invoke({'topic' : 'array'})

print(result)

#chain.get_graph().print_ascii()