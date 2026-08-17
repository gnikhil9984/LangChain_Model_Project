from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
import os
from langchain_core.runnables import RunnableParallel, RunnableBranch, RunnableLambda
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel , Field
from typing import  Literal

load_dotenv()

LLM = HuggingFaceEndpoint(
    repo_id= "meta-llama/Llama-3.1-8B-Instruct",
    task= "text-generartion",
    huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN")
)

model = ChatHuggingFace(llm = LLM)

parser = StrOutputParser()

class feedback(BaseModel):
    sentiment : Literal['positive' , 'negative'] = Field(description= 'give the sentiment of the feedback')

parser2 = PydanticOutputParser(pydantic_object= feedback)

template1 = PromptTemplate(
    template= 'Classify the feedback text that which type of impact is shown by the whole {sentence}. \n {format_instruction}',
    input_variables= ['sentence'],
    partial_variables= {'format_instruction' : parser2.get_format_instructions()}
)

classifier_chain = template1 | model | parser2

prompt1 = PromptTemplate(
    template= 'Write an appropiate response to this positive feedback \n {feedback}',
    input_variables= ['feedback']
)

prompt2 = PromptTemplate(
    template= 'Write an appropiate response to this negative feedback \n {feedback}',
    input_variables= ['feedback']
)

branch_chain = RunnableBranch(
    (lambda x : x.sentiment == 'positive' , prompt1 | model | parser),
    (lambda x : x.sentiment == 'negative' , prompt2 | model | parser),
    RunnableLambda(lambda x : "could not find sentiment")
)

chain = classifier_chain | branch_chain

result = chain.invoke({'sentence' : 'The heavy, suffocating gray of the afternoon pressed down upon the fractured landscape like a shroud of damp ash. Nothing grew here; the soil had long since surrendered to a bitter, saline crust that choked every desperate root and blighted every fledgling stem. A relentless, whining wind swept across the barren flats, carrying with it the sharp, acrid scent of decay and the hollow rattle of dead stalks scraping against one another in a dreary chorus of despair. There was no warmth left in the world, only a persistent, marrow-deep chill that numbed the fingers and clouded the mind with a dull, suffocating dread. '})

print(result)



