from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain_core.runnables import RunnableSequence, RunnableParallel, RunnablePassthrough, RunnableLambda, RunnableBranch
import os

load_dotenv()

LLM = HuggingFaceEndpoint(
    repo_id= "meta-llama/Llama-3.1-8B-Instruct",
    task= "text-generartion",
    huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN")
)

model = ChatHuggingFace(llm = LLM)

def word_count(text):
    return len(text.split())

prompt1 = PromptTemplate(
    template='Write a detailed report on {topic}',
    input_variables=['topic']
)

prompt2 = PromptTemplate(
    template='Summarize the following text under 150 words\n {text}',
    input_variables=['text']
)

parser = StrOutputParser()

report_gen_chain = prompt1 | model | parser

branch_chain = RunnableBranch(
    (lambda x: len(x.split())>500, prompt2 | model | parser),
    RunnablePassthrough()
)
count_word = RunnableLambda(word_count)

f_chain = RunnableSequence(report_gen_chain, branch_chain)

final_chain = RunnableParallel({
    'report': RunnableSequence(report_gen_chain, branch_chain),
    'word_count': RunnableSequence(f_chain , count_word)
})

result = final_chain.invoke({'topic':'Russia vs Ukraine'})

final_result = """{} \n word count - {}""".format(result['report'], result['word_count'])

print(final_result)

