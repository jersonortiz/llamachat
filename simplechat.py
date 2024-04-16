import time

from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

llm = Ollama(model="llama2-uncensored")

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are world class technical documentation writer."),
    ("user", "{input}")
])

output_parser = StrOutputParser()

chain = prompt | llm | output_parser

#a = chain.invoke({"input": "how can langsmith help with testing?"})

while True:
    query = input('>>>')
    start = time.perf_counter()
    a = chain.invoke({"input": query})
    print(time.perf_counter() - start)
    print(a)



