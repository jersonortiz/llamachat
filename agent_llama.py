from langchain_community.document_loaders.pdf import PyPDFLoader
from langchain_community.llms import Ollama
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate, \
    PromptTemplate
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.vectorstores.utils import filter_complex_metadata
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
import time
from langchain.chains import create_history_aware_retriever
from langchain_core.prompts import MessagesPlaceholder
from langchain.tools.retriever import create_retriever_tool
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.agents import AgentExecutor
from langchain.agents import AgentExecutor, tool
from langchain.agents import initialize_agent
from langchain_community.chat_models import ChatOllama



class Llama:

    llm = None
    prompt = None
    retriever = None
    retrieval_chain = None
    vector= None
    retriever = None

    def __init__(self, history = True):
        self.with_history = history
        self.host = "https://e28e-35-184-25-249.ngrok-free.app"
        self.llm = Ollama(base_url=self.host, model="openhermes")
        self.prompt = self.get_prompt()
        self.chat_history = []

    def create_search_tool(self):

        search = TavilySearchResults()
        retriever_tool = create_retriever_tool(
            self.retriever,
            "langsmith_search",
            "Search for information about LangSmith. For any questions about LangSmith, you must use this tool!",
        )

        tools = [search, retriever_tool]

        prompt = ChatPromptTemplate.from_messages([SystemMessagePromptTemplate(prompt=PromptTemplate(input_variables=[], template='You are a helpful assistant')),
                                                   MessagesPlaceholder(variable_name='chat_history', optional=True),
                                                   HumanMessagePromptTemplate(prompt=PromptTemplate(input_variables=['input'], template='{input}')),
                                                   MessagesPlaceholder(variable_name='agent_scratchpad')
                                                   ])

        llm_with_tools = self.llm.bind_tools(tools)

        from langchain.agents.format_scratchpad.openai_tools import (
            format_to_openai_tool_messages,
        )
        from langchain.agents.output_parsers.openai_tools import OpenAIToolsAgentOutputParser

        agent = (
                {
                    "input": lambda x: x["input"],
                    "agent_scratchpad": lambda x: format_to_openai_tool_messages(
                        x["intermediate_steps"]
                    ),
                }
                | prompt
                | llm_with_tools
                | OpenAIToolsAgentOutputParser()
        )

        #agent = create_tool_calling_agent(self.llm, tools, prompt)

        agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

        res = agent_executor.invoke({"input": "hi!"})

        print(res)


    def get_prompt(self):
        if self.with_history:
            return ChatPromptTemplate.from_messages([
                ("system", "Answer the user's questions based on the below context:\n\n{context}"),
                MessagesPlaceholder(variable_name="chat_history"),
                ("user", "{input}"),
            ])

        return ChatPromptTemplate.from_template("""Answer the following question based only on the provided context:
        <context>{context}</context>
        Question: {input}"""
                                                )

    def load_pdf(self, pdf_file_path):
        loader = PyPDFLoader(file_path=pdf_file_path)
        return loader.load()

    def load_web(self, url='https://docs.smith.langchain.com/tracing'):
        loader = WebBaseLoader(url)
        return loader.load()

    def ingest(self, docs):

        embeddings = OllamaEmbeddings( base_url=self.host, model="all-minilm")
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=256,
                                                       chunk_overlap=20,
                                                       length_function=len,
                                                       is_separator_regex=False, )

        documents = text_splitter.split_documents(docs)
        documents = filter_complex_metadata(documents)

        start = time.perf_counter()
        self.vector = FAISS.from_documents(documents, embeddings)

        print(time.perf_counter() - start)
        self.etriever = self.vector.as_retriever()

    def create_retriever_chain(self):

        document_chain = create_stuff_documents_chain(self.llm, self.prompt)
        if self.with_history:
            self.retrieval_chain = self.history_retriever_chain(self.retriever)
        self.retrieval_chain = create_retrieval_chain(self.retriever , document_chain)

    def history_retriever_chain(self, retriever):
        contextualize_q_prompt = ChatPromptTemplate.from_messages([
                                MessagesPlaceholder(variable_name="chat_history"),
                                ("user", "{input}"),
                                ("system", "Given the above conversation, generate a search query to look up to get information relevant to the conversation")
                            ])
        return create_history_aware_retriever(self.llm, retriever, contextualize_q_prompt)

    def ask(self, query: str):

        if self.with_history:

            response= self.retrieval_chain.invoke({
                "chat_history": self.chat_history,
                "input": query
            })
            self.chat_history.extend([HumanMessage(content=query), AIMessage(content=response["answer"])])
            return response

        return self.retrieval_chain.invoke({"input": query})