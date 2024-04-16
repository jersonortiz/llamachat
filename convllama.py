from langchain_community.document_loaders.pdf import PyPDFLoader
from langchain_community.llms import Ollama
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
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
from langchain_community.chat_message_histories.in_memory import ChatMessageHistory
from langchain.memory import ConversationBufferMemory

class Llama:

    llm = None
    prompt = None
    retriever = None
    document_chain = None
    retrieval_chain = None
    chat_history = []
    host= ""



    def __init__(self):
        self.host = "https://4ed7-34-87-23-200.ngrok-free.app"
        self.llm = Ollama(base_url=self.host, model="llama2-uncensored")
        self.prompt = ChatPromptTemplate.from_messages([
        ("system", "Answer the user's questions based on the below context:\n\n{context}"),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
    ])

        self.document_chain = create_stuff_documents_chain(self.llm, self.prompt)


    def loadpdf(self, pdf_file_path):
        loader = PyPDFLoader(file_path=pdf_file_path)
        return loader.load()


    def loadWeb(self):
        loader = WebBaseLoader("https://docs.smith.langchain.com/tracing")
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
        vector = FAISS.from_documents(documents, embeddings)
        print(vector.index.ntotal)
        print(time.perf_counter() - start)

        self.retriever = vector.as_retriever()
        #self.historychain()
        self.retrieval_chain = create_retrieval_chain(self.retriever , self.document_chain)

    def historychain(self):
        contextualize_q_prompt = ChatPromptTemplate.from_messages([
                                MessagesPlaceholder(variable_name="chat_history"),
                                ("user", "{input}"),
                                ("system", "Given the above conversation, generate a search query to look up to get information relevant to the conversation")
                            ])
        self.retriever_chain = create_history_aware_retriever(self.llm, self.retriever, contextualize_q_prompt)

    def ask(self, query: str):
        #return self.retrieval_chain.invoke({"input": query})
        #chat_history = [HumanMessage(content="Can LangSmith help test my LLM applications?"), AIMessage(content="Yes!")]



        response= self.retrieval_chain.invoke({
            "chat_history": self.chat_history,
            "input": query
        })
        self.chat_history.extend([HumanMessage(content=query), AIMessage(content=response["answer"])])
        return response