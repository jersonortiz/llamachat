from langchain_community.document_loaders.pdf import PyPDFLoader
from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.vectorstores.utils import filter_complex_metadata
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
import time

class Llama:

    llm = None
    prompt = None
    retriever = None
    document_chain = None
    retrieval_chain = None
    history = []
    host = ""

    def __init__(self):
#"export OLLAMA_HOST="
        self.host = "https://4ed7-34-87-23-200.ngrok-free.app"
        self.llm = Ollama( base_url=self.host, model="llama2-uncensored")
        self.prompt = ChatPromptTemplate.from_template("""Answer the following question based only on the provided context:
                                                    <context>
                                                        {context}
                                                    </context>
                                                    Question: {input}"""
                                                  )

        self.document_chain = create_stuff_documents_chain(self.llm, self.prompt)


    def load_pdf(self,pdf_file_path):

        loader = PyPDFLoader(file_path=pdf_file_path)
        return loader.load()

    def load_web(self):
        loader = WebBaseLoader("https://docs.smith.langchain.com/tracing")
        return loader.load()

    def ingest(self ,docs):

        embeddings = OllamaEmbeddings(base_url=self.host, model="all-minilm")

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=256,
                                                       chunk_overlap=20,
                                                       length_function=len,
                                                       is_separator_regex=False, )

        documents = text_splitter.split_documents(docs)
        documents = filter_complex_metadata(documents)

        start = time.perf_counter()
        vector = FAISS.from_documents(documents, embeddings)
        print(vector.index.ntotal)
        print(time.perf_counter()-start)

        self.retriever = vector.as_retriever()
        self.retrieval_chain = create_retrieval_chain(self.retriever, self.document_chain)

    def ask(self, query: str):
        return  self.retrieval_chain.invoke({"input": query})
