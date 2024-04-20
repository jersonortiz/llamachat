import time
#from chatllama import Llama
#from convllama import LlamaHistory


if __name__ == "__main__":

    """
    chat = Llama()
    docs = chat.load_web(url = 'https://docs.smith.langchain.com/overview"')
    chat.ingest(docs)
    chat.create_search_tool()
    """


    chat = Llama(history=False)
    docs  = chat.load_web()
    chat.ingest(docs)
    chat.create_retriever_chain()

    while True:
        query = input('>>>')
        if query == "/exit":
            break
        start = time.perf_counter()
        response = chat.ask(query)
        print(response["answer"])
        print(time.perf_counter() - start)

    """ 
    start = time.perf_counter()
    response = chat.ask("wat is tracing?")
    print(response["answer"])
    print(time.perf_counter() - start)
    """