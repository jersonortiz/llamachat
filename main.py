import time
#from chatllama import Llama
from convllama import Llama

if __name__ == "__main__":
    chat = Llama()
    docs  = chat.loadWeb()
    chat.ingest(docs)

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