from langchain_community.chat_message_histories.in_memory import ChatMessageHistory
from langchain_core.messages import HumanMessage, AIMessage

a = """{'input': 'wat is tracing?', 'context': [Document(page_content='tokens an agent used at each stepTo get started, check out the Quick Start Guide.After that, peruse the Concepts Section to better understand the different components involved with tracing.If you want to learn how to accomplish a particular task, check', metadata={'source': 'https://docs.smith.langchain.com/tracing', 'title': 'Tracing Overview | 🦜️🛠️ LangSmith', 'description': 'Tracing is a powerful tool for understanding the behavior of your LLM application.', 'language': 'en'}), Document(page_content='LangSmith has best-in-class tracing capabilities, regardless of whether or not you are using LangChain.Tracing can help you track down issues like:An unexpected end resultWhy an agent is loopingWhy a chain was slower than expectedHow many tokens an agent', metadata={'source': 'https://docs.smith.langchain.com/tracing', 'title': 'Tracing Overview | 🦜️🛠️ LangSmith', 'description': 'Tracing is a powerful tool for understanding the behavior of your LLM application.', 'language': 'en'}), Document(page_content='Tracing Overview | 🦜️🛠️ LangSmith', metadata={'source': 'https://docs.smith.langchain.com/tracing', 'title': 'Tracing Overview | 🦜️🛠️ LangSmith', 'description': 'Tracing is a powerful tool for understanding the behavior of your LLM application.', 'language': 'en'}), Document(page_content='OverviewTracing is a powerful tool for understanding the behavior of your LLM application.', metadata={'source': 'https://docs.smith.langchain.com/tracing', 'title': 'Tracing Overview | 🦜️🛠️ LangSmith', 'description': 'Tracing is a powerful tool for understanding the behavior of your LLM application.', 'language': 'en'})], 'answer': " Tracing is a way to track what your language model is doing as it runs. This can help you understand why something isn't working the way you expected and how to fix it. Tracing can also be used to optimize your code for better performance by identifying areas where resources are being used inefficiently.\n"}"""

print("a")

chat_history = [HumanMessage(content="Can LangSmith help test my LLM applications?"), AIMessage(content="Yes!")]

print(chat_history)


history = ChatMessageHistory()

history.add_user_message("hi! This is human")

history.add_ai_message("whats up human?")

print(history.messages)