from langchain_community.tools.tavily_search import TavilySearchResults

search = TavilySearchResults(tavily_api_key= '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')

print(search.invoke("what is the weather in SF"))