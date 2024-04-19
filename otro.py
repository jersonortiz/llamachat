from crewai import Agent, Task, Crew, Process
from langchain_core.tools import Tool

import os
from crewai import Agent, Task, Crew, Process

### OLLAMA (THANKS TO LANGCHAIN)
from crewai import Agent
from crewai_tools import SerperDevTool
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.llms import Ollama


host = "https://7c8c-35-194-191-67.ngrok-free.app"

ollama_llm = Ollama(base_url=host,model="llama2:13b")


search_tool = SerperDevTool()

research_agent = Agent(
  role='Researcher',
  goal='Find and summarize the latest AI news',
  backstory="""You're a researcher at a large company.
  You're responsible for analyzing data and providing insights
  to the business.""",
  verbose=True,
llm=ollama_llm

)

task = Task(
  description='Find on internet and summarize the latest AI news',
  expected_output='A bullet list summary of the top 5 most important AI news',
  agent=research_agent,
    tools=[search_tool]

)


crew = Crew(
    agents=[research_agent],
    tasks=[task],
    verbose=2
)




result = crew.kickoff()
print(result)

