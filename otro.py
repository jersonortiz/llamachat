from crewai import Agent, Task, Crew, Process
from langchain_core.tools import Tool

import os
from crewai import Agent, Task, Crew, Process

### OLLAMA (THANKS TO LANGCHAIN)
from crewai import Agent
from crewai_tools import SerperDevTool
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_openai import ChatOpenAI
from langchain_community.llms import Ollama


host = "https://7c8c-35-194-191-67.ngrok-free.app"

ollama_llm = Ollama(base_url=host,model="openhermes")

llm= ChatOpenAI(base_url=host,model="openhermes",api_key="NA")
#openhermes
#mistral

search_tool = SerperDevTool()

research_agent = Agent(
  role='Researcher',
  goal='Find and summarize the latest AI news',
  backstory="""You're a researcher at a large company.
  You're responsible for analyzing data and providing insights
  to the business.""",
  verbose=True,
llm=ollama_llm,


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


#result = crew.kickoff()
#print(result)


# Creating a senior researcher agent with memory and verbose mode
researcher = Agent(
  role='Senior Researcher',
  goal='Uncover groundbreaking technologies in {topic}',
  verbose=True,
  memory=True,
  backstory=(
    "Driven by curiosity, you're at the forefront of"
    "innovation, eager to explore and share knowledge that could change"
    "the world."
  ),
  tools=[search_tool],
  allow_delegation=True,
llm=ollama_llm,
)
# Creating a writer agent with custom tools and delegation capability
writer = Agent(
  role='Writer',
  goal='Narrate compelling tech stories about {topic}',
  verbose=True,
  memory=True,
  backstory=(
    "With a flair for simplifying complex topics, you craft"
    "engaging narratives that captivate and educate, bringing new"
    "discoveries to light in an accessible manner."
  ),
  tools=[search_tool],
  allow_delegation=False,
llm=ollama_llm,
)
# Research task
research_task = Task(
  description=(
    "Identify the next big trend in {topic}."
    "Focus on identifying pros and cons and the overall narrative."
    "Your final report should clearly articulate the key points,"
    "its market opportunities, and potential risks."
  ),
  expected_output='A comprehensive 3 paragraphs long report on the latest AI trends.',
  tools=[search_tool],
  agent=researcher,
)
# Writing task with language model configuration
write_task = Task(
  description=(
    "Compose an insightful article on {topic}."
    "Focus on the latest trends and how it's impacting the industry."
    "This article should be easy to understand, engaging, and positive."
  ),
  expected_output='A 4 paragraph article on {topic} advancements formatted as markdown.',
  tools=[search_tool],
  agent=writer,
  async_execution=False,
  output_file='new-blog-post.md'  # Example of output customization
)


# Forming the tech-focused crew with some enhanced configurations
crew = Crew(
  agents=[researcher, writer],
  tasks=[research_task, write_task],
share_crew=True,
  process=Process.sequential,  # Optional: Sequential task execution is default
  memory=True,
  cache=True,
  max_rpm=100,
    verbose=2,
llm=ollama_llm,
    embedder={
        "provider": "huggingface",
        "config": {
            "model": "mixedbread-ai/mxbai-embed-large-v1",  # https://huggingface.co/mixedbread-ai/mxbai-embed-large-v1
        }
    }

)

result = crew.kickoff(inputs={'topic': 'AI in healthcare'})
print(result)