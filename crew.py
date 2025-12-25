from crewai import Agent, Crew, Process, Task, LLM
from crewai.project import CrewBase, agent, crew, task
from crewai.agents.agent_builder.base_agent import BaseAgent
from typing import List
import requests
import json
from crewai.knowledge.source.string_knowledge_source import BaseKnowledgeSource
from typing import List, Dict
from pydantic import Field
from pydantic import BaseModel
from crewai_tools import ScrapeElementFromWebsiteTool

    
llm = LLM(
    model="ollama/phi3-mini:latest",
    base_url="http://localhost:11434"
)


@CrewBase
class LatestAiDevelopment():
    # Initialize the tool
    scrape_tool = ScrapeElementFromWebsiteTool(
        website_url="https://swr.bootcss.com/",
        css_element=".subheading-anchor"
    )

    # Define an agent that uses the tool
    web_scraper_agent = Agent(
        role="Web Scraper",
        goal="Extract specific information from websites",
        backstory="An expert in web scraping who can extract targeted content from web pages.",
        tools=[scrape_tool],
        verbose=True,
        llm=llm,
    )

    # Example task to extract headlines from a news website
    scrape_task = Task(
        description="Extract the main headlines from the swr homepage. Use the CSS selector '.subheading-anchor' to target the headline elements.",
        expected_output="A list of the main headlines from swr.",
        agent=web_scraper_agent,
    )

    # Create and run the crew
    analysis_crew = Crew(agents=[web_scraper_agent], tasks=[scrape_task])

    @crew
    def crew(self) -> Crew:
        return self.analysis_crew
