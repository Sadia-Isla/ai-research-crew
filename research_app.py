import streamlit as st
import os
from crewai import Agent, Task, Crew, Process
from langchain_openai import ChatOpenAI
from crewai.tools import BaseTool
from duckduckgo_search import DDGS

# 1. Page Config
st.set_page_config(page_title="Multi-Agent AI", layout="wide")
st.title("🚀 Multi-Agent Research System")

# 2. Key Handling (Crucial Fix for the "Native Provider" Error)
with st.sidebar:
    st.header("Authentication")
    user_key = st.text_input("OpenAI API Key", type="password")
    if user_key:
        # This injects the key into the system environment where CrewAI looks for it
        os.environ["OPENAI_API_KEY"] = user_key
    elif "OPENAI_API_KEY" not in os.environ:
        st.warning("Please enter your API Key to enable the agents.")

# 3. Stable Search Tool (Class-based is the most stable for Pydantic)
class SearchTool(BaseTool):
    name: str = "internet_search"
    description: str = "Useful for searching the internet about a specific topic."

    def _run(self, query: str) -> str:
        with DDGS() as ddgs:
            results = [r for r in ddgs.text(query, max_results=3)]
            return str(results)

# 4. App Logic
topic = st.text_input("Enter a topic for the agents to collaborate on:")

if st.button("Start Collaboration"):
    if not os.environ.get("OPENAI_API_KEY"):
        st.error("API Key is missing.")
    elif topic:
        try:
            # Explicitly define the LLM
            llm = ChatOpenAI(model="gpt-4o", api_key=os.environ["OPENAI_API_KEY"])
            search = SearchTool()

            # Agent 1: Researcher
            researcher = Agent(
                role='Researcher',
                goal=f'Find 3 key facts about {topic}',
                backstory="Expert at finding and verifying technical information.",
                tools=[search],
                llm=llm,
                verbose=True,
                allow_delegation=False # Keeps it simple/stable
            )

            # Agent 2: Writer
            writer = Agent(
                role='Writer',
                goal='Write a professional summary',
                backstory="Expert at turning raw data into beautiful reports.",
                llm=llm,
                verbose=True,
                allow_delegation=False
            )

            # Tasks
            task1 = Task(description=f"Research {topic}", expected_output="Bullet points", agent=researcher)
            task2 = Task(description="Summarize findings", expected_output="3 paragraph report", agent=writer)

            # Crew
            crew = Crew(
                agents=[researcher, writer],
                tasks=[task1, task2],
                process=Process.sequential
            )

            with st.status("🤖 Agents are working...") as status:
                result = crew.kickoff()
                status.update(label="Done!", state="complete")

            st.markdown(result)

        except Exception as e:
            st.error(f"Execution Error: {e}")
