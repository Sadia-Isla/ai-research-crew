import streamlit as st
import os
from crewai import Agent, Task, Crew, Process
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool

# 1. App UI Setup
st.set_page_config(page_title="AI Multi-Agent Research", layout="wide")
st.title("🤖 Multi-Agent Research Crew")
st.markdown("This system uses a **Researcher** and a **Technical Writer** to collaborate on your topic.")

with st.sidebar:
    st.header("Configuration")
    api_key = st.text_input("OpenAI API Key", type="password")
    if not api_key:
        api_key = st.secrets.get("OPENAI_API_KEY", "")
    
    st.info("Built with CrewAI & GPT-4o")

# 2. Simple, Stable Search Tool
@tool("internet_search")
def internet_search(query: str):
    """Searches the internet for information on a given topic."""
    from duckduckgo_search import DDGS
    with DDGS() as ddgs:
        results = [r for r in ddgs.text(query, max_results=3)]
        return str(results)

# 3. App Logic
topic = st.text_input("What should the agents research?", placeholder="e.g. Next-gen battery technologies")

if st.button("Run Multi-Agent Crew"):
    if not api_key:
        st.error("Please provide an OpenAI API Key.")
    elif topic:
        try:
            # Initialize LLM
            llm = ChatOpenAI(model="gpt-4o", openai_api_key=api_key)

            # Agent 1: The Specialist Researcher
            researcher = Agent(
                role='Senior Research Analyst',
                goal=f'Find the most recent breakthroughs in {topic}',
                backstory="You are an expert at identifying emerging trends and verifying technical data.",
                tools=[internet_search],
                llm=llm,
                verbose=True
            )

            # Agent 2: The Professional Writer
            writer = Agent(
                role='Technical Content Strategist',
                goal=f'Create a high-level executive summary about {topic}',
                backstory="You specialize in transforming complex research into clear, actionable reports.",
                llm=llm,
                verbose=True
            )

            # Task 1: Researching
            research_task = Task(
                description=f"Conduct deep research on {topic}. Identify 3 key 2025-2026 developments.",
                expected_output="A list of key findings with supporting data.",
                agent=researcher
            )

            # Task 2: Writing (uses research from Task 1)
            write_task = Task(
                description=f"Use the researcher's findings to write a professional 3-paragraph report.",
                expected_output="A markdown-formatted executive summary.",
                agent=writer
            )

            # Orchestrate the Crew
            crew = Crew(
                agents=[researcher, writer],
                tasks=[research_task, write_task],
                process=Process.sequential,
                verbose=True
            )

            with st.status("👨‍💻 Agents are collaborating...", expanded=True) as status:
                result = crew.kickoff()
                status.update(label="Collaboration Complete!", state="complete")

            st.subheader("Final Output")
            st.markdown(result)

        except Exception as e:
            st.error(f"Error: {e}")
    else:
        st.warning("Please enter a topic.")
