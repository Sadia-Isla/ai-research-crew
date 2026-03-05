import streamlit as st
import os
from crewai import Agent, Task, Crew, Process
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool

# 1. Page Config
st.set_page_config(page_title="AI Research Crew", layout="wide")
st.title("🚀 AI Research Crew")

# 2. Sidebar Setup
with st.sidebar:
    st.header("Settings")
    api_key = st.text_input("OpenAI API Key", type="password")
    if not api_key:
        # Check Streamlit secrets if text input is empty
        api_key = st.secrets.get("OPENAI_API_KEY", "")
    
    if not api_key:
        st.warning("Please enter an OpenAI API Key to start.")

# 3. Robust Tool Definition
@tool("internet_search")
def internet_search(query: str):
    """Search the internet for the latest information on a topic."""
    from duckduckgo_search import DDGS
    with DDGS() as ddgs:
        # We convert the generator to a list to ensure it's serializable
        results = [r for r in ddgs.text(query, max_results=5)]
        return str(results)

# 4. Main App Logic
topic = st.text_input("Enter a research topic:", placeholder="e.g., Future of Mars Colonization")

if st.button("Start Research"):
    if not api_key:
        st.error("API Key is missing!")
    elif not topic:
        st.error("Please enter a topic.")
    else:
        try:
            # Initialize LLM
            llm = ChatOpenAI(model="gpt-4o", openai_api_key=api_key)
            
            # --- THE FIX IS HERE ---
            # We pass the tool directly in a list. 
            # CrewAI 1.x+ is strict about the tool being a valid BaseTool instance.
            agent_tools = [internet_search]

            # Define Agent
            researcher = Agent(
                role='Lead Research Analyst',
                goal=f'Provide a deep-dive report on {topic}',
                backstory="You are an expert researcher known for finding factual, up-to-date data.",
                tools=agent_tools, # Updated to use the validated list
                llm=llm,
                verbose=True,
                allow_delegation=False
            )

            # Define Task
            research_task = Task(
                description=f"Research {topic}. Focus on major breakthroughs in 2025 and 2026.",
                expected_output="A detailed markdown report with at least 3 sections.",
                agent=researcher
            )

            # Run Crew
            crew = Crew(
                agents=[researcher],
                tasks=[research_task],
                verbose=True
            )

            with st.status("🤖 Crew is investigating...", expanded=True) as status:
                result = crew.kickoff()
                status.update(label="Research Complete!", state="complete")

            st.markdown("### Final Report")
            st.markdown(result)

        except Exception as e:
            # Enhanced error reporting for debugging
            st.error(f"An error occurred: {e}")
            if "validation error" in str(e).lower():
                st.info("This is a Pydantic versioning conflict. Ensure your requirements.txt is updated.")
