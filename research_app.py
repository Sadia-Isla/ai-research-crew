import streamlit as st
import os
from crewai import Agent, Task, Crew, Process
from langchain_openai import ChatOpenAI
# Fixed: Importing from langchain_core as it is already installed in your logs
from langchain_core.tools import tool

# --- 1. Custom Tool Definition ---
class SearchTools():
    @tool("search_internet")
    def search_internet(query: str):
        """Search the internet about a given topic and return relevant results."""
        from duckduckgo_search import DDGS
        with DDGS() as ddgs:
            # Safely fetch results
            results = [r for r in ddgs.text(query, max_results=5)]
            return results

# --- 2. Streamlit UI ---
st.set_page_config(page_title="AI Research Crew", layout="wide")
st.title("🚀 AI Research Crew")

with st.sidebar:
    st.header("Setup")
    api_key = st.text_input("OpenAI API Key", type="password")
    st.info("Make sure 'langchain' and 'duckduckgo-search' are in requirements.txt")

topic = st.text_input("What would you like to research?")

# --- 3. Execution Logic ---
if st.button("Run Research"):
    if not api_key:
        st.error("Please provide an OpenAI API Key.")
    elif not topic:
        st.warning("Please enter a research topic.")
    else:
        try:
            # Initialize the LLM
            llm = ChatOpenAI(model="gpt-4o", openai_api_key=api_key)
            
            # Initialize our custom tool
            internet_search_tool = SearchTools.search_internet

            # Define Researcher Agent
            researcher = Agent(
                role='Senior Researcher',
                goal=f'Find comprehensive and factual info on {topic}',
                backstory="An expert analyst with a focus on 2025-2026 developments.",
                tools=[internet_search_tool],
                llm=llm,
                verbose=True,
                allow_delegation=False
            )

            # Define the Research Task
            task = Task(
                description=f"Analyze {topic}. Provide a deep dive into current trends and future outlook.",
                expected_output="A well-formatted markdown report with bullet points.",
                agent=researcher
            )

            # Assemble the Crew
            crew = Crew(
                agents=[researcher], 
                tasks=[task], 
                process=Process.sequential,
                verbose=True
            )

            with st.spinner("The Crew is researching... please wait."):
                result = crew.kickoff()
                st.success("Research Complete!")
                st.markdown("---")
                st.markdown(result)
                
        except Exception as e:
            st.error(f"Execution Error: {str(e)}")
            st.info("Check your OpenAI API key and balance.")
