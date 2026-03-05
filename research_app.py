import streamlit as st
import os
from crewai import Agent, Task, Crew, Process
from langchain_openai import ChatOpenAI

# Attempt to import the search tool safely
try:
    from langchain_community.tools import DuckDuckGoSearchRun
    from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
    HAS_SEARCH = True
except ImportError:
    HAS_SEARCH = False

st.set_page_config(page_title="AI Research Crew", page_icon="🕵️‍♂️")

# --- UI Header ---
st.title("🕵️‍♂️ AI Research Crew")

# --- Sidebar Configuration ---
with st.sidebar:
    st.header("Configuration")
    api_key = st.text_input("OpenAI API Key", type="password")
    
    if not HAS_SEARCH:
        st.error("⚠️ 'duckduckgo-search' is not installed. Please add it to requirements.txt")

# --- Main Logic ---
topic = st.text_input("What do you want the AI to research?", placeholder="e.g. The future of Quantum Computing")

if st.button("Start Research"):
    if not api_key:
        st.warning("Please enter your OpenAI API Key in the sidebar.")
    elif not HAS_SEARCH:
        st.error("App cannot run without search dependencies.")
    elif topic:
        try:
            # 1. Initialize Tools & LLM
            wrapper = DuckDuckGoSearchAPIWrapper()
            search_tool = DuckDuckGoSearchRun(api_wrapper=wrapper)
            llm = ChatOpenAI(model="gpt-4o", openai_api_key=api_key)

            # 2. Define Agent
            researcher = Agent(
                role='Lead Researcher',
                goal=f'Find the latest information about {topic}',
                backstory="You are a meticulous researcher with a knack for finding hidden details.",
                tools=[search_tool],
                llm=llm,
                verbose=True
            )

            # 3. Define Task
            task = Task(
                description=f"Conduct a comprehensive search on {topic} and summarize key findings.",
                expected_output="A detailed 4-paragraph report.",
                agent=researcher
            )

            # 4. Run Crew
            crew = Crew(agents=[researcher], tasks=[task], verbose=True)
            
            with st.spinner("Analyzing... this may take a minute."):
                result = crew.kickoff()
                st.success("Research Complete!")
                st.markdown(result)

        except Exception as e:
            st.error(f"An error occurred during execution: {e}")
    else:
        st.info("Please enter a topic.")
