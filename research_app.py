import streamlit as st
import os
from crewai import Agent, Task, Crew, Process
from langchain_openai import ChatOpenAI
from langchain.tools import tool

# --- 1. Custom Tool Definition (Bypasses the buggy wrapper) ---
class SearchTools():
    @tool("search_internet")
    def search_internet(query: str):
        """Search the internet about a given topic and return relevant results."""
        from duckduckgo_search import DDGS
        with DDGS() as ddgs:
            results = [r for r in ddgs.text(query, max_results=5)]
            return results

# --- 2. Streamlit UI ---
st.set_page_config(page_title="AI Research Crew", layout="wide")
st.title("🚀 AI Research Crew")

with st.sidebar:
    st.header("Setup")
    api_key = st.text_input("OpenAI API Key", type="password")
    st.info("Ensure 'duckduckgo-search' is in your requirements.txt")

topic = st.text_input("What would you like to research?")

# --- 3. Execution ---
if st.button("Run Research"):
    if not api_key:
        st.error("Please provide an OpenAI API Key.")
    elif not topic:
        st.warning("Please enter a research topic.")
    else:
        try:
            llm = ChatOpenAI(model="gpt-4o", openai_api_key=api_key)
            
            # Initialize our custom tool
            internet_search_tool = SearchTools.search_internet

            researcher = Agent(
                role='Senior Researcher',
                goal=f'Find comprehensive info on {topic}',
                backstory="Expert at sifting through internet noise to find facts.",
                tools=[internet_search_tool],
                llm=llm,
                verbose=True,
                allow_delegation=False
            )

            task = Task(
                description=f"Research {topic} and provide a summary of the latest 2025-2026 news.",
                expected_output="A structured markdown report.",
                agent=researcher
            )

            crew = Crew(agents=[researcher], tasks=[task], verbose=True)

            with st.spinner("Crew is searching..."):
                result = crew.kickoff()
                st.success("Done!")
                st.markdown(result)
                
        except Exception as e:
            st.error(f"Execution Error: {str(e)}")
            st.info("Tip: If you see 'ddgs' errors, click 'Reboot App' in Streamlit Cloud.")
