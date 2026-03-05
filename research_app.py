import streamlit as st
import os
from crewai import Agent, Task, Crew, Process, LLM
from duckduckgo_search import DDGS 

# 1. COMPLETELY WIPE the OpenAI key from the environment
if "OPENAI_API_KEY" in os.environ:
    del os.environ["OPENAI_API_KEY"]

st.set_page_config(page_title="Simplified Research", page_icon="🕵️‍♂️")

with st.sidebar:
    st.title("🔑 Configuration")
    groq_key = st.text_input("Enter Groq API Key", type="password")

# Custom Search Tool
class InternetSearchTool:
    def search(self, query: str):
        with DDGS() as ddgs:
            return str([r for r in ddgs.text(query, max_results=3)])

st.title("🕵️‍♂️ Fast Research System")
topic = st.text_input("Topic", placeholder="e.g. Llama 4 release date")

if st.button("Run Research", type="primary"):
    if not groq_key:
        st.error("Missing Groq Key!")
    else:
        try:
            # Use the Native CrewAI LLM wrapper to bypass OpenAI checks
            my_llm = LLM(
                model="groq/llama-3.3-70b-versatile",
                api_key=groq_key
            )

            # Researcher Agent
            researcher = Agent(
                role='Researcher',
                goal=f'Find 3 facts about {topic}',
                backstory="You are a helpful research assistant.",
                llm=my_llm,
                tools=[InternetSearchTool().search],
                allow_delegation=False,
                verbose=True
            )

            # Simple Task
            task = Task(
                description=f"Summarize 3 key findings about {topic}.",
                expected_output="A short bulleted list.",
                agent=researcher
            )

            # Simple Crew (No Memory, No Manager)
            crew = Crew(
                agents=[researcher],
                tasks=[task],
                verbose=True
            )

            with st.spinner("Processing..."):
                result = crew.kickoff()
                st.success("Done!")
                st.markdown(str(result))
                
        except Exception as e:
            st.error(f"Error: {str(e)}")
