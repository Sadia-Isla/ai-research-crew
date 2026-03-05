import streamlit as st
import os
from crewai import Agent, Task, Crew, Process
from crewai.tools import tool
from langchain_groq import ChatGroq
from duckduckgo_search import DDGS 

# --- CRITICAL: Bypass the OpenAI initialization check with a "valid-length" dummy key ---
os.environ["OPENAI_API_KEY"] = "sk-111111111111111111111111111111111111111111111111"

st.set_page_config(page_title="Reliable Research Crew", page_icon="🕵️‍♂️")

with st.sidebar:
    st.title("🔑 API Settings")
    groq_key = st.text_input("Enter Groq API Key", type="password")
    if st.button("Reset Session"):
        st.rerun()

# --- Correct Tool Definition using @tool decorator ---
@tool("internet_search")
def internet_search(query: str):
    """Search the internet for information on a given topic and return results."""
    try:
        with DDGS() as ddgs:
            results = [r for r in ddgs.text(query, max_results=3)]
            return str(results)
    except Exception as e:
        return f"Search error: {e}"

st.title("🕵️‍♂️ Multi-Agent Research System")
topic = st.text_input("Research Topic", placeholder="e.g., Llama 3.3 release features")

if st.button("Start Research", type="primary"):
    if not groq_key:
        st.error("Please provide your Groq API Key!")
    else:
        try:
            # 1. Initialize Groq LLM
            llm = ChatGroq(
                model="llama-3.3-70b-versatile", 
                groq_api_key=groq_key,
                temperature=0.7
            )

            # 2. Define Agents
            researcher = Agent(
                role='Senior Research Analyst',
                goal=f'Identify 3 key facts about {topic}',
                backstory="Expert at finding accurate info quickly.",
                tools=[internet_search], # Now correctly decorated
                llm=llm,
                memory=False,
                verbose=True,
                allow_delegation=False
            )

            writer = Agent(
                role='Content Writer',
                goal=f'Summarize the findings for {topic}',
                backstory="Technical writer who makes things easy to read.",
                llm=llm,
                memory=False,
                verbose=True,
                allow_delegation=False
            )

            # 3. Define Tasks
            task1 = Task(
                description=f"Find 3 major developments regarding {topic}.",
                expected_output="3 bullet points with facts.",
                agent=researcher
            )

            task2 = Task(
                description="Format the research into a short Markdown report.",
                expected_output="A 2-paragraph report.",
                agent=writer
            )

            # 4. Assemble and Run
            crew = Crew(
                agents=[researcher, writer],
                tasks=[task1, task2],
                process=Process.sequential,
                memory=False,
                verbose=True
            )

            with st.status("🔍 Researching...", expanded=True):
                result = crew.kickoff()
                st.success("Done!")

            st.subheader("📝 Final Report")
            st.markdown(str(result))
            
        except Exception as e:
            st.error(f"Execution Error: {str(e)}")

st.caption("Built with CrewAI and Groq (Llama 3.3)")
