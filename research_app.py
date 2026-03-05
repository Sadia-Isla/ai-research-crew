import streamlit as st
from crewai import Agent, Task, Crew, Process
from crewai.tools import BaseTool
from langchain_groq import ChatGroq
from duckduckgo_search import DDGS 

# --- Page Config ---
st.set_page_config(page_title="AI Research Crew", page_icon="🕵️‍♂️")

# --- Sidebar ---
with st.sidebar:
    st.title("🔑 Configuration")
    groq_key = st.text_input("Enter Groq API Key", type="password")
    st.info("Using Llama 3.3. Memory is disabled to prevent OpenAI errors.")

# --- Custom Search Tool ---
class InternetSearchTool(BaseTool):
    name: str = "internet_search"
    description: str = "Search the internet for information."
    def _run(self, query: str) -> str:
        try:
            with DDGS() as ddgs:
                return str([r for r in ddgs.text(query, max_results=5)])
        except Exception as e:
            return f"Search failed: {str(e)}"

search_tool = InternetSearchTool()

st.title("🕵️‍♂️ Simple Research System")
topic = st.text_input("Research Topic", placeholder="e.g., Quantum Computing Trends")

if st.button("Start Research", type="primary"):
    if not groq_key:
        st.error("Please provide your Groq API Key!")
    elif not topic:
        st.warning("Please enter a topic.")
    else:
        try:
            # Initialize LLM with the latest Llama 3.3 model
            llm = ChatGroq(model="llama-3.3-70b-versatile", groq_api_key=groq_key)

            # 1. Define Agents (Memory set to False to avoid API errors)
            researcher = Agent(
                role='Research Analyst',
                goal=f'Find 3 key facts about {topic}',
                backstory="Expert researcher.",
                tools=[search_tool],
                llm=llm,
                memory=False,
                verbose=True
            )

            writer = Agent(
                role='Technical Writer',
                goal=f'Summarize the research on {topic}',
                backstory="Executive summary expert.",
                llm=llm,
                memory=False,
                verbose=True
            )

            # 2. Define Tasks
            task1 = Task(
                description=f"Research 3 major developments in {topic}.",
                expected_output="3 detailed bullet points.",
                agent=researcher
            )

            task2 = Task(
                description="Write a short Markdown report based on the research.",
                expected_output="A 3-paragraph report.",
                agent=writer
            )

            # 3. Assemble Crew (Memory set to False)
            crew = Crew(
                agents=[researcher, writer],
                tasks=[task1, task2],
                process=Process.sequential,
                memory=False,
                verbose=True
            )

            with st.status("🚀 Working...", expanded=True):
                result = crew.kickoff()

            st.subheader("📝 Final Report")
            st.markdown(str(result))
            
        except Exception as e:
            st.error(f"Error: {str(e)}")
