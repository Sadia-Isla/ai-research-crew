import streamlit as st
from crewai import Agent, Task, Crew, Process
from langchain_openai import ChatOpenAI
from crewai.tools import BaseTool
from duckduckgo_search import DDGS

# 1. Professional UI Setup
st.set_page_config(page_title="AI Multi-Agent Research", layout="wide")
st.title("🤖 Multi-Agent Research Intelligence")
st.markdown("A professional multi-agent system demonstrating **Sequential Task Delegation**.")

# 2. Define a Stable Custom Tool (Pydantic-safe)
class InternetSearchTool(BaseTool):
    name: str = "internet_search"
    description: str = "Searches the internet for the latest information on a given topic."

    def _run(self, query: str) -> str:
        with DDGS() as ddgs:
            results = [r for r in ddgs.text(query, max_results=3)]
            return str(results)

# 3. Sidebar for API Keys
with st.sidebar:
    st.header("Authentication")
    api_key = st.text_input("OpenAI API Key", type="password")
    if not api_key:
        api_key = st.secrets.get("OPENAI_API_KEY", "")

# 4. Main App Logic
topic = st.text_input("Enter Research Topic:", placeholder="e.g., Sustainable Aviation Fuel trends 2026")

if st.button("🚀 Run Multi-Agent Crew"):
    if not api_key:
        st.error("Please provide an OpenAI API Key.")
    elif topic:
        try:
            # Initialize LLM
            llm = ChatOpenAI(model="gpt-4o", openai_api_key=api_key)
            search_tool = InternetSearchTool()

            # --- AGENT 1: THE RESEARCHER ---
            researcher = Agent(
                role='Lead Research Analyst',
                goal=f'Identify 3 major breakthroughs in {topic}',
                backstory="Expert at deep-web data retrieval and technical verification.",
                tools=[search_tool],
                llm=llm,
                verbose=True
            )

            # --- AGENT 2: THE WRITER ---
            writer = Agent(
                role='Technical Writer',
                goal='Synthesize research into a professional executive summary',
                backstory="Specializes in translating technical data for stakeholders.",
                llm=llm,
                verbose=True
            )

            # --- TASKS ---
            task1 = Task(
                description=f"Research the latest 2025-2026 news regarding {topic}.",
                expected_output="A list of 3 key findings with details.",
                agent=researcher
            )

            task2 = Task(
                description="Write a 3-paragraph professional report based on the findings.",
                expected_output="A markdown formatted report.",
                agent=writer
            )

            # --- EXECUTION ---
            crew = Crew(
                agents=[researcher, writer],
                tasks=[task1, task2],
                process=Process.sequential,
                verbose=True
            )

            with st.status("👨‍💻 Agents collaborating...", expanded=True) as status:
                result = crew.kickoff()
                status.update(label="Analysis Complete!", state="complete")

            st.subheader("Final Intelligence Report")
            st.markdown(result)

        except Exception as e:
            st.error(f"Error: {e}")
    else:
        st.warning("Please enter a topic.")
