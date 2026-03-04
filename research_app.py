import streamlit as st
import os
import sys
import io
import re
from crewai import Agent, Task, Crew, Process
from crewai.tools import BaseTool
from langchain_groq import ChatGroq
from duckduckgo_search import DDGS 

# --- Page Config ---
st.set_page_config(page_title="AI Research Crew", page_icon="🕵️‍♂️", layout="wide")

# --- Step 1: Fix OpenAI Requirement ---
os.environ["OPENAI_API_KEY"] = "NA"

# --- Step 2: Custom Search Tool (Pydantic v2 Compatible) ---
class InternetSearchTool(BaseTool):
    name: str = "internet_search"
    description: str = "Search the internet for the latest information on a given topic."

    def _run(self, query: str) -> str:
        """Execute the search logic using DuckDuckGo."""
        try:
            with DDGS() as ddgs:
                results = [r for r in ddgs.text(query, max_results=5)]
                return str(results)
        except Exception as e:
            return f"Search failed: {str(e)}"

search_tool = InternetSearchTool()

# --- Helper: Capture Agent Thoughts ---
class TerminalCapture(list):
    def __enter__(self):
        self._stdout = sys.stdout
        sys.stdout = self._stringio = io.StringIO()
        return self

    def __exit__(self, *args):
        self.extend(self._stringio.getvalue().splitlines())
        del self._stringio
        sys.stdout = self._stdout

# --- Sidebar: Configuration ---
with st.sidebar:
    st.title("🔑 Configuration")
    groq_key = st.text_input("Enter Groq API Key", type="password")
    st.info("Get your free key at [console.groq.com](https://console.groq.com)")
    st.markdown("---")
    if st.button("Reset Session", use_container_width=True):
        st.rerun()

st.title("🕵️‍♂️ Multi-Agent Research System")
st.markdown("""
An autonomous **AI Pipeline** where specialized agents collaborate in real-time.
- **Researcher:** Browses the live web for factual data.
- **Writer:** Synthesizes data into an executive Markdown report.
""")

# --- Input Area ---
topic = st.text_input("Research Topic", placeholder="e.g., Future of Fusion Energy 2026")

if st.button("Start Research Pipeline", type="primary"):
    if not groq_key:
        st.error("Please provide a Groq API Key!")
    elif not topic:
        st.warning("Please enter a topic.")
    else:
        try:
            # 3. Initialize LLM (Groq)
            llm = ChatGroq(model="llama-3.3-70b-versatile", groq_api_key=groq_key)

            # 4. Define Agents
            researcher = Agent(
                role='Senior Research Analyst',
                goal=f'Find the 3 most important developments regarding {topic}',
                backstory="Expert at deep-web research and fact verification.",
                tools=[search_tool],
                llm=llm,
                verbose=True, # Enable logging
                allow_delegation=False
            )

            writer = Agent(
                role='Professional Content Strategist',
                goal=f'Write an executive summary based on the research provided for {topic}',
                backstory="Specialist in technical writing and executive summaries.",
                llm=llm,
                verbose=True, # Enable logging
                allow_delegation=False
            )

            # 5. Define Tasks
            research_task = Task(
                description=f"Search the internet and identify 3 key trends in {topic}.",
                expected_output="A list of 3 detailed bullet points with facts.",
                agent=researcher
            )

            write_task = Task(
                description=f"Using the research provided, write a 3-paragraph executive report in Markdown.",
                expected_output="A well-formatted Markdown report.",
                agent=writer
            )

            # 6. Assemble the Crew
            crew = Crew(
                agents=[researcher, writer],
                tasks=[research_task, write_task],
                process=Process.sequential,
                manager_llm=llm
            )

            # 7. Execution UI with Live Logs
            st.subheader("🧠 Agent Thought Process")
            log_container = st.empty() # Placeholder for live logs
            
            with st.status("🚀 Agents are collaborating...", expanded=True) as status:
                # Capture the "Internal Monologue" of the agents
                with TerminalCapture() as logs:
                    result = crew.kickoff()
                
                # Clean and display logs to show the "Writer" actually worked
                clean_logs = "\n".join([line for line in logs if "Working on" in line or "Action" in line])
                log_container.code(clean_logs if clean_logs else "Pipeline execution successful.")
                status.update(label="✅ Success!", state="complete", expanded=False)

            # 8. Display Result
            st.subheader("📝 Final Research Report")
            st.markdown(str(result))
            
            # 9. Download Feature
            st.download_button(
                label="📥 Download Report (.txt)",
                data=str(result),
                file_name=f"research_report.txt",
                mime="text/plain"
            )

        except Exception as e:
            st.error(f"An error occurred: {str(e)}")

# --- Footer ---
st.markdown("---")
st.caption("Built with CrewAI, Groq, and Streamlit.")
