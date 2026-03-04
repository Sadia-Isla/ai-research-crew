import streamlit as st
import os
import sys
import io
from crewai import Agent, Task, Crew, Process
from crewai.tools import BaseTool
from langchain_groq import ChatGroq
from duckduckgo_search import DDGS 

# --- Page Config ---
st.set_page_config(page_title="AI Research Crew", page_icon="🕵️‍♂️", layout="wide")

# --- Step 1: Ensure no OpenAI environment variable exists ---
if "OPENAI_API_KEY" in os.environ:
    del os.environ["OPENAI_API_KEY"]

# --- Step 2: Custom Search Tool ---
class InternetSearchTool(BaseTool):
    name: str = "internet_search"
    description: str = "Search the internet for the latest information on a given topic."

    def _run(self, query: str) -> str:
        try:
            with DDGS() as ddgs:
                results = [r for r in ddgs.text(query, max_results=5)]
                return str(results)
        except Exception as e:
            return f"Search failed: {str(e)}"

search_tool = InternetSearchTool()

# --- Sidebar: Configuration ---
with st.sidebar:
    st.title("🔑 Configuration")
    groq_key = st.text_input("Enter Groq API Key", type="password")
    st.info("Get your free key at [console.groq.com](https://console.groq.com)")
    st.markdown("---")
    if st.button("Reset Session", use_container_width=True):
        st.rerun()

st.title("🕵️‍♂️ Multi-Agent Research System")
st.markdown("Autonomous research pipeline powered by **Groq**.")

# --- Input Area ---
topic = st.text_input("Research Topic", placeholder="e.g., Future of Fusion Energy 2026")

if st.button("Start Research Pipeline", type="primary"):
    if not groq_key:
        st.error("Please provide a Groq API Key!")
    elif not topic:
        st.warning("Please enter a topic.")
    else:
        try:
            # 3. Initialize LLM
            llm = ChatGroq(model="llama-3.3-70b-versatile", groq_api_key=groq_key)

            # 4. Define Agents 
            # Note: We specify the llm for each agent explicitly
            researcher = Agent(
                role='Senior Research Analyst',
                goal=f'Find the 3 most important developments regarding {topic}',
                backstory="Expert at deep-web research and fact verification.",
                tools=[search_tool],
                llm=llm,
                verbose=True,
                allow_delegation=False
            )

            writer = Agent(
                role='Professional Content Strategist',
                goal=f'Write an executive summary based on the research provided for {topic}',
                backstory="Specialist in technical writing and executive summaries.",
                llm=llm,
                verbose=True,
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
            # manager_llm=llm and embedder configuration stops OpenAI calls
            crew = Crew(
                agents=[researcher, writer],
                tasks=[research_task, write_task],
                process=Process.sequential,
                manager_llm=llm,
                # This tells CrewAI not to use the default OpenAI embedder
                embedder={
                    "provider": "google",
                    "config": {"model": "models/embedding-001", "task_type": "retrieval_document"}
                } if False else None # Setting to None is safest for pure Groq
            )

            # 7. Execution UI
            with st.status("🚀 Agents are collaborating...", expanded=True) as status:
                st.write("🔍 Researching and Writing...")
                result = crew.kickoff()
                status.update(label="✅ Success!", state="complete", expanded=False)

            # 8. Display Result
            st.subheader("📝 Final Research Report")
            st.markdown(str(result))
            
            st.download_button(
                label="📥 Download Report (.txt)",
                data=str(result),
                file_name=f"research_report.txt",
                mime="text/plain"
            )

        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            if "OPENAI_API_KEY" in str(e):
                st.info("💡 Pro Tip: CrewAI is trying to find an OpenAI key. Ensure the 'Crew' object has 'manager_llm' set.")

# --- Footer ---
st.markdown("---")
st.caption("Built with CrewAI, Groq, and Streamlit.")
