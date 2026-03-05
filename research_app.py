import streamlit as st
import os
from crewai import Agent, Task, Crew, Process
from crewai.tools import BaseTool
from langchain_groq import ChatGroq
from duckduckgo_search import DDGS 

# --- Page Config ---
st.set_page_config(page_title="AI Research Crew", page_icon="🕵️‍♂️", layout="wide")

# --- Sidebar: Configuration ---
with st.sidebar:
    st.title("🔑 API Settings")
    groq_key = st.text_input("Enter Groq API Key", type="password")
    openai_key = st.text_input("Enter OpenAI API Key", type="password")
    st.info("Groq runs the Llama model. OpenAI handles Agent Memory.")
    st.markdown("---")
    if st.button("Reset Session"):
        st.rerun()

# --- Custom Search Tool ---
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

st.title("🕵️‍♂️ Multi-Agent Research System")
st.markdown("Autonomous research pipeline powered by **Llama 3.1 (Groq)** and **OpenAI Memory**.")

topic = st.text_input("Research Topic", placeholder="e.g., Next-gen Battery Technology 2026")

if st.button("Start Research Pipeline", type="primary"):
    if not groq_key or not openai_key:
        st.error("Please provide both API Keys in the sidebar!")
    elif not topic:
        st.warning("Please enter a topic.")
    else:
        try:
            # Set environment variable for CrewAI memory
            os.environ["OPENAI_API_KEY"] = openai_key

            # Initialize Groq LLM - Using llama-3.1-70b-versatile for stability
            llm = ChatGroq(model="llama-3.1-70b-versatile", groq_api_key=groq_key)

            # Define Agents
            researcher = Agent(
                role='Senior Research Analyst',
                goal=f'Find the 3 most important developments regarding {topic}',
                backstory="Expert at deep-web research and fact verification.",
                tools=[search_tool],
                llm=llm,
                verbose=True,
                memory=True
            )

            writer = Agent(
                role='Professional Content Strategist',
                goal=f'Write an executive summary based on the research provided for {topic}',
                backstory="Specialist in technical writing and executive summaries.",
                llm=llm,
                verbose=True,
                memory=True
            )

            # Define Tasks
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

            # Assemble the Crew
            crew = Crew(
                agents=[researcher, writer],
                tasks=[research_task, write_task],
                process=Process.sequential,
                memory=True,
                verbose=True
            )

            # Execution UI
            with st.status("🚀 Agents are collaborating...", expanded=True) as status:
                st.write("🔍 Researching and Writing...")
                result = crew.kickoff()
                status.update(label="✅ Success!", state="complete", expanded=False)

            # Display Result
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

st.markdown("---")
st.caption("Built with CrewAI, Groq, and OpenAI.")
