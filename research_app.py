import streamlit as st
import os
from crewai import Agent, Task, Crew, Process
from langchain_groq import ChatGroq
from duckduckgo_search import DDGS  # Direct import to fix the error
from langchain.tools import tool

# --- Page Config ---
st.set_page_config(page_title="AI Research Crew", page_icon="🕵️‍♂️", layout="wide")

# --- Custom Search Tool (Fixes the ddgs Error) ---
@tool("internet_search")
def internet_search(query: str):
    """Search the internet for the latest information on a given topic."""
    try:
        with DDGS() as ddgs:
            results = [r for r in ddgs.text(query, max_results=5)]
            return str(results)
    except Exception as e:
        return f"Search failed: {str(e)}"

# --- Sidebar: Configuration ---
with st.sidebar:
    st.title("🔑 Configuration")
    groq_key = st.text_input("Enter Groq API Key", type="password")
    st.info("Get your free key at [console.groq.com](https://console.groq.com)")
    st.markdown("---")
    if st.button("Clear Chat History", use_container_width=True):
        st.rerun()

st.title("🕵️‍♂️ Multi-Agent Research System")
st.markdown("""
This system uses a **Collaborative AI Crew** to research the web and write professional reports.
- **Researcher:** Scans the live web for the latest facts.
- **Writer:** Structures facts into a polished Markdown report.
""")

# --- Input Area ---
topic = st.text_input("Research Topic", placeholder="e.g., Future of AI in Cybersecurity 2025")

if st.button("Start Research Pipeline", type="primary"):
    if not groq_key:
        st.error("Please provide a Groq API Key in the sidebar!")
    elif not topic:
        st.warning("Please enter a research topic.")
    else:
        try:
            # 1. Setup LLM
            llm = ChatGroq(model="llama-3.3-70b-versatile", groq_api_key=groq_key)

            # 2. Define Agents
            researcher = Agent(
                role='Senior Research Analyst',
                goal=f'Find the 3 most important developments regarding {topic}',
                backstory="Expert at deep-web research and fact verification.",
                tools=[internet_search],  # Using our custom tool
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

            # 3. Define Tasks
            research_task = Task(
                description=f"Search the internet and identify 3 key trends or breakthroughs in {topic}.",
                expected_output="A list of 3 detailed bullet points with facts.",
                agent=researcher
            )

            write_task = Task(
                description=f"Using the research provided, write a 3-paragraph executive report in Markdown.",
                expected_output="A well-formatted Markdown report.",
                agent=writer
            )

            # 4. Assemble the Crew
            crew = Crew(
                agents=[researcher, writer],
                tasks=[research_task, write_task],
                process=Process.sequential
            )

            # 5. Execution UI
            with st.status("🚀 Agents are collaborating...", expanded=True) as status:
                st.write("🔍 Researcher is scanning live data...")
                result = str(crew.kickoff()) # Convert result to string
                status.update(label="✅ Success!", state="complete", expanded=False)

            # 6. Display Result
            st.subheader("📝 Final Research Report")
            st.markdown(result)
            
            # 7. Download Feature
            st.download_button(
                label="📥 Download Report (.txt)",
                data=result,
                file_name=f"research_report.txt",
                mime="text/plain"
            )

        except Exception as e:
            st.error(f"An error occurred: {str(e)}")

# --- Footer ---
st.markdown("---")
st.caption("Built with CrewAI, Groq (Llama 3.3), and Streamlit.")
