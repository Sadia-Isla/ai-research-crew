import streamlit as st
import os
import time
from crewai import Agent, Task, Crew, Process
from langchain_openai import ChatOpenAI
from crewai.tools import BaseTool
from duckduckgo_search import DDGS
from fpdf import FPDF

# --- 1. Page Configuration ---
st.set_page_config(page_title="AI Research Crew", page_icon="🕵️‍♂️", layout="wide")

# --- 2. Professional Header & Instructions ---
st.title("🕵️‍♂️ Multi-Agent Research Intelligence System")
st.markdown("""
### Welcome! This isn't just a standard AI chat.
This system uses **Agentic Collaboration** to find and summarize information. 
When you enter a topic, two specialized AI agents start a "workflow":
1.  **The Researcher Agent:** Connects to the live internet to find 3-5 factual breakthroughs.
2.  **The Writer Agent:** Takes those findings and synthesizes them into a professional executive report.
""")

st.divider()

# --- 3. Sidebar: User Onboarding & API Key ---
with st.sidebar:
    st.header("🔑 Getting Started")
    st.markdown("""
    **New to this? Follow these steps:**
    
    1. **Get a Key:** You need an OpenAI API key. Create one here:
    [OpenAI API Dashboard](https://platform.openai.com/api-keys)
    
    2. **Enter Key:** Paste it below and press **Enter**.
    """)
    
    user_key = st.text_input("Enter OpenAI API Key:", type="password")
    
    if user_key:
        os.environ["OPENAI_API_KEY"] = user_key
        st.success("✅ API Key Activated!")
    else:
        st.warning("⚠️ API Key required to run agents.")

    st.divider()
    st.markdown("### 💡 Pro-Tip")
    st.info("The agents work best on specific topics like 'Impact of AI on 2026 job markets' rather than broad words like 'Science'.")

# --- 4. Stable Search Tool Class ---
class SearchTool(BaseTool):
    name: str = "internet_search"
    description: str = "Useful for searching the internet about a specific topic."

    def _run(self, query: str) -> str:
        try:
            time.sleep(1) # Tiny delay to prevent rate limits
            with DDGS() as ddgs:
                results = [r for r in ddgs.text(query, max_results=5)]
                return str(results) if results else "No recent data found. The agent will use internal knowledge."
        except Exception:
            return "Search temporarily unavailable. The agent will provide a general overview instead."

# --- 5. Main Research Input ---
topic = st.text_input("🔍 What would you like the AI Crew to investigate?", 
                     placeholder="e.g. Current state of solid-state batteries 2026")

if st.button("🚀 Start Agent Collaboration"):
    if not os.environ.get("OPENAI_API_KEY"):
        st.error("Please enter your API Key in the sidebar first!")
    elif not topic:
        st.warning("Please enter a research topic to begin.")
    else:
        try:
            llm = ChatOpenAI(model="gpt-4o", api_key=os.environ["OPENAI_API_KEY"])
            search_tool = SearchTool()

            # Agents
            researcher = Agent(
                role='Lead Research Specialist',
                goal=f'Find the 3-5 most important recent facts about {topic}',
                backstory="You are a meticulous researcher. You focus on accuracy and ignore unreliable sources.",
                tools=[search_tool],
                llm=llm,
                verbose=True,
                allow_delegation=False
            )

            writer = Agent(
                role='Senior Content Editor',
                goal='Write a professional executive summary based on the research provided',
                backstory="You are an expert at making complex information clear for busy professionals.",
                llm=llm,
                verbose=True,
                allow_delegation=False
            )

            # Tasks
            task1 = Task(description=f"Researching {topic}...", expected_output="Key findings list.", agent=researcher)
            task2 = Task(description="Writing final report...", expected_output="Executive Markdown report.", agent=writer)

            # Execution
            crew = Crew(agents=[researcher, writer], tasks=[task1, task2], process=Process.sequential)

            with st.status("👨‍💻 Agents are collaborating...", expanded=True) as status:
                result = crew.kickoff()
                status.update(label="✅ Research & Writing Complete!", state="complete")

            # Result Display
            st.divider()
            st.subheader("📋 Final Intelligence Report")
            st.markdown(result)

            # PDF Download Logic
            pdf = FPDF()
            pdf.add_page()
            pdf.set_font("Arial", size=12)
            safe_text = str(result).encode('ascii', 'ignore').decode('ascii')
            pdf.multi_cell(0, 10, txt=safe_text)
            pdf_bytes = pdf.output(dest='S').encode('latin-1')

            st.download_button(
                label="📥 Download Report as PDF",
                data=pdf_bytes,
                file_name=f"Report_{topic.replace(' ', '_')}.pdf",
                mime="application/pdf"
            )
            
            with st.expander("🛠️ View Performance Metrics"):
                st.write("Usage Stats for this run:")
                st.json(crew.usage_metrics)

        except Exception as e:
            st.error(f"Something went wrong: {e}")
