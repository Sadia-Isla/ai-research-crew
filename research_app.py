import streamlit as st
import os
import time
from crewai import Agent, Task, Crew, Process
from langchain_openai import ChatOpenAI
from crewai.tools import BaseTool
from duckduckgo_search import DDGS
from fpdf import FPDF

# 1. Page Configuration
st.set_page_config(page_title="AI Research Crew", page_icon="🕵️‍♂️", layout="wide")

st.title("🕵️‍♂️ Multi-Agent Research Intelligence System")
st.markdown("""
### Welcome! This app is your personal AI Research Team.
This system uses **two specialized AI agents** that collaborate to give you a deep report:
1.  **The Researcher:** Scours the live internet for data.
2.  **The Writer:** Turns that data into a professional executive summary.
""")

st.divider()

# 2. Sidebar Setup
with st.sidebar:
    st.header("🔑 Setup Instructions")
    st.markdown("[Get OpenAI API Key](https://platform.openai.com/api-keys)")
    
    user_key = st.text_input("Paste OpenAI API Key & Press Enter:", type="password")
    
    if user_key:
        os.environ["OPENAI_API_KEY"] = user_key
        st.success("✅ Key accepted!")
    else:
        st.info("Waiting for API Key...")

    if st.button("🗑️ Clear Previous Research"):
        st.rerun()

# 3. Enhanced Search Tool (Handles Rate Limits better)
class SearchTool(BaseTool):
    name: str = "internet_search"
    description: str = "Useful for searching the internet about a specific topic."

    def _run(self, query: str) -> str:
        try:
            # Adding a tiny delay to avoid being flagged as a bot
            time.sleep(1) 
            with DDGS() as ddgs:
                results = [r for r in ddgs.text(query, max_results=5)]
                if not results:
                    return "No results found. The search provider might be experiencing high traffic."
                return str(results)
        except Exception as e:
            return f"Search currently unavailable due to: {str(e)}. Please try again in a moment."

# 4. Main Input Section
topic = st.text_input("🔍 What topic should the AI Crew research?", 
                     placeholder="e.g. The impact of Generative AI on Software Engineering 2026")

if st.button("🚀 Start Research Collaboration"):
    if not os.environ.get("OPENAI_API_KEY"):
        st.error("Please enter your OpenAI API Key in the sidebar first!")
    elif not topic:
        st.warning("Please enter a research topic.")
    else:
        try:
            llm = ChatOpenAI(model="gpt-4o", api_key=os.environ["OPENAI_API_KEY"])
            search_tool = SearchTool()

            # Agents
            researcher = Agent(
                role='Lead Research Specialist',
                goal=f'Identify 3-5 specific facts and recent news about {topic}',
                backstory="You are a meticulous researcher. If search fails, you explain why clearly.",
                tools=[search_tool],
                llm=llm,
                verbose=True,
                allow_delegation=False
            )

            writer = Agent(
                role='Senior Content Editor',
                goal='Write a professional summary based ONLY on the research provided',
                backstory="You are an expert at creating executive summaries.",
                llm=llm,
                verbose=True,
                allow_delegation=False
            )

            # Tasks
            task1 = Task(description=f"Deep dive research into {topic}.", expected_output="Key findings list.", agent=researcher)
            task2 = Task(description="Write a 3-paragraph report. If research was blocked, write a guide on the topic instead.", expected_output="Markdown report.", agent=writer)

            # Execution
            crew = Crew(agents=[researcher, writer], tasks=[task1, task2], process=Process.sequential)

            with st.status("👨‍💻 Agents are collaborating...", expanded=True) as status:
                result = crew.kickoff()
                status.update(label="✅ All tasks complete!", state="complete")

            st.divider()
            st.subheader("📋 Final Intelligence Report")
            st.markdown(result)

            # PDF Generation Fix (Handles Special Characters)
            pdf = FPDF()
            pdf.add_page()
            pdf.set_font("Arial", size=12)
            # Remove characters that FPDF doesn't like
            safe_text = str(result).encode('ascii', 'ignore').decode('ascii')
            pdf.multi_cell(0, 10, txt=safe_text)
            pdf_bytes = pdf.output(dest='S').encode('latin-1')

            st.download_button(
                label="📥 Download Report as PDF",
                data=pdf_bytes,
                file_name=f"Report_{topic.replace(' ', '_')}.pdf",
                mime="application/pdf"
            )
            
            with st.expander("🛠️ View Agent Performance Metrics"):
                st.json(crew.usage_metrics)

        except Exception as e:
            st.error(f"Something went wrong: {e}")
