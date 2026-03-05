import streamlit as st
import os
from crewai import Agent, Task, Crew, Process
from langchain_openai import ChatOpenAI
from crewai.tools import BaseTool
from duckduckgo_search import DDGS
from fpdf import FPDF

# 1. Page Configuration & Custom Styling
st.set_page_config(page_title="AI Research Crew", page_icon="🕵️‍♂️", layout="wide")

# Title and User-Friendly Description
st.title("🕵️‍♂️ Multi-Agent Research Intelligence System")
st.markdown("""
### Welcome! This app is your personal AI Research Team.
Instead of one AI just answering a question, this system uses **two specialized AI agents** that talk to each other to give you a deeper, more factual report:
1.  **The Researcher:** Scours the live internet for the latest news and data.
2.  **The Writer:** Takes that raw data and turns it into a professional, easy-to-read executive summary.
""")

st.divider()

# 2. Sidebar: Onboarding & Setup
with st.sidebar:
    st.header("🔑 Setup Instructions")
    st.markdown("""
    **Step 1:** You need an OpenAI API Key.
    If you don't have one, you can create it here:
    [Get OpenAI API Key](https://platform.openai.com/api-keys)
    
    **Step 2:** Enter your key below and press **Enter**.
    """)
    
    # The 'Enter' button logic is handled by Streamlit's text_input automatically 
    # when the user presses Enter, but we will add a confirmation message.
    user_key = st.text_input("Paste OpenAI API Key here:", type="password")
    
    if user_key:
        os.environ["OPENAI_API_KEY"] = user_key
        st.success("✅ Key accepted! You are ready to go.")
    else:
        st.info("Waiting for API Key...")

    st.divider()
    st.markdown("### 💡 How to use")
    st.write("1. Enter a topic (e.g., 'Future of EV batteries').")
    st.write("2. Click 'Start Research'.")
    st.write("3. Watch the agents collaborate in real-time.")
    st.write("4. Download your final report as a PDF.")

# 3. Custom Search Tool Class
class SearchTool(BaseTool):
    name: str = "internet_search"
    description: str = "Useful for searching the internet about a specific topic."

    def _run(self, query: str) -> str:
        with DDGS() as ddgs:
            results = [r for r in ddgs.text(query, max_results=3)]
            return str(results)

# 4. Main Input Section
topic = st.text_input("🔍 What topic do you want the AI Crew to research?", 
                     placeholder="e.g. Impact of AI on the 2026 Job Market")

if st.button("🚀 Start Research Collaboration"):
    if not os.environ.get("OPENAI_API_KEY"):
        st.error("Please enter your OpenAI API Key in the sidebar first!")
    elif not topic:
        st.warning("Please enter a research topic to begin.")
    else:
        try:
            # Initialize LLM & Tool
            llm = ChatOpenAI(model="gpt-4o", api_key=os.environ["OPENAI_API_KEY"])
            search_tool = SearchTool()

            # Agent Definitions
            researcher = Agent(
                role='Lead Research Specialist',
                goal=f'Find the 3 most important recent facts about {topic}',
                backstory="You are a meticulous researcher. You only use credible data and ignore the noise.",
                tools=[search_tool],
                llm=llm,
                verbose=True,
                allow_delegation=False
            )

            writer = Agent(
                role='Senior Content Editor',
                goal='Write a professional summary based on the research provided',
                backstory="You are famous for making complex information easy to understand for busy executives.",
                llm=llm,
                verbose=True,
                allow_delegation=False
            )

            # Task Definitions
            task1 = Task(
                description=f"Research the latest news and data regarding {topic}.",
                expected_output="A list of key findings with brief explanations.",
                agent=researcher
            )

            task2 = Task(
                description="Synthesize the research findings into a professional 3-paragraph executive summary.",
                expected_output="A well-formatted Markdown report.",
                agent=writer
            )

            # The Crew
            crew = Crew(
                agents=[researcher, writer],
                tasks=[task1, task2],
                process=Process.sequential
            )

            # Execution UI
            with st.status("👨‍💻 The Agents are collaborating (Researching -> Writing)...", expanded=True) as status:
                result = crew.kickoff()
                status.update(label="✅ All tasks complete!", state="complete")

            # --- Results Display ---
            st.divider()
            st.subheader("📋 Final Intelligence Report")
            st.markdown(result)

            # PDF Generation
            pdf = FPDF()
            pdf.add_page()
            pdf.set_font("Arial", size=12)
            # FPDF can't handle some special characters, so we clean it slightly
            clean_text = str(result).encode('latin-1', 'ignore').decode('latin-1')
            pdf.multi_cell(0, 10, txt=clean_text)
            pdf_bytes = pdf.output(dest='S').encode('latin-1')

            st.download_button(
                label="📥 Download Report as PDF",
                data=pdf_bytes,
                file_name=f"Research_Report_{topic.replace(' ', '_')}.pdf",
                mime="application/pdf"
            )
            
            # Resume-friendly usage stats
            with st.expander("🛠️ View Agent Performance Metrics"):
                st.json(crew.usage_metrics)

        except Exception as e:
            st.error(f"Something went wrong: {e}")
            st.info("Check your API key balance or try a more specific topic.")
