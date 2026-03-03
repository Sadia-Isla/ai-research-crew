import streamlit as st
import os
from crewai import Agent, Task, Crew, Process
from langchain_groq import ChatGroq
from langchain_community.tools import DuckDuckGoSearchRun

# --- Page Config ---
st.set_page_config(page_title="AI Research Crew", page_icon="🕵️‍♂️", layout="wide")

# --- Sidebar: Configuration ---
with st.sidebar:
    st.title("🔑 Configuration")
    groq_key = st.text_input("Enter Groq API Key", type="password")
    st.info("Get your free key at [console.groq.com](https://console.groq.com)")
    st.markdown("---")
    st.write("🕵️‍♂️ **Agent 1:** Senior Researcher")
    st.write("✍️ **Agent 2:** Content Strategist")
    
    if st.button("Clear Chat", use_container_width=True):
        st.rerun()

st.title("🕵️‍♂️ Multi-Agent Research System")
st.markdown("""
This system uses a **Collaborative AI Crew** to research the web and write professional reports.
1. The **Researcher** finds the latest facts via DuckDuckGo.
2. The **Writer** structures those facts into a polished Markdown report.
""")

# --- Input Area ---
topic = st.text_input("What topic should the agents research?", placeholder="e.g., Impact of Generative AI on Software Engineering 2025")

if st.button("Start Research Pipeline", type="primary"):
    if not groq_key:
        st.error("Please provide a Groq API Key in the sidebar!")
    elif not topic:
        st.warning("Please enter a research topic.")
    else:
        try:
            # 1. Setup LLM & Search Tool
            # Using Llama 3.3 70B for high-quality reasoning
            llm = ChatGroq(model="llama-3.3-70b-versatile", groq_api_key=groq_key)
            search_tool = DuckDuckGoSearchRun()

            # 2. Define Agents
            researcher = Agent(
                role='Senior Research Analyst',
                goal=f'Find the 3 most important and recent developments regarding {topic}',
                backstory="""You are an expert at deep-web research. You know how to filter 
                noise from facts and provide high-quality, cited information.""",
                tools=[search_tool],
                llm=llm,
                verbose=True,
                allow_delegation=False
            )

            writer = Agent(
                role='Professional Content Strategist',
                goal=f'Write an executive summary based on the research provided for {topic}',
                backstory="""You specialize in taking raw data and turning it into 
                professional, easy-to-read Markdown reports for business leaders.""",
                llm=llm,
                verbose=True,
                allow_delegation=False
            )

            # 3. Define Tasks
            research_task = Task(
                description=f"Search the internet and identify 3 key trends or breakthroughs in {topic}.",
                expected_output="A list of 3 detailed bullet points with facts and context.",
                agent=researcher
            )

            write_task = Task(
                description=f"Using the research provided, write a 3-paragraph executive report in Markdown.",
                expected_output="A well-formatted Markdown report with a title and clear sections.",
                agent=writer
            )

            # 4. Assemble the Crew
            # Sequential process ensures the writer waits for the researcher's data
            crew = Crew(
                agents=[researcher, writer],
                tasks=[research_task, write_task],
                process=Process.sequential
            )

            # 5. Execution UI with Status Updates
            with st.status("🚀 Agents are collaborating...", expanded=True) as status:
                st.write("🔍 Researcher is scanning live web data...")
                result = crew.kickoff()
                status.update(label="✅ Research & Writing Complete!", state="complete", expanded=False)

            # 6. Display Final Result
            st.subheader("📝 Final Research Report")
            st.markdown(result)
            
            # 7. Download Feature
            st.download_button(
                label="📥 Download Report (.txt)",
                data=str(result),
                file_name=f"{topic.lower().replace(' ', '_')}_report.txt",
                mime="text/plain"
            )

        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            if "ddgs" in str(e).lower():
                st.info("💡 Hint: Ensure 'duckduckgo-search' is installed in your requirements.txt")

# --- Footer ---
st.markdown("---")
st.caption("Built with CrewAI, Groq (Llama 3.3), and Streamlit.")
