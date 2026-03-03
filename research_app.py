import streamlit as st
import os
from crewai import Agent, Task, Crew, Process
from langchain_groq import ChatGroq
from langchain_community.tools import DuckDuckGoSearchRun

# --- Page Config ---
st.set_page_config(page_title="AI Research Crew", page_icon="🕵️‍♂️", layout="wide")

# --- Sidebar: API Keys ---
with st.sidebar:
    st.title("🔑 Configuration")
    groq_key = st.text_input("Enter Groq API Key", type="password")
    st.info("Get your key at [console.groq.com](https://console.groq.com)")
    st.markdown("---")
    st.write("🕵️‍♂️ **Agent 1:** Researcher")
    st.write("✍️ **Agent 2:** Writer")

st.title("🕵️‍♂️ Multi-Agent Research System")
st.markdown("Enter a topic, and my AI agents will research the web and write a report for you.")

# --- Input Area ---
topic = st.text_input("What do you want to research?", placeholder="e.g., Quantum Computing in 2025")

if st.button("Start Research"):
    if not groq_key:
        st.error("Please provide a Groq API Key!")
    elif not topic:
        st.warning("Please enter a topic.")
    else:
        try:
            # 1. Setup LLM & Search Tool
            llm = ChatGroq(model="llama-3.3-70b-versatile", groq_api_key=groq_key)
            search_tool = DuckDuckGoSearchRun()

            # 2. Define Agents
            researcher = Agent(
                role='Senior Research Lead',
                goal=f'Uncover the 3 most groundbreaking facts about {topic}',
                backstory="An expert analyst known for finding hidden trends and verifying facts.",
                tools=[search_tool],
                llm=llm,
                verbose=True,
                allow_delegation=False
            )

            writer = Agent(
                role='Technical Content Strategist',
                goal=f'Write a professional 3-paragraph summary about {topic}',
                backstory="A world-class writer who simplifies complex data into engaging reports.",
                llm=llm,
                verbose=True,
                allow_delegation=False
            )

            # 3. Define Tasks
            research_task = Task(
                description=f"Identify the 3 latest and most impactful developments in {topic}.",
                expected_output="A list of 3 bullet points with detailed facts.",
                agent=researcher
            )

            write_task = Task(
                description=f"Using the research, write a professional report for an executive audience.",
                expected_output="A 3-paragraph report in Markdown format.",
                agent=writer
            )

            # 4. Assemble the Crew
            crew = Crew(
                agents=[researcher, writer],
                tasks=[research_task, write_task],
                process=Process.sequential
            )

            # 5. Execution UI
            with st.status("🚀 Agents are working...", expanded=True) as status:
                st.write("🔍 Researcher is scanning the web...")
                result = crew.kickoff()
                status.update(label="✅ Research Complete!", state="complete", expanded=False)

            # 6. Display Result
            st.subheader("📝 Final Research Report")
            st.markdown(result)
            
            # 7. Export Feature
            st.download_button(
                label="Download Report as Text",
                data=str(result),
                file_name=f"{topic.replace(' ', '_')}_report.txt",
                mime="text/plain"
            )

        except Exception as e:
            st.error(f"An error occurred: {str(e)}")

# --- Footer ---
st.markdown("---")
st.caption("Built with CrewAI, Groq, and Streamlit.")
