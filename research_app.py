import streamlit as st
from crewai import Agent, Task, Crew, Process
from crewai_tools import ScrapeWebsiteTool, SerperDevTool
from langchain_openai import ChatOpenAI
from langchain.tools import tool

# 1. App Configuration
st.set_page_config(page_title="Multi-Agent AI Research", layout="wide")
st.title("🤖 Multi-Agent Research Team")

with st.sidebar:
    st.header("Authentication")
    api_key = st.text_input("OpenAI API Key", type="password")
    st.info("This app uses GPT-4o and Multi-Agent orchestration (CrewAI) to conduct deep research.")

# 2. Stable Search Tool Implementation
# We use a simple @tool decorator which is highly stable for resumes
@tool("internet_search")
def internet_search(query: str):
    """Searches the internet for information on a given topic."""
    from duckduckgo_search import DDGS
    with DDGS() as ddgs:
        results = [r for r in ddgs.text(query, max_results=3)]
        return str(results)

# 3. Main Logic
topic = st.text_input("Enter a complex topic for research:", placeholder="e.g., The impact of Quantum Computing on Cybersecurity")

if st.button("🚀 Start Multi-Agent Process"):
    if not api_key:
        st.warning("Please enter your OpenAI API Key in the sidebar.")
    elif topic:
        try:
            # Initialize the LLM (GPT-4o is best for multi-agent reasoning)
            llm = ChatOpenAI(model="gpt-4o", openai_api_key=api_key)

            # --- AGENT 1: THE RESEARCHER ---
            researcher = Agent(
                role='Senior Research Lead',
                goal=f'Uncover groundbreaking information about {topic}',
                backstory="You are a world-class researcher. You excel at finding the latest news and verifying facts.",
                tools=[internet_search],
                llm=llm,
                verbose=True
            )

            # --- AGENT 2: THE WRITER ---
            writer = Agent(
                role='Technical Content Strategist',
                goal=f'Summarize the research into a professional report about {topic}',
                backstory="You specialize in taking complex data and turning it into clear, executive summaries.",
                llm=llm,
                verbose=True
            )

            # --- DEFINE TASKS ---
            research_task = Task(
                description=f"Research the top 3 latest breakthroughs in {topic}. Provide sources.",
                expected_output="A list of 3 key findings with brief descriptions.",
                agent=researcher
            )

            writing_task = Task(
                description=f"Write a 3-paragraph executive summary based on the research. Use Markdown.",
                expected_output="A professionally formatted report in Markdown.",
                agent=writer
            )

            # --- ORCHESTRATE THE CREW ---
            crew = Crew(
                agents=[researcher, writer],
                tasks=[research_task, writing_task],
                process=Process.sequential, # Researcher finishes, then Writer starts
                verbose=True
            )

            with st.spinner("Agents are collaborating... please wait."):
                result = crew.kickoff()
                st.success("Collaboration Successful!")
                st.markdown("---")
                st.subheader("Final Intelligence Report")
                st.markdown(result)

        except Exception as e:
            st.error(f"Error: {str(e)}")
    else:
        st.info("Provide a topic to begin.")
