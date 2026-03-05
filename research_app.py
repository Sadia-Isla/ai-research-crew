import streamlit as st
from crewai import Agent, Task, Crew, Process
from langchain_openai import ChatOpenAI
from langchain_community.tools import DuckDuckGoSearchRun
import os
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper

# Use the wrapper to avoid initialization errors
wrapper = DuckDuckGoSearchAPIWrapper()
search_tool = DuckDuckGoSearchRun(api_wrapper=wrapper)
# 1. Setup Streamlit Page Configuration
st.set_page_config(page_title="AI Research Crew", layout="wide")

st.title("🚀 AI Research Crew")
st.markdown("Enter a topic below to have a specialized AI crew research and write a report for you.")

# 2. Sidebar for Configuration (API Key handling)
with st.sidebar:
    st.header("Settings")
    # Priority: Secrets (for live deployment) > User Input > Env Var
    api_key = st.text_input("OpenAI API Key", type="password")
    if not api_key:
        api_key = os.getenv("OPENAI_API_KEY")
    
    if not api_key:
        st.warning("Please provide an OpenAI API Key to proceed.")
        st.stop()

# Initialize Search Tool
search_tool = DuckDuckGoSearchRun()

# 3. Define the Agents
def create_research_crew(topic):
    # LLM configuration
    llm = ChatOpenAI(model_name="gpt-4-turbo", openai_api_key=api_key)

    researcher = Agent(
        role='Senior Research Analyst',
        goal=f'Uncover cutting-edge developments in {topic}',
        backstory="""You are an expert at identifying emerging trends and 
        analyzing complex data. You provide structured and factual insights.""",
        tools=[search_tool],
        llm=llm,
        verbose=True
    )

    writer = Agent(
        role='Tech Content Strategist',
        goal=f'Craft a compelling report on {topic}',
        backstory="""You transform complex research into engaging, 
        easy-to-read articles for a professional audience.""",
        llm=llm,
        verbose=True
    )

    # 4. Define the Tasks
    research_task = Task(
        description=f"Analyze the latest trends and breakthroughs in {topic}. Focus on 2024-2025 data.",
        expected_output="A bullet-point summary of the top 5 key findings.",
        agent=researcher
    )

    write_task = Task(
        description=f"Using the research provided, write a 3-paragraph blog post about {topic}.",
        expected_output="A full blog post in markdown format.",
        agent=writer
    )

    # 5. Assemble the Crew
    crew = Crew(
        agents=[researcher, writer],
        tasks=[research_task, write_task],
        process=Process.sequential
    )
    
    return crew

# 6. UI Logic
topic_input = st.text_input("What topic should we research today?", placeholder="e.g. Multi-agent AI systems")

if st.button("Run Research Crew"):
    if not topic_input:
        st.error("Please enter a topic.")
    else:
        with st.status("🤖 Crew is working...", expanded=True) as status:
            st.write("Initializing agents...")
            crew = create_research_crew(topic_input)
            st.write("Researching and Writing...")
            result = crew.kickoff()
            status.update(label="Research Complete!", state="complete", expanded=False)

        st.subheader("Final Report")
        st.markdown(result)
